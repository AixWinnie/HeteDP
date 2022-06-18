
import dgl
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import FeatDP
from data.dataprepro import get_label, get_node_mask
from utils import micro_macro_f1_score, set_random_seed, get_device
from utilsDP import feat_noise as add_noise

def featureLearning(g, data, args):
    num_heads = 8
    num_hidden = args.hidden_feat
    dropout = args.dropout_feat
    weight_decay = args.weight_decay_feat
    delta = torch.tensor(args.delta)
    epsilon = args.epsilon_feat

    score = micro_macro_f1_score
    features_all = {}
    device = get_device(args.gpu_id)
    #learning each node embedding
    for n in g.ntypes:
        print("Learning the "+n+" type embedding...")
        set_random_seed(args.seed)
        num_classes = data.num_classes(n)
        degress = torch.zeros(g.num_nodes(n))
        gs = [dgl.metapath_reachable_graph(g, metapath) for metapath in data.metapaths(n)]
        for i in range(len(gs)):
            gs[i] = dgl.add_self_loop(dgl.remove_self_loop(gs[i]))
            degress +=  gs[i].in_degrees()   
            gs[i] = gs[i].to(device)

        #check feat
        if len(g.nodes[n].data) == 0 or 'feat' not in g.nodes[n].data:
            features = None
            fdp_model = FeatDP(
                len(gs), g.num_nodes(n), num_hidden, num_hidden, num_classes, num_heads, dropout)
        else:
            features = g.nodes[n].data['feat']
            fdp_model = FeatDP(
            len(gs), g.num_nodes(n), features.shape[1], num_hidden, num_classes, num_heads, dropout)

        #check label
        if len(g.nodes[n].data) == 0 or 'label' not in g.nodes[n].data:
            labels = get_label(degress, g.num_nodes(n), num_classes)
        else:
            labels = g.nodes[n].data['label']
        labels = labels.to(device)

        #check dataset spliting
        if len(g.nodes[n].data) == 0 or 'train_mask' not in g.nodes[n].data:
            train_mask, val_mask, test_mask = get_node_mask(g.num_nodes(n), args.seed)
        else:
            train_mask = g.nodes[n].data['train_mask']
            val_mask = g.nodes[n].data['val_mask']
            test_mask = g.nodes[n].data['test_mask']
            
        g = g.to(device)
        if features is not None:
            features = features.cuda()
        
        fdp_model = fdp_model.to(device)
        optimizer = optim.Adam(fdp_model.parameters(), lr=args.lr_feat, weight_decay=weight_decay)
        sens_size = 2.0
        sens_size = torch.tensor([sens_size]).expand(g.num_nodes(n))
        sens_size = sens_size.to(device)
        sens_size = sens_size.reshape(-1,1)
        for epoch in range(100):
            fdp_model.train()
            train_features = features
            train_features = add_noise(features, delta, sens_size, epsilon)
            logits, atten = fdp_model(gs, train_features)
            sens_size = torch.matmul(atten,sens_size)
            sens_size = sens_size.detach()

            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_metrics, features, atten = evaluate(fdp_model, gs, features, labels, test_mask, score)
        
        g = g.cpu()
        features_all[n] = features.detach().cpu()
    print("Finished the feature learning!")
    return g, features_all

def evaluate(model, gs, features, labels, mask, score):
    model.eval()
    with torch.no_grad():
        if features == None:
            features = model.embed_layer()
        embed,atten = model.fdp(gs, features)
        logits = model.predict(embed)
    return score(logits[mask], labels[mask]), embed, atten
