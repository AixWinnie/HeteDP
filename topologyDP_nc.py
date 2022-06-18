import torch
import torch.optim as optim
from dgl.dataloading import EdgeDataLoader, MultiLayerNeighborSampler

from model import TopoDP, CrossEntropyLoss, compute_acc_unsupervised as compute_acc
from utilsDP import clip_and_accumulate, add_noise
from utils import get_device, set_random_seed, load_config

class NegativeSampler(object):
    def __init__(self, g, k):
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for etype in g.canonical_etypes
        }
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            n = len(src)
            if n % self.k == 0:
                dst = self.weights[etype].multinomial(n, replacement=True)
                dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
            else:
                dst = self.weights[etype].multinomial(n*self.k, replacement=True)
            src = src.repeat_interleave(self.k)
            result_dict[etype] = (src, dst)
        return result_dict

def topologylearning(g, args, data):
    print("Start topology structure learning...")
    device = get_device(args.gpu_id)
    set_random_seed(args.seed)
    category = data.predict_ntype
    g = g.to(device)
    features = g.ndata['feat']
    in_dim = features[category].shape[1]
    labels = g.nodes[category].data['label']
    labels = labels.to(device)

    train_mask = g.nodes[category].data['train_mask']
    val_mask = g.nodes[category].data['val_mask']
    test_mask = g.nodes[category].data['test_mask']
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

    k = 5 
    train_eid_dict = {etype: g.edges(etype=etype, form='eid')
                        for etype in g.etypes}
    #samlper
    sampler = MultiLayerNeighborSampler([10,25], output_device=device)
    dataloader = EdgeDataLoader(
        g, train_eid_dict, 
        sampler, 
        negative_sampler=NegativeSampler(g, k), 
        batch_size=args.batchsize, shuffle=True, drop_last=False, 
        num_workers=0, device = device)

    topodp_args = load_config(args, g)
    tdp_model = TopoDP(in_dim, topodp_args.hidden, topodp_args.out, g.etypes)
    tdp_model = tdp_model.to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, tdp_model.parameters()), lr=args.lr_topo)
    loss_func = CrossEntropyLoss() 
    loss_func = loss_func.to(device)
    
    pltLoss = []
    best_test_mic = 0
    for epoch in range(args.epochs):
        tdp_model.train()
        losses = []
        for input_nodes, pos_g, neg_g, blocks in dataloader:
            pos_score, neg_score, node_embeddings = tdp_model(pos_g, neg_g, blocks, blocks[0].srcdata['feat'], device)
            loss = loss_func(pos_score, neg_score)
            loss -= kl_divergence(node_embeddings, tdp_model)
            if loss<0:
                break
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()

            clip_and_accumulate(topodp_args, tdp_model)
            add_noise(topodp_args, tdp_model)
            
            optimizer.step()
        pltLoss.append(sum(losses) / len(losses))

        val_mi, test_mi, val_ma, test_ma, feat = evaluate(tdp_model, g, features, category, labels, train_idx, val_idx, test_idx, args, device)
        print('Epoch {:d} | Loss {:.4f} | Valid mic: {:.4f} | Valid mac: {:.4f} | Test mic: {:.4f} | Test mac: {:.4f} '.
                format(epoch, sum(losses) / len(losses), val_mi, val_ma, test_mi, test_ma))
        if test_mi > best_test_mic:
            best_eval_mic = val_mi
            best_eval_mac = val_ma
            best_test_mic = test_mi
            best_test_mac = test_ma
            
    print("Best Val Micro: {:.4f} | Val Macro: {:.4f} | Test Micro: {:.4f} | Test Macro: {:.4f}".format(
        best_eval_mic, best_eval_mac, best_test_mic, best_test_mac))
    return

def kl_divergence(y, model):
    if isinstance(y,dict):
        kls = []
        for x in y:
            kl = 0.5 / y[x].size(0) * (
                    1 + 2 * model.log_std[x] - model.mean[x] ** 2 - torch.exp(model.log_std[x]) ** 2).sum(
                1).mean()
            kls.append(kl)
        return sum(kls) / len(kls)
    else:
        return 0.5 / model.log_std[y].size(0) * (
                1 + 2 * model.log_std[y] - model.mean[y] ** 2 - torch.exp(model.log_std[y]) ** 2).sum(
            1).mean()

def evaluate(model, g, nfeat, category, labels, train_nids, val_nids, test_nids, args, device):
    model.eval()
    with torch.no_grad():
        pred, feat = model.inference_nc(g, nfeat, category, device, args.batchsize)
    model.train()
    val_mi, test_mi, val_ma, test_ma = compute_acc(pred, labels, train_nids, val_nids, test_nids)
    return val_mi, test_mi, val_ma, test_ma, feat
