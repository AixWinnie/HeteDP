import numpy as np
import torch
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, EdgeDataLoader
from sklearn.metrics import roc_auc_score

from model import TopoDP, CrossEntropyLoss
from utilsDP import clip_and_accumulate, add_noise
from utils import get_device, load_config

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
    etype = data.predict_etype
    in_dim = g.nodes[etype[0]].data['feat'].shape[1]
    k = 5

    # check cuda
    use_cuda = args.gpu_id >= 0
    if use_cuda:
        train_eid_dict = {e: g.edata['train_mask'][e].nonzero(as_tuple=False).squeeze().cuda()
                            for e in g.canonical_etypes}
        val_eid_dict = {e: g.edata['val_mask'][e].nonzero(as_tuple=False).squeeze().cuda()
                            for e in g.canonical_etypes}
        test_eid_dict = {e: g.edata['test_mask'][e].nonzero(as_tuple=False).squeeze().cuda()
                            for e in g.canonical_etypes}
    else:
        train_eid_dict = {e: g.edata['train_mask'][e].nonzero(as_tuple=False).squeeze()
                            for e in g.canonical_etypes}
        val_eid_dict = {e: g.edata['val_mask'][e].nonzero(as_tuple=False).squeeze()
                            for e in g.canonical_etypes}
        test_eid_dict = {e: g.edata['test_mask'][e].nonzero(as_tuple=False).squeeze()
                            for e in g.canonical_etypes}
    g = g.to(device)

    #samlper
    sampler = MultiLayerFullNeighborSampler(2, output_device=device)
    dataloader = EdgeDataLoader(
        g, train_eid_dict, 
        sampler, negative_sampler=NegativeSampler(g, k), 
        shuffle=True, drop_last=False, num_workers=0, 
        batch_size=args.batchsize, device = device
    )
    val_loader = EdgeDataLoader(
        g, val_eid_dict,
        sampler, negative_sampler=NegativeSampler(g, k), 
        shuffle=True, drop_last=False, num_workers=0, 
        batch_size=args.batchsize, device = device
    )
    test_loader = EdgeDataLoader(
        g, test_eid_dict, 
        sampler, negative_sampler=NegativeSampler(g, k), 
        shuffle=True, drop_last=False, num_workers=0, 
        batch_size=args.batchsize, device = device
    )
    
    topodp_args = load_config(args, g)
    tdp_model = TopoDP(in_dim, topodp_args.hidden, topodp_args.out, g.etypes)
    tdp_model = tdp_model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, tdp_model.parameters()), lr=args.lr_topo)
    loss_func = CrossEntropyLoss() 
    loss_func = loss_func.to(device)
    
    pltLoss = []
    for epoch in range(args.epochs):
        tdp_model.train()
        losses = []
        for input_nodes, pos_g, neg_g, blocks in dataloader:
            optimizer.zero_grad()
            pos_score, neg_score, logits = tdp_model(pos_g, neg_g, blocks, blocks[0].srcdata['feat'], device)
            loss = loss_func(pos_score[etype], neg_score[etype])
            loss -= kl_divergence(etype[0], logits, tdp_model) + kl_divergence(etype[2], logits, tdp_model)
            losses.append(loss.item())
            loss.backward()

            clip_and_accumulate(topodp_args, tdp_model)
            add_noise(topodp_args, tdp_model)

            optimizer.step()

        val_roc = evaluate(tdp_model, val_loader, etype, device)
        pltLoss.append(sum(losses) / len(losses))
        print('Epoch {:d} | Loss {:.4f} | Valid roc: {:.4f} '.
                format(epoch, sum(losses) / len(losses), val_roc))
    
    tdp_model.eval()
    test_roc = evaluate(tdp_model, test_loader, etype, device)
    print("Test roc: {:.4f} ".format(test_roc))
    return

def kl_divergence(x, y, model):
    return 0.5 / y[x].size(0) * (
                1 + 2 * model.log_std[x] - model.mean[x] ** 2 - torch.exp(model.log_std[x]) ** 2).sum(
            1).mean()

def evaluate(model, loader, etype, device):
    model.eval()
    roc_scores = []
    for input_nodes, pos_g, neg_g, blocks in loader:
        blocks = [b.to(device) for b in blocks]
        pred = model.inference_lp(blocks, blocks[0].srcdata['feat'], device)
        score = torch.matmul(pred[etype[0]],pred[etype[2]].t())
        adj = pos_g.adjacency_matrix(etype=etype).to_dense().to(device)
        adj_neg = neg_g.adjacency_matrix(etype=etype).to_dense().to(device)
        score_pos = torch.zeros(pos_g.num_edges(etype))
        score_neg = torch.zeros(neg_g.num_edges(etype))
        score_pos = torch.sigmoid(score[torch.nonzero(adj*score, as_tuple=True)]).cpu().detach().numpy()
        score_neg = torch.sigmoid(score[torch.nonzero(adj_neg*score, as_tuple=True)]).cpu().detach().numpy()
        labels_all = np.hstack(([np.ones(len(score_pos)), np.zeros(len(score_neg))]))
        preds_all = np.hstack((score_pos,score_neg))
        
        roc_score = roc_auc_score(labels_all, preds_all)
        roc_scores.append(roc_score)

    return sum(roc_scores) / len(roc_scores)

