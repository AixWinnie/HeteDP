import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.linear_model as lm
import sklearn.metrics as skm
import numpy as np
import dgl
from dgl.nn.pytorch import GraphConv, HeteroGraphConv, GATConv
from utilsDP import create_cum_grads

class FeatDP(nn.Module):
    def __init__(self, num_metapaths, num_node, in_dim, hidden_dim, out_dim, num_heads, dropout):
        """Feature learning """
        super().__init__()
        self.embed_layer = HeteGraphEmbed(num_node, in_dim)
        self.fdp = FeatLayer(num_metapaths, in_dim, hidden_dim, num_heads, dropout)
        self.predict = nn.Linear(num_heads * hidden_dim, out_dim)

    def forward(self, gs, h):   
        if h == None:
            h = self.embed_layer()
        h, atten = self.fdp(gs, h) 
        out = self.predict(h) 
        return out, atten

class HeteGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    def __init__(self,
                 num_node,
                 embed_size,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0):
        super(HeteGraphEmbed, self).__init__()
        self.num_node = num_node
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        
        embed = nn.Parameter(torch.Tensor(self.num_node, self.embed_size))
        nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
        self.embeds = embed

    def forward(self):
        """Forward computation """
        return self.embeds

class FeatLayer(nn.Module):
    def __init__(self, num_metapaths, in_dim, out_dim, num_heads, dropout):
        """Node Attention """
        super().__init__()
        self.node_attention = nn.ModuleList([
            GATConv(in_dim, out_dim, num_heads, dropout, dropout, activation=F.elu)
            for _ in range(num_metapaths)
        ])
        """Semantic Attention """
        self.semantic_attention = SemanticAttention(in_dim=num_heads * out_dim)

    def forward(self, gs, h):
        zp = []
        alpha ={}
        for i, gat, g in zip(range(len(gs)),self.node_attention, gs):
            temp_zp, temp_alpha = gat(g, h, True)
            zp.append(temp_zp.flatten(start_dim=1))
            temp_alpha = temp_alpha.flatten(start_dim=1).mean(1)
            alpha[i]=temp_alpha

        zp = torch.stack(zp, dim=1)
        z, atten = self.semantic_attention(zp, alpha, gs)
   
        return z, atten

class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z, alpha, gs):
        atten = 0
        w = self.project(z)
        w = F.leaky_relu(w).mean(dim=0)
        beta = torch.softmax(w, dim=0)
        for g,a,b in zip(gs, alpha, beta):
            src, dst = g.edges()
            indices = np.vstack((src.cpu(), dst.cpu()))
            values = alpha[a] * b
            edge_g = torch.sparse_coo_tensor(indices, values, (g.num_nodes(),g.num_nodes()))
            atten += edge_g.to_dense()

        beta = beta.expand((z.shape[0],) + beta.shape)
        z = (beta * z).sum(dim=1)
        return z, atten




class TopoDP(nn.Module):
    """Topology learning """
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        self.n_classes = out_feats
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats, allow_zero_in_degree=True) 
            for rel in rel_names
        }, aggregate='sum'))
        self.layers.append(HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats, allow_zero_in_degree=True) 
            for rel in rel_names
        }, aggregate='sum'))
        self.layers.append(HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats, allow_zero_in_degree=True) 
            for rel in rel_names
        }, aggregate='sum'))

        self.pred = HeteDotProductPredictor()
        self.cum_grads = create_cum_grads(self)
    
    def encoder(self, blocks, features, device):
        h = self.layers[0](blocks[0], features)
        h = {k: F.relu(v) for k, v in h.items()}
        self.mean = self.layers[1](blocks[1], h)
        self.log_std = self.layers[2](blocks[1], h)

        for i in self.mean:
            gaussian_noise = torch.randn(self.mean[i].size(0), self.n_classes)
            gaussian_noise = gaussian_noise.to(device)
            h[i] = self.mean[i] + gaussian_noise * torch.exp(self.log_std[i])
        return h

    def decoder(self, h, pos_g, neg_g):
        pos_score = self.pred(pos_g, h)
        neg_score = self.pred(neg_g, h)
        return pos_score, neg_score
    
    def forward(self, pos_g, neg_g, blocks, features, device):
        h = self.encoder(blocks, features, device)
        pos_score, neg_score = self.decoder(h, pos_g, neg_g)
        return pos_score, neg_score, h

    def inference_nc(self, g, x, category, device, batch_size):
        '''Node classification'''
        y = torch.zeros(g.num_nodes(category), self.n_classes)
        out = {n : torch.zeros(g.num_nodes(n), 16).to(device) for n in g.ntypes}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2, output_device=device)
        dict_nc = {n : torch.arange(g.num_nodes(n)).to(device) for n in g.ntypes}
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            dict_nc,
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False)
        
        for input_nodes, output_nodes, blocks in dataloader:
            h = blocks[0].srcdata['feat']
            h = self.layers[0](blocks[0], h)
            h = {k: F.relu(v) for k, v in h.items()}
            self.mean = self.layers[1](blocks[1], h)
            self.log_std = self.layers[2](blocks[1], h)
            for i in self.mean:
                gaussian_noise = torch.randn(self.mean[i].size(0), self.n_classes).to(device)
                h[i] = self.mean[i] + gaussian_noise * torch.exp(self.log_std[i])
                if i in output_nodes:
                    out[i][output_nodes[i]] = h[i]
            y[output_nodes[category]] = h[category].cpu()
        return y, out

    def inference_lp(self, blocks, features, device):  
        '''Link prediction'''     
        h = self.layers[0](blocks[0], features)
        h = {k: F.relu(v) for k, v in h.items()}
        self.mean = self.layers[1](blocks[1], h)
        self.log_std = self.layers[2](blocks[1], h)
        for i in self.mean:
            gaussian_noise = torch.randn(self.mean[i].size(0), self.n_classes).to(device)
            h[i] = self.mean[i] + gaussian_noise * torch.exp(self.log_std[i])
        return h

class HeteDotProductPredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    fn.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']



class CrossEntropyLoss(nn.Module):
    def forward(self, pos_score, neg_score):
        if isinstance(pos_score, dict):
            losses = []
            for e in pos_score:
                score = torch.cat([pos_score[e], neg_score[e]])
                label = torch.cat([torch.ones_like(pos_score[e]), torch.zeros_like(neg_score[e])]).long()
                loss = F.binary_cross_entropy_with_logits(score, label.float())
                losses.append(loss)
            return sum(losses) / len(losses)
        else:
            score = torch.cat([pos_score, neg_score])
            label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).long()
            loss = F.binary_cross_entropy_with_logits(score, label.float())
            return loss
        
def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    emb_inf = np.isinf(emb)
    emb[emb_inf] = 0
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    f1_macro_eval = skm.f1_score(val_labels, pred[val_nids], average='macro')
    f1_macro_test = skm.f1_score(test_labels, pred[test_nids], average='macro')
    return f1_micro_eval, f1_micro_test, f1_macro_eval, f1_macro_test