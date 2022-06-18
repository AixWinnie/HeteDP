import os
import pickle

import dgl
import dgl.function as fn
import numpy as np
import scipy.io as sio
import torch
from dgl.data import DGLDataset
from dgl.data.utils import _get_dgl_url, download, save_graphs, load_graphs, \
    generate_mask_tensor, idx2mask

from data.dataprepro import get_node_mask

class ACMDataset(DGLDataset):
    """
    Citation network

    Statistics
    -----
    * Nodes: 17351 author, 4025 paper, 72 field
    * Edges: 13407 paper-author, 4025 paper-field
    * Categories: 3
    * Node_split: 20% train, 10% valid
    * Edge_split: 85% train, 5% valid
    """

    
    _seed = 42

    def __init__(self):
        super().__init__('ACM', _get_dgl_url('dataset/ACM.mat'))

    def download(self):
        file_path = os.path.join(self.raw_dir, 'ACM.mat')
        if not os.path.exists(file_path):
            download(self.url, path=file_path)

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        self.g = graphs[0]
        # save_graphs会将bool转换成uint8
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes['paper'].data[k] = self.g.nodes['paper'].data[k].bool()
            self.g.nodes['author'].data[k] = self.g.nodes['author'].data[k].bool()
            self.g.nodes['field'].data[k] = self.g.nodes['field'].data[k].bool()

    def process(self):
        data = sio.loadmat(os.path.join(self.raw_dir, 'ACM.mat'))
        p_vs_l = data['PvsL']  # paper-field?
        p_vs_a = data['PvsA']  # paper-author
        p_vs_t = data['PvsT']  # paper-term, bag of words
        p_vs_c = data['PvsC']  # paper-conference, labels come from that

        # We assign
        # (1) KDD papers as class 0 (data mining),
        # (2) SIGMOD and VLDB papers as class 1 (database),
        # (3) SIGCOMM and MobiCOMM papers as class 2 (communication)
        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        p_vs_c_filter = p_vs_c[:, conf_ids]
        #p_vs_t_filter = p_vs_t[:, ]
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
        p_vs_l = p_vs_l[p_selected]
        p_vs_a = p_vs_a[p_selected]
        p_vs_t = p_vs_t[p_selected]
        p_vs_c = p_vs_c[p_selected]

        self.g = dgl.heterograph({
            ('paper', 'pa', 'author'): p_vs_a.nonzero(),
            ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
            ('paper', 'pf', 'field'): p_vs_l.nonzero(),
            ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
        })
        paper_features = torch.FloatTensor(p_vs_t.toarray())  # (4025, 1903)
        #给label打标签
        pc_p, pc_c = p_vs_c.nonzero()
        paper_labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            paper_labels[pc_p[pc_c == conf_id]] = label_id
        paper_labels = torch.from_numpy(paper_labels)

        float_mask = np.zeros(len(pc_p))
        for conf_id in conf_ids:
            pc_c_mask = (pc_c == conf_id)
            float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
        train_idx = np.where(float_mask <= 0.2)[0]
        val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]

        num_paper_nodes = self.g.num_nodes('paper')
        train_mask = generate_mask_tensor(idx2mask(train_idx, num_paper_nodes))
        val_mask = generate_mask_tensor(idx2mask(val_idx, num_paper_nodes))
        test_mask = generate_mask_tensor(idx2mask(test_idx, num_paper_nodes))
        self.g.nodes['paper'].data['feat'] = paper_features
        self.g.nodes['paper'].data['label'] = paper_labels
        self.g.nodes['paper'].data['train_mask'] = train_mask
        self.g.nodes['paper'].data['val_mask'] = val_mask
        self.g.nodes['paper'].data['test_mask'] = test_mask

        for n in ['author', 'field']:
            train_mask, val_mask, test_mask = get_node_mask(self.g.num_nodes(n), self._seed)
            self.g.nodes[n].data['train_mask'] = train_mask
            self.g.nodes[n].data['val_mask'] = val_mask
            self.g.nodes[n].data['test_mask'] = test_mask
        
        self.g.multi_update_all({'pa': (fn.copy_u('feat', 'm'), fn.mean('m', 'feat'))}, 'sum')
        n_field = self.g.num_nodes('field')
        self.g.nodes['field'].data['feat'] = torch.cat((torch.eye(n_field),torch.zeros(n_field,paper_features.shape[1]-n_field)),dim=1)
        
        for k in ('pa', 'ap', 'pf', 'fp'):
            self.g.edges[k].data['train_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.85)
            self.g.edges[k].data['val_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.05)
            self.g.edges[k].data['test_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.1)

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.g

    def __len__(self):
        return 1

    def num_classes(self, ntype):
        if ntype == "author":
            return 3
        if ntype == "paper":
            return 3
        if ntype == "field":
            return 3
    
    def metapaths(self, ntype):
        if ntype == "author":
            return [['ap', 'pa'], ['ap','pf', 'fp', 'pa']]
        if ntype == "paper":
            return [['pa', 'ap'], ['pf', 'fp']]
        if ntype == "field":
            return [['fp', 'pf'], ['fp','pa', 'ap', 'pf']]

    @property
    def predict_ntype(self):
        return 'paper'

    @property
    def predict_etype(self):
        return 'paper', 'pa', 'author'