import itertools
import os

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import makedirs, save_graphs, load_graphs, \
    generate_mask_tensor, idx2mask

from data.dataprepro import split_idx, get_node_mask, get_label

class AmazonDataset(DGLDataset):
    """
    E-commerce network
    
    Statistics
    -----
    * Nodes: 6170 user, 2753 item, 3857 view, 22 category
    * Edges: 195791 user-item, 5694 item-view, 5508 item-category
    * Categories: 3
    * Node_split: 20% train, 10% valid
    * Edge_split: 85% train, 5% valid
    """
    

    _url = './data/Amazon/'
    _raw_files = [
        'item_brand.dat', 'item_category.dat', 'item_view.dat', 'user_item.dat'
    ]
    _seed = 42

    def __init__(self):
        super().__init__('Amazon', self._url)

    def download(self):
        if not os.path.exists(self.raw_path):
            makedirs(self.raw_path)

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        self.g = graphs[0]
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes['category'].data[k] = self.g.nodes['category'].data[k].bool()
            self.g.nodes['item'].data[k] = self.g.nodes['item'].data[k].bool()
            self.g.nodes['user'].data[k] = self.g.nodes['user'].data[k].bool()
            self.g.nodes['view'].data[k] = self.g.nodes['view'].data[k].bool()
        
        for k in ('ic', 'ci', 'iv', 'vi', 'ui', 'iu'):
            self.g.edges[k].data['train_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.85)
            self.g.edges[k].data['val_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.05)
            self.g.edges[k].data['test_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.1)


    def process(self):
        self.item_brand, self.item_category, self.item_view, self.user_item  = self._read_raw_data()
        self.g = self._build_graph()
        self._add_ndata()
        self._add_edata()

    def _read_raw_data(self):
        item_brand = self._read_file('item_brand.dat', names=['item_id', 'brand_id'], sep=',', index_col=None)
        item_category = self._read_file('item_category.dat', names=['item_id', 'category_id'], sep=',', index_col=None)
        item_view = self._read_file('item_view.dat', names=['item_id', 'view_id'], sep=',', index_col=None)
        user_item = self._read_file('user_item.dat', names=['user_id', 'item_id'], sep=',', index_col=[0,1])
        return item_brand, item_category, item_view, user_item

    def _read_file(self, filename, names, sep, index_col, encoding='utf8'):
        return pd.read_csv(self._url+filename, header=None, sep=sep, names=names, usecols=index_col,
             encoding=encoding)
    
    def _build_graph(self):
        ib_i, ib_b = self.item_brand['item_id'].to_list(), self.item_brand['brand_id'].to_list()
        ic_i, ic_c = self.item_category['item_id'].to_list(), self.item_category['category_id'].to_list()
        iv_i, iv_v = self.item_view['item_id'].to_list(), self.item_view['view_id'].to_list()
        ui_u, ui_i = self.user_item['user_id'].to_list(), self.user_item['item_id'].to_list()
        return dgl.heterograph({
            ('item', 'ic', 'category'): (ic_i, ic_c),
            ('category', 'ci', 'item'): (ic_c, ic_i),
            ('item', 'iv', 'view'): (iv_i, iv_v),
            ('view', 'vi', 'item'): (iv_v, iv_i),
            ('user', 'ui', 'item'): (ui_u, ui_i),
            ('item', 'iu', 'user'): (ui_i, ui_u)
        })
    
    def _add_ndata(self):  
        dim = 128
        for n in self.g.ntypes:
            train_mask, val_mask, test_mask = get_node_mask(self.g.num_nodes(n), self._seed)
            self.g.nodes[n].data['train_mask'] = train_mask
            self.g.nodes[n].data['val_mask'] = val_mask
            self.g.nodes[n].data['test_mask'] = test_mask

            self.g.nodes[n].data['feat'] = torch.randn(self.g.num_nodes(n), dim)

        n_items = self.g.num_nodes('item')
        degress = torch.zeros(n_items)
        num_classes=3
        for e in ('ui','vi','ci'):
            degress += self.g.in_degrees(etype=e)
        labels = get_label(degress, n_items, num_classes)
        self.g.nodes['item'].data['label'] = labels
        

    def _add_edata(self):
        for k in ('ic', 'ci', 'iv', 'vi', 'ui', 'iu'):
            self.g.edges[k].data['train_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.85)
            self.g.edges[k].data['val_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.05)
            self.g.edges[k].data['test_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.1)

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))

    def __getitem__(self, i):
        return self.g

    def __len__(self):
        return 1

    def num_classes(self, ntype):
        if ntype == "category":
            return 2
        if ntype == "item":
            return 3
        if ntype == "user":
            return 3
        if ntype == "view":
            return 3

    def metapaths(self, ntype):
        if ntype == "category":
            return [['ci', 'ic'], ['ci', 'iv', 'vi', 'ic'], ['ci', 'iu', 'ui', 'ic']]
        if ntype == "item":
            return [['iu', 'ui'], ['ic', 'ci'], ['iv', 'vi']]
        if ntype == "user":
            return [['ui', 'iu'], ['ui', 'ic', 'ci', 'iu'], ['ui', 'iv', 'vi', 'iu']]
        if ntype == "view":
            return [['vi', 'iv'], ['vi', 'iu', 'ui', 'iv'], ['vi', 'ic', 'ci', 'iv']]

    @property
    def predict_ntype(self):
        return 'item'

    @property
    def predict_etype(self):
        return 'item', 'iu', 'user'