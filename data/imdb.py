import itertools
import os

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import download, save_graphs, load_graphs
from sklearn.feature_extraction.text import CountVectorizer
from data.dataprepro import get_node_mask

class IMDBDataset(DGLDataset):
    """
    Movie network
    
    Statistics
    -----
    * Nodes: 4278 movie, 5257 actor, 2081 director
    * Edges: 12828 movie-actor, 4278 movie-director
    * Categories: 3
    * Node_split: 20% train, 10% valid
    * Edge_split: 85% train, 5% valid
    """
    

    _url = 'https://raw.githubusercontent.com/Jhy1993/HAN/master/data/imdb/movie_metadata.csv'
    _seed = 42

    def __init__(self):
        super().__init__('imdb', self._url)

    def download(self):
        file_path = os.path.join(self.raw_dir, 'imdb.csv')
        if not os.path.exists(file_path):
            download(self.url, path=file_path)

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        self.g = graphs[0]
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes['movie'].data[k] = self.g.nodes['movie'].data[k].bool()
            self.g.nodes['actor'].data[k] = self.g.nodes['movie'].data[k].bool()
            self.g.nodes['director'].data[k] = self.g.nodes['movie'].data[k].bool()
        
        for k in ('ma', 'am', 'md', 'dm'):
            self.g.edges[k].data['train_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.85)
            self.g.edges[k].data['val_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.05)
            self.g.edges[k].data['test_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.1)

    def process(self):
        self.data = pd.read_csv(os.path.join(self.raw_dir, 'imdb.csv'), encoding='utf8') \
            .dropna(axis=0, subset=['actor_1_name', 'director_name']).reset_index(drop=True)
        self.labels = self._extract_labels()
        self.movies = list(sorted(m.strip() for m in self.data['movie_title']))
        self.directors = list(sorted(set(self.data['director_name'])))
        self.actors = list(sorted(set(itertools.chain.from_iterable(
            self.data[c].dropna().to_list()
            for c in ('actor_1_name', 'actor_2_name', 'actor_3_name')
        ))))
        self.g = self._build_graph()
        self._add_ndata()
        self._add_edata()

    def _extract_labels(self):
        labels = np.full(len(self.data), -1)
        for i, genres in self.data['genres'].iteritems():
            for genre in genres.split('|'):
                if genre == 'Action':
                    labels[i] = 0
                    break
                elif genre == 'Comedy':
                    labels[i] = 1
                    break
                elif genre == 'Drama':
                    labels[i] = 2
                    break
        other_idx = np.where(labels == -1)[0]
        self.data = self.data.drop(other_idx).reset_index(drop=True)
        return np.delete(labels, other_idx, axis=0)

    def _build_graph(self):
        ma, md = set(), set()
        for m, row in self.data.iterrows():
            d = self.directors.index(row['director_name'])
            md.add((m, d))
            for c in ('actor_1_name', 'actor_2_name', 'actor_3_name'):
                if row[c] in self.actors:
                    a = self.actors.index(row[c])
                    ma.add((m, a))
        ma, md = list(ma), list(md)
        ma_m, ma_a = [e[0] for e in ma], [e[1] for e in ma]
        md_m, md_d = [e[0] for e in md], [e[1] for e in md]
        return dgl.heterograph({
            ('movie', 'ma', 'actor'): (ma_m, ma_a),
            ('actor', 'am', 'movie'): (ma_a, ma_m),
            ('movie', 'md', 'director'): (md_m, md_d),
            ('director', 'dm', 'movie'): (md_d, md_m)
        })

    def _add_ndata(self):
        vectorizer = CountVectorizer(min_df=5)
        features = vectorizer.fit_transform(self.data['plot_keywords'].fillna('').values)
        self.g.nodes['movie'].data['feat'] = torch.from_numpy(features.toarray()).float()
        self.g.nodes['movie'].data['label'] = torch.from_numpy(self.labels).long()

        self.g.multi_update_all({
            'ma': (fn.copy_u('feat', 'm'), fn.mean('m', 'feat')),
            'md': (fn.copy_u('feat', 'm'), fn.mean('m', 'feat'))
        }, 'sum')

        for n in self.g.ntypes:
            train_mask, val_mask, test_mask = get_node_mask(self.g.num_nodes(n), self._seed)
            self.g.nodes[n].data['train_mask'] = train_mask
            self.g.nodes[n].data['val_mask'] = val_mask
            self.g.nodes[n].data['test_mask'] = test_mask

    def _add_edata(self):
        for k in ('ma', 'am', 'md', 'dm'):
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
        if ntype == "movie":
            return 3
        if ntype == "actor":
            return 3
        if ntype == "director":
            return 3

    def metapaths(self, ntype):
        if ntype == "movie":
            return [['ma', 'am'], ['md', 'dm']]
        if ntype == "actor":
            return [['am', 'ma'], ['am', 'md', 'dm', 'ma']]
        if ntype == "director":
            return [['dm', 'md'], ['dm', 'ma', 'am', 'md']]

    @property
    def predict_ntype(self):
        return 'movie'
    
    @property
    def predict_etype(self):
        return 'movie', 'ma', 'actor'
