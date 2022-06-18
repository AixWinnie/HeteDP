import imp
import os
import dgl
import pandas as pd
import scipy.io as sio
import torch
from dgl.data import DGLDataset
from dgl.data.utils import makedirs, download, save_graphs, load_graphs
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords

from data.dataprepro import get_node_mask

class DBLPDataset(DGLDataset):
    """
    Citation network
    
    Statistics
    -----
    * Nodes: 4057 author, 14328 paper, 20 conf, 7723 term
    * Edges: 19645 paper-author, 14328 paper-conf, 85810 paper-term
    * Categories: 4
    * Node_split: 20% train, 10% valid
    * Edge_split: 85% train, 5% valid
    """


    _url = 'https://raw.githubusercontent.com/Jhy1993/HAN/master/data/DBLP_four_area/'
    _url2 = 'https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg'
    _raw_files = [
        'readme.txt', 'author_label.txt', 'paper.txt', 'conf_label.txt', 'term.txt',
        'paper_author.txt', 'paper_conf.txt', 'paper_term.txt'
    ]
    _seed = 42

    def __init__(self):
        super().__init__('DBLP', self._url)

    def download(self):
        if not os.path.exists(self.raw_path):
            makedirs(self.raw_path)
        for file in self._raw_files:
            download(self.url + file, os.path.join(self.raw_path, file))

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        self.g = graphs[0]
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes['author'].data[k] = self.g.nodes['author'].data[k].bool()
            self.g.nodes['paper'].data[k] = self.g.nodes['paper'].data[k].bool()
            self.g.nodes['conf'].data[k] = self.g.nodes['conf'].data[k].bool()
            self.g.nodes['term'].data[k] = self.g.nodes['term'].data[k].bool()
        
        for k in ('pa', 'ap', 'pc', 'cp', 'pt', 'tp'):
            self.g.edges[k].data['train_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.85)
            self.g.edges[k].data['val_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.05)
            self.g.edges[k].data['test_mask'] = torch.zeros(self.g.num_edges(k), dtype=torch.bool).bernoulli(0.1)

    def process(self):
        self.authors, self.papers, self.confs, self.terms, \
            self.paper_author, self.paper_conf, self.paper_term = self._read_raw_data()
        self._filter_nodes_and_edges()
        self._lemmatize_terms()
        self._remove_stopwords()
        self._reset_index()

        self.g = self._build_graph()
        self._add_ndata()
        self._add_edata()

    def _read_raw_data(self):
        authors = self._read_file('author_label.txt', names=['id', 'label', 'name'], index_col='id')
        papers = self._read_file('paper.txt', names=['id', 'title'], index_col='id', encoding='cp1252')
        confs = self._read_file('conf_label.txt', names=['id', 'label', 'name', 'dummy'], index_col='id')
        terms = self._read_file('term.txt', names=['id', 'name'], index_col='id')
        paper_author = self._read_file('paper_author.txt', names=['paper_id', 'author_id'])
        paper_conf = self._read_file('paper_conf.txt', names=['paper_id', 'conf_id'])
        paper_term = self._read_file('paper_term.txt', names=['paper_id', 'term_id'])
        return authors, papers, confs, terms, paper_author, paper_conf, paper_term

    def _read_file(self, filename, names, index_col=None, encoding='utf8'):
        return pd.read_csv(
            os.path.join(self.raw_path, filename), sep='\t', names=names, index_col=index_col,
            keep_default_na=False, encoding=encoding
        )

    def _filter_nodes_and_edges(self):
        self.paper_author = self.paper_author[self.paper_author['author_id'].isin(self.authors.index)]
        paper_ids = self.paper_author['paper_id'].drop_duplicates()
        self.papers = self.papers.loc[paper_ids]
        self.paper_conf = self.paper_conf[self.paper_conf['paper_id'].isin(paper_ids)]
        self.paper_term = self.paper_term[self.paper_term['paper_id'].isin(paper_ids)]
        self.terms = self.terms.loc[self.paper_term['term_id'].drop_duplicates()]

    def _lemmatize_terms(self):
        lemmatizer = WordNetLemmatizer()
        lemma_id_map, term_lemma_map = {}, {}
        for index, row in self.terms.iterrows():
            lemma = lemmatizer.lemmatize(row['name'])
            term_lemma_map[index] = lemma_id_map.setdefault(lemma, index)
        self.terms = pd.DataFrame(
            list(lemma_id_map.keys()), columns=['name'],
            index=pd.Index(lemma_id_map.values(), name='id')
        )
        self.paper_term.loc[:, 'term_id'] = [
            term_lemma_map[row['term_id']] for _, row in self.paper_term.iterrows()
        ]
        self.paper_term.drop_duplicates(inplace=True)

    def _remove_stopwords(self):
        stop_words = sklearn_stopwords.union(nltk_stopwords.words('english'))
        self.terms = self.terms[~(self.terms['name'].isin(stop_words))]
        self.paper_term = self.paper_term[self.paper_term['term_id'].isin(self.terms.index)]

    def _reset_index(self):
        self.authors.reset_index(inplace=True)
        self.papers.reset_index(inplace=True)
        self.confs.reset_index(inplace=True)
        self.terms.reset_index(inplace=True)
        author_id_map = {row['id']: index for index, row in self.authors.iterrows()}
        paper_id_map = {row['id']: index for index, row in self.papers.iterrows()}
        conf_id_map = {row['id']: index for index, row in self.confs.iterrows()}
        term_id_map = {row['id']: index for index, row in self.terms.iterrows()}

        self.paper_author.loc[:, 'author_id'] = [author_id_map[i] for i in self.paper_author['author_id'].to_list()]
        self.paper_conf.loc[:, 'conf_id'] = [conf_id_map[i] for i in self.paper_conf['conf_id'].to_list()]
        self.paper_term.loc[:, 'term_id'] = [term_id_map[i] for i in self.paper_term['term_id'].to_list()]
        for df in (self.paper_author, self.paper_conf, self.paper_term):
            df.loc[:, 'paper_id'] = [paper_id_map[i] for i in df['paper_id']]

    def _build_graph(self):
        pa_p, pa_a = self.paper_author['paper_id'].to_list(), self.paper_author['author_id'].to_list()
        pc_p, pc_c = self.paper_conf['paper_id'].to_list(), self.paper_conf['conf_id'].to_list()
        pt_p, pt_t = self.paper_term['paper_id'].to_list(), self.paper_term['term_id'].to_list()
        return dgl.heterograph({
            ('paper', 'pa', 'author'): (pa_p, pa_a),
            ('author', 'ap', 'paper'): (pa_a, pa_p),
            ('paper', 'pc', 'conf'): (pc_p, pc_c),
            ('conf', 'cp', 'paper'): (pc_c, pc_p),
            ('paper', 'pt', 'term'): (pt_p, pt_t),
            ('term', 'tp', 'paper'): (pt_t, pt_p)
        })

    def _add_ndata(self):
        _raw_file2 = os.path.join(self.raw_dir, 'DBLP4057_GAT_with_idx.mat')
        if not os.path.exists(_raw_file2):
            raise FileNotFoundError('Please download the file {} according to the extraction code: 6b3h and save to {}'.format(
                self._url2, _raw_file2
            ))
        mat = sio.loadmat(_raw_file2)
        self.g.nodes['author'].data['feat'] = torch.from_numpy(mat['features']).float()
        self.g.nodes['author'].data['label'] = torch.tensor(self.authors['label'].to_list())
        
        for n in ['paper', 'conf', 'term']:
            self.g.nodes[n].data['feat'] = torch.randn(self.g.num_nodes(n), mat['features'].shape[1])
        
        for n in self.g.ntypes:
            train_mask, val_mask, test_mask = get_node_mask(self.g.num_nodes(n), self._seed)
            self.g.nodes[n].data['train_mask'] = train_mask
            self.g.nodes[n].data['val_mask'] = val_mask
            self.g.nodes[n].data['test_mask'] = test_mask

        self.g.nodes['conf'].data['label'] = torch.tensor(self.confs['label'].to_list())
        
    
    #NOTE
    def _add_edata(self):
        for k in ('pa', 'ap', 'pc', 'cp', 'pt', 'tp'):
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
            return 4
        if ntype == "paper":
            return 4
        if ntype == "conf":
            return 4
        if ntype == "term":
            return 4

    def metapaths(self, ntype):
        if ntype == "author":
            return [['ap', 'pa'], ['ap', 'pc', 'cp', 'pa'], ['ap', 'pt', 'tp', 'pa']]
        if ntype == "paper":
            return [['pa', 'ap'], ['pt', 'tp'], ['pc', 'cp']]
        if ntype == "conf":
            return [['cp', 'pc'], ['cp', 'pa', 'ap', 'pc'], ['cp', 'pt', 'tp', 'pc']]
        if ntype == "term":
            return [['tp', 'pt'], ['tp', 'pa', 'ap', 'pt'], ['tp', 'pc', 'cp', 'pt']]

    @property
    def predict_ntype(self):
        return 'author'

    @property
    def predict_etype(self):
        return 'author', 'ap', 'paper'

