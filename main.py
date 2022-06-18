import argparse
import os
from data import acm, dblp, imdb, amazon
from featDP import featureLearning

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main(args):
    dataset = args.dataset
    if dataset == "acm":
        data = acm.ACMDataset()
    elif dataset == "dblp":
        data = dblp.DBLPDataset()
    elif dataset == "imdb":
        data = imdb.IMDBDataset()
    elif dataset == "amazon":
        data = amazon.AmazonDataset()
    g = data[0]
    #1
    g_o = g
    gn, feat = featureLearning(g, data, args)
    g_o.ndata['feat'] = feat

    if args.task == "lp":
        from topologyDP_lp import topologylearning
    elif args.task == "nc":
        from topologyDP_nc import topologylearning
    else:
        return
    #2                                                                                                                                      
    topologylearning(g_o, args, data)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HeteDP')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--lr_feat', type=float, default=0.005, help='Initial learning rate of feature privacy-preserving.')
    parser.add_argument('--lr_topo', type=float, default=0.001, help='Initial learning rate of topology privacy-preserving.')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to downstream task.')
    parser.add_argument('--hidden_feat', '-hf', type=int, default=8, help='Number of units in feature privacy-preserving hidden layer.')
    parser.add_argument('--hidden_topo', '-hs', type=int, default=32, help='Number of units in topology privacy-preserving hidden layer.')
    parser.add_argument('--out_topo', '-o', type=int, default=16, help='Number of units in topology privacy-preserving hidden layer.')
    parser.add_argument('--delta', '-delta', type=float, default=1e-5, help='Probabilistic parameters of breaking privacy protections.')
    parser.add_argument("--epsilon_feat", type=float, default=1, help="Privacy budget of feature privacy-preserving.")
    parser.add_argument("--epsilon_topo", type=float, default=1, help="Privacy budget of topology privacy-preserving.")
    parser.add_argument("--batchsize", type=int, default=2048, help="Number of batchsize.")
    parser.add_argument('--task', choices=['lp', 'nc'], default='lp', help='Training task.')
    parser.add_argument('--dataset', '-d', type=str, choices=['acm', 'dblp', 'imdb', 'amazon'], default='acm', help='Dataset to use.')
    parser.add_argument("--dropout_feat", type=float, default=0.8, help="Dropout probability of feature privacy-preserving.")
    parser.add_argument("--weight_decay_feat", type=float, default=0.001, help="Regularization coefficients of feature privacy-preserving.")
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use.')
    
    args = parser.parse_args()
    print(args)
    main(args)
