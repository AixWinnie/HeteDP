# HeteDP
Code for "Heterogeneous Graph Neural Network for Privacy-Preserving Recommendation"

# Environment
Our experimental environments are listed in `environments.yaml`, you can create a virtual environment with conda and run the following order.
```
conda env create -f environments.yaml
```

# Install
Enter the virtual environment and run the `requirements.txt`.
```
pip install -r requirements.txt
```

# Usage
Run the following order to train our model with setting custom parameters.
```
python3 main.py -e=50 -ef=0.5 -et=0.5 --task='lp' -d='acm' -g=0
```

# Thanks
Some of the code was forked from the following repositories:
* [pytorch](https://github.com/ZZy979/pytorch-tutorial)
* [opacus](https://github.com/pytorch/opacus/tree/main/opacus)
* [dgl](https://github.com/dmlc/dgl)
