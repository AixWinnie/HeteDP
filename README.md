# HeteDP
Code for "Heterogeneous Graph Neural Network for Privacy-Preserving Recommendation"

# Environment
Our experimental environments are listed in `environments.yaml`, you can create a virtual environment with conda and run the following orders.
```
conda env create -f environments.yaml
```

# Install
Enter the virtual environment and run the `requirements.txt`.
```
pip install -r requirements.txt
```

# Usage

```
python3 main.py -e=50 -ef=0.5 -et=0.5 --task='lp' -d='acm' -g=0
```
