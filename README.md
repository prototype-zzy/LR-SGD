# LR-SGD
LR-SGD: Layer-based Random SGD For Distributed Deep Learning

We propose an efficient sparsification method, layer-based random SGD (LR-SGD), that randomly select a certain number of layers of the DNN model to be exchanged instead of some elements of each tensor, which reduces communication while keep the performance close to the SSGD. Particularly, we use a hyper parameter k (i.e., indicates the number of layers to be selected) to adjust the compressing ratio while two probabilistic models are utilized to select the layers being exchanged. To validate the proposed method, we conduct several experiments on different datasets with two different scale DNN models on a stimulative cluster.

This repository contains research code for the experiments.

# Code organization

-   [LR-SGD_gather.py](LR-SGD_gather.py) and [LR-SGD_reduce.py](LR-SGD_reduce.py) are implements of our method with different collective operations. [SSGD.py](SSGD.py) and [Top-K.py](Top-K.py) are implements of synchronized stochastic gradient descent and top-k sparsification. They all have entrypoint.
-   [models/](models/__init__.py) contains some DNN models. 
-   Other files are imported as utils.
