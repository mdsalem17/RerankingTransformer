# Reranking Transformer Adapted For ViQuAE

Forked from uvavision/RerankingTransformer (https://github.com/uvavision/RerankingTransformer). Code for [Instance-level Image Retrieval using Reranking Transformers](https://arxiv.org/abs/2103.12236)
Fuwen Tan, Jiangbo Yuan, Vicente Ordonez, ICCV 2021.


## Software required
The code is only tested on Linux 64:

```
  conda create -n rrt python=3.6
  conda activate rrt
  pip install -r requirements.txt
```

## Organization

To prepare ViQuAE dataset, please refer to the folder [ViQuAE](ViQuAE).

To use the code for experiments on Google Landmarks v2, Revisited Oxford/Paris and ViQuAE, please refer to the folder [RRT_GLD](RRT_GLD).
