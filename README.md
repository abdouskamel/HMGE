# Hierarchical Aggregations for High-Dimensional Multiplex Graph Embedding

## Requirements
Python 3.6
numpy
scipy
scikit-learn
pytorch
tqdm

## Run
`main.py DATASET_NAME LINK_PREDICTION`

- Specify the dataset name in `DATASET_NAME`. The possible values are the following : biogrid_4503_bis, biogrid_4211, dblp_5124, imdb_3000, STRING-DB_4083
- Set `LINK_PREDICTION` to False if you want to perform node classification. Set it to true if you want to perform link prediction.

Examples :

`main.py biogrid_4211 False`
`main.py biogrid_4211 True`
