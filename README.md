# HMGE : Hierarchical Aggregations for High-Dimensional Multiplex Graph Embedding

![illustration]([https://github.com/abdouskamel/HMGE/blob/main/illustration.png?raw=true])

We investigate the problem of multiplex graph embedding, that is, graphs in which nodes interact through multiple types of relations (dimensions). In recent years, several methods have been developed to address this problem. However, the need for more effective and specialized approaches grows with the production of graph data with diverse characteristics. In particular, real-world multiplex graphs may exhibit a high number of dimensions, making it difficult to construct a single consensus representation. Furthermore, important information can be hidden in complex latent structures scattered in multiple dimensions. To address these issues, we propose \methodname, a novel embedding method based on hierarchical aggregation for high-dimensional multiplex graphs. Hierarchical aggregation consists in learning a hierarchical combination of the graph dimensions and refining the embeddings at each hierarchy level. Non-linear combinations are computed from previous ones, thus uncovering complex information and latent structures hidden in the multiplex graph dimensions. Moreover, we leverage mutual information maximization between local patches and global summaries to train the model without supervision. This allows to captures globally relevant information present in diverse locations of the graph. Detailed experiments on synthetic and real-world data illustrate the suitability of our approach on downstream supervised tasks, including link prediction and node classification. 

## Requirements
Python 3.6 <br />
numpy <br />
scipy <br />
scikit-learn <br />
pytorch <br />
tqdm

## Run
`main.py DATASET_NAME LINK_PREDICTION`

- Specify the dataset name in `DATASET_NAME`. The possible values are the following : biogrid_4503_bis, biogrid_4211, dblp_5124, imdb_3000, STRING-DB_4083
- Set `LINK_PREDICTION` to False if you want to perform node classification. Set it to true if you want to perform link prediction.

Examples :

`main.py biogrid_4211 False` <br />
`main.py biogrid_4211 True`

## Reference
If you find this work useful in your research, please consider citing the following paper:

```
@ARTICLE{abdous2023hierarchical,
  author={Abdous, Kamel and Mrabah, Nairouz and Bouguessa, Mohamed},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Hierarchical Aggregations for High-Dimensional Multiplex Graph Embedding}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TKDE.2023.3305809}
}
```
