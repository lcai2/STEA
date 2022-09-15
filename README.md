# STEA
Code for COLING2022 paper “A Simple Temporal Information Matching Mechanism for Entity Alignment Between Temporal Knowledge Graphs”
# Datasets
The datasets are from [TEA-GNN](https://github.com/soledad921/TEA-GNN).
ent_ids_1: ids for entities in source KG;
ent_ids_2: ids for entities in target KG;
triples_1: relation triples encoded by ids in source KG;
triples_2: relation triples encoded by ids in target KG;
rel_ids_1: ids for entities in source KG;
rel_ids_2: ids for entities in target KG;
sup_pairs + ref_pairs: entity alignments
# Environment
Anaconda>=4.5.11  
Python>=3.7.11  
pytorch>=1.10.1  
# Usage:
Use the following command:  
python main.py
# Acknowledgement
We refer to the code of RREA. Thanks for their great contributions!
# Citation
If you use this model or code, please cite it as follows:  
@inproceedings{STEA,
  author    = {Li Cai, Xin Mao, Meirong Ma, Hao Yuan, Jianchao Zhu, Man Lan},
  title     = {A Simple Temporal Information Matching Mechanism for Entity Alignment Between Temporal Knowledge Graphs},
  booktitle = {COLLING},
  pages     = {},
  year      = {2022}
}
