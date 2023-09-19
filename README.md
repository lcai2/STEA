# STEA
Code for COLING2022 paper [“A Simple Temporal Information Matching Mechanism for Entity Alignment Between Temporal Knowledge Graphs”](https://arxiv.org/abs/2209.09677).
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
# Usage
Use the following command:  
python main.py  
Before conducting unsupervised experiments, you need to run "get_unsup_seeds.py" to obtain unsupervised seeds.  
The obtained unsupervised seed file--"*.npy" is stored in the "simt" folder in the corresponding dataset folder.  
You also need to set the parameter "unsupervise" to "True" in the "args.py". 
# Acknowledgement
We refer to the code of RREA. Thanks for their great contributions!
# Citation
If you use this model or code, please cite it as follows:  
@inproceedings{Cai2022STEA,   
  author    = {Li Cai and Xin Mao and Meirong Ma and Hao Yuan and Jianchao Zhu and Man Lan},  
  title     = {A Simple Temporal Information Matching Mechanism for Entity Alignment Between Temporal Knowledge Graphs},  
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},  
  pages     = {2075--2086},  
  year      = {2022}, 
}
