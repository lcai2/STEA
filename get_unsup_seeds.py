import os
import numpy as np
from utils import *
from multiprocessing import Pool
import time

file_path = "./data/"
dataset = "ICEWS05-15/"
ts = time.time()
print("unsup_seeds_generation_start:"+str(ts))
def load_dict(file_path):

    train_pair = load_alignment_pair(file_path + 'sup_pairs')
    print("train_pair_len: "+str(len(train_pair)))
    dev_pair = load_alignment_pair(file_path + 'ref_pairs')
    print("dev_pair_len: "+str(len(dev_pair)))
    all_pair = train_pair + dev_pair
    print("all_pair_len: "+str(len(all_pair)))

    entity1,rel1,time1,quadruples1 = load_quadruples(file_path + 'triples_1')
    print("quadruples1_entity_len: "+str(len(entity1)))
    print("quadruples1_rel_len: "+str(len(rel1)))
    print("quadruples1_time_len: "+str(len(time1)))
    print("quadruples1_len: "+str(len(quadruples1)))

    time_dict1 = {}
    for i in entity1:
        time_dict1[i] = []
    for head,r,tail,tb,te in quadruples1:
        if (tb==te):
            time_dict1[head].append(tb); time_dict1[tail].append(tb)
        else:
            time_dict1[head].append(tb); time_dict1[tail].append(tb)
            time_dict1[head].append(te); time_dict1[tail].append(te)

    entity2,rel2,time2,quadruples2 = load_quadruples(file_path + 'triples_2')
    print("quadruples2_entity_len: "+str(len(entity2)))  
    print("quadruples2_rel_len: "+str(len(rel2)))
    print("quadruples2_time_len: "+str(len(time2)))
    print("quadruples2_len: "+str(len(quadruples2)))
    time_dict2 = {}
    for i in entity2:
        time_dict2[i] = []
    for head,r,tail,tb,te in quadruples2:
        if (tb==te):
            time_dict2[head].append(tb); time_dict2[tail].append(tb)
        else:
            time_dict2[head].append(tb); time_dict2[tail].append(tb)
            time_dict2[head].append(te); time_dict2[tail].append(te)
        

    return all_pair,entity1,entity2,time_dict1,time_dict2

all_pair, entity1, entity2, time_dict1, time_dict2 = load_dict(file_path+dataset)

thread_num =30

file_name1 = file_path+dataset+"simt/simt_" + dataset[0:-1] + "_ab.npy"
file_name2 = file_path+dataset+"simt/simt_" + dataset[0:-1] + "_ba.npy"
if os.path.exists(file_name1):
    m1 = np.load(file_name1)
    m2 = np.load(file_name2)
else:
    list1 = []
    for k in time_dict1.keys():
        dict_k = list2dict(time_dict1[k])
        list1.append(dict_k)
    list2 = []
    for k in time_dict2.keys():
        dict_k = list2dict(time_dict2[k])
        list2.append(dict_k)
    tsth = time.time()
    print("ICEWS thread_sim_matrix_start:" + str(tsth))
    m1 = thread_sim_matrix(list1,list2)
    m2 = thread_sim_matrix(list2,list1)
    cost_time = time.time()-tsth
    print("thread cost_time: ", str(cost_time))

    np.save(file_name1,m1)
    np.save(file_name2,m2)

tmp_index1 = []
tmp_index2 = []
for i in range(m1.shape[0]):
    if((len(m1[i][m1[i]==np.max(m1[i])])>1) | (np.max(m1[i]) != 1)): 
        continue
    else:
        tmp_index1.append([list(entity1)[i],list(entity2)[np.argmax(m1[i])]])
for j in range(m2.shape[0]):
    if((len(m2[j][m2[j]==np.max(m2[j])])>1) | (np.max(m2[j]) != 1)): 
        continue
    else:
        tmp_index2.append([list(entity1)[np.argmax(m2[j])],list(entity2)[j]])


sup_pair = []
tmp_index2_set=set([tuple(seed) for seed in tmp_index2])
for i in range(len(tmp_index1)):
    item_tuple=tuple(tmp_index1[i])
    if item_tuple in tmp_index2_set:
        sup_pair.append(tmp_index1[i])

file_name = file_path+dataset+"simt/unsup_seeds_" + dataset[0:-1] + ".npy"
np.save(file_name,sup_pair)
unsup_seeds_set=set([tuple(seed) for seed in sup_pair])
unsup_dev = []
for i in range(len(all_pair)):
    item_tuple=all_pair[i]
    if item_tuple not in unsup_seeds_set:
        unsup_dev.append(list(all_pair[i]))
file_name = file_path+dataset+"simt/unsup_dev_" + dataset[0:-1] + ".npy"
np.save(file_name,unsup_dev)

cost_time = time.time()-ts
print("program cost_time: ", str(cost_time))
