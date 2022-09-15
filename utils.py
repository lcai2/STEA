from pickle import NONE
import numpy as np
import numba as nb
import scipy.sparse as sp
import scipy
import os
from multiprocessing import Pool
import torch
from log import logger

def normalize_adj(adj):
    adj = sp.coo_matrix(adj) 
    rowsum = np.array(adj.sum(1))  
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def sp2torch_sparse(X):
    coo = X.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = X.shape
    X=torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return X

def load_quadruples(file_name):
    quadruples = []   
    entity = set() 
    rel = set([0]) 
    time = set()
    for line in open(file_name,'r'):
        items = line.split()
        if len(items) == 4:
            head,r,tail,t = [int(item) for item in items] 
            entity.add(head); entity.add(tail); rel.add(r); time.add(t)  
            quadruples.append((head,r,tail,t,t)) 
        else:
            head,r,tail,tb,te = [int(item) for item in items]  
            entity.add(head); entity.add(tail); rel.add(r); time.add(tb); time.add(te)  
            quadruples.append((head,r,tail,tb,te))  
    return entity,rel,time,quadruples

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2))) 
    return alignment_pair

def get_matrix(triples,entity,rel,time):
    ent_size = max(entity) + 1  
    rel_size = max(rel) + 1   
    time_size = max(time) + 1
    logger.info("entity size & relation_size & timestamp_size: %d, %d, %d." %(ent_size,rel_size,time_size))
    adj_matrix = sp.lil_matrix((ent_size,ent_size))  
    adj_features = sp.lil_matrix((ent_size,ent_size))  
    rel_features = sp.lil_matrix((ent_size,rel_size))  
    time_dict = {}
    
    for i in range(max(entity)+1):
        adj_features[i,i] = 1 
        time_dict[i] = []

    for head,r,tail,tb,te in triples:
        adj_matrix[head,tail] = 1; adj_matrix[tail,head] = 1;  
        adj_features[head,tail] = 1; adj_features[tail,head] = 1;  
        rel_features[head,r] = 1; rel_features[tail,r] = 1  
        if (te==tb):
            time_dict[head].append(tb); time_dict[tail].append(tb)
        else:
            time_dict[head].append(tb); time_dict[tail].append(tb)
            time_dict[head].append(te); time_dict[tail].append(te)

    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(rel_features)

    adj_matrix = sp2torch_sparse(adj_matrix)
    adj_features = sp2torch_sparse(adj_features)
    rel_features = sp2torch_sparse(rel_features)

    return adj_matrix,adj_features,rel_features,time_dict

def load_data(path, ratio=1000):
    entity1,rel1,time1,quadruples1 = load_quadruples(path + 'triples_1')
    entity2,rel2,time2,quadruples2 = load_quadruples(path + 'triples_2')

    train_pair = load_alignment_pair(path + 'sup_pairs')
    dev_pair = load_alignment_pair(path + 'ref_pairs')
    dev_pair = train_pair[ratio:]+dev_pair
    train_pair = train_pair[:ratio]
    
    all_pair = train_pair + dev_pair
   
    adj_matrix,adj_features,rel_features,time_dict= get_matrix(quadruples1+quadruples2,entity1.union(entity2),rel1.union(rel2),time1.union(time2))

    return np.array(train_pair),np.array(dev_pair),np.array(all_pair),adj_matrix,adj_features,rel_features,time_dict


thread_num=30 

def list2dict(time_list):
    dic={}
    for i in time_list:
        dic[i]=time_list.count(i)
    return dic

def sim_matrix(t1,t2):
    size_t1 = len(t1)
    size_t2 = len(t2)
    matrix = np.zeros((size_t1,size_t2))
    for i in range(size_t1):
        len_a = sum(t1[i].values())
        for j in range(size_t2):
            len_b = sum(t2[j].values())
            len_ab = len_a + len_b
            set_ab = {}
            set_ab = t1[i].keys() & t2[j].keys()
            if (len(set_ab)==0):
                matrix[i,j] = 0
                continue
            count = 0
            for k in set_ab:
                count = count + (min(t1[i][k],t2[j][k])-1)
            count = len(set_ab) + count
            matrix[i,j] = (count*2) / len_ab
    return matrix

def div_array(arr,n):
    arr_len = len(arr)
    k = arr_len // n
    ls_return = []
    for i in range(n-1):
        ls_return.append(arr[i*k:i*k+k])
    ls_return.append(arr[(n-1)*k:])
    return ls_return

def thread_sim_matrix(t1,t2):
    pool = Pool(processes=thread_num)
    reses = list()
    tasks_t1 = div_array(t1,thread_num)    
    for task_t1 in tasks_t1:
        reses.append(pool.apply_async(sim_matrix,args=(task_t1,t2)))
    pool.close()
    pool.join()
    matrix = None
    for res in reses:
        val = res.get()
        if matrix is None:
            matrix = np.array(val)
        else:
            matrix = np.concatenate((matrix,val),axis=0)
    return matrix
    
def pair_simt(time_dict,pair):
    t1 = [list2dict(time_dict[e1]) for e1, e2 in pair]
    t2 = [list2dict(time_dict[e2]) for e1, e2 in pair]
    m = thread_sim_matrix(t1,t2)
    return m

def get_simt(file_name,time_dict,dev_pair):
    if os.path.exists(file_name):
        pair_mt = np.load(file_name)
    else:
        pair_mt = pair_simt(time_dict,dev_pair)
        np.save(file_name,pair_mt)
    return pair_mt
