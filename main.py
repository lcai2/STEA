import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot

from tqdm import *
from utils import *
from CSLS import *
from model_sgnn import *

from log import set_file_handler, logger
from args import args

main_st = time.time()

data_path = "./data/"
result_path = "./result/" 
dataset = args.dataset + "/"
ratio=args.seed
set_file_handler(result_path,args.dataset,str(args.seed),"STEA")
logger.info(args)
random.seed(100)

hidden_size = args.dim
gamma = args.gamma
lr = args.lr
dropout_rate = args.dropout
alpha = args.alpha 
nthread = args.nthread
depth = args.depth
device = torch.device('cuda:'+args.gpu)
unsupervised = args.unsupervised
unsupervised = False

if(unsupervised == True):
    logger.info("unsupervised")
else:
    logger.info("supervised")


train_pair,dev_pair,all_pair,adj_matrix,adj_features,rel_features,time_dict = load_data(data_path+dataset,ratio=ratio)

node_size = adj_features.shape[0] 
rel_size = rel_features.shape[1] 
batch_size = node_size

if (unsupervised):
    file_name = data_path+dataset+"simt/unsup_seeds_"+ dataset[0:-1] + ".npy"
    train_pair = np.load(file_name)

    file_name = data_path+dataset+"simt/unsup_dev_"+ dataset[0:-1] + ".npy"
    dev_pair = np.load(file_name)

    file_name = data_path+dataset+"simt/simt_"+ dataset[0:-1] + "_unsup_dev.npy"
    pair_mt = get_simt(file_name,time_dict,dev_pair)

else:
    file_name = data_path+dataset+"simt/simt_"+ dataset[0:-1] +"_" + str(ratio) + ".npy"
    pair_mt = get_simt(file_name,time_dict,dev_pair)

logger.info("train_pair_len: "+str(len(train_pair)))
logger.info("dev_pair_len: "+str(len(dev_pair)))
logger.info("all_pair_len: "+str(len(all_pair)))

def get_embedding():
    inputs = [adj_matrix,adj_features,rel_features]
    model.eval()
    with torch.no_grad():
        out_features = model(inputs)
    return out_features

def CSLS_test(m = pair_mt, alpha = alpha, thread_number = nthread, csls=10,accurate = True):
    vec = get_embedding()
    vec = vec.cpu().detach().numpy()
    Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
    Rvec = np.array([vec[e2] for e1, e2 in dev_pair])
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    pre_set,hits1 = eval_alignment_by_sim_mat(Lvec, Rvec, alpha, m, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
    return pre_set,hits1

def get_train_set(batch_size = batch_size):
    negative_ratio =  batch_size // len(train_pair) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_pair,axis=0),axis=0,repeats=negative_ratio),newshape=(-1,2))
    np.random.shuffle(train_set)
    train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set,np.random.randint(0,node_size,train_set.shape)],axis = -1) 
    return train_set

model = SimpleGNN( node_size = node_size,
                 rel_size = rel_size,
                 hidden_size = hidden_size,
                 dropout_rate = dropout_rate,
                 depth = depth,
                 device = device)

model.to(device=device)

criterion = Alignment_loss( gamma= gamma,
                            batch_size= batch_size,
                            device=device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1) 
np.random.shuffle(rest_set_2)

itera =5
epoch = 1200
model.train()
for turn in range(itera):
    logger.info("iteration %d start."%turn)

    for i in range(epoch):
        train_set = get_train_set()
        inputs = [adj_matrix,adj_features,rel_features]

        optimizer.zero_grad()
        output = model(inputs)
        loss_train = criterion(output, train_set)
        loss_train.backward()
        optimizer.step()

        if i%300==299:
            CSLS_test()

    new_pair = []
    vec = get_embedding()
    vec = vec.cpu().detach().numpy()
    Lvec = np.array([vec[e] for e in rest_set_1])
    Rvec = np.array([vec[e] for e in rest_set_2])
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)

    t1 = [list2dict(time_dict[e1]) for e1 in rest_set_1]
    t2 = [list2dict(time_dict[e2]) for e2 in rest_set_2]
    m1 = thread_sim_matrix(t1,t2)
    m2 = thread_sim_matrix(t2,t1)

    A,_ = eval_alignment_by_sim_mat(Lvec, Rvec, alpha, m1, [1, 5, 10], nthread,10,True,False) # 花时间
    B,_ = eval_alignment_by_sim_mat(Rvec, Lvec, alpha, m2, [1, 5, 10], nthread,10,True,False)
    A = sorted(list(A)); B = sorted(list(B))
    for a,b in A:
        if  B[b][1] == a:
            new_pair.append([rest_set_1[a],rest_set_2[b]])
    logger.info("generate new semi-pairs: %d." % len(new_pair))

    train_pair = np.concatenate([train_pair,np.array(new_pair)],axis = 0)
    unsuppervised = False
    for e1,e2 in new_pair:
        if e1 in rest_set_1:
            rest_set_1.remove(e1)

    for e1,e2 in new_pair:
        if e2 in rest_set_2:
            rest_set_2.remove(e2)

main_cost = time.time()-main_st
logger.info("all cost time: %.2fs." % main_cost)


