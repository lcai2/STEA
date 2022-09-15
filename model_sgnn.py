import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGNN(nn.Module):
    def __init__(self,
                 node_size,
                 rel_size,
                 hidden_size,
                 dropout_rate,
                 depth,
                 device):
        super(SimpleGNN, self).__init__()  
        self.node_size = node_size
        self.rel_size = rel_size
        self.hidden_size = hidden_size
        self.dropout = dropout_rate
        self.depth = depth
        self.device = device

        self.ent_emb = nn.Embedding(node_size,hidden_size)
        self.rel_emb = nn.Embedding(rel_size,hidden_size)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, data):
        adj = data[0]
        ent_ent = data[1]
        ent_rel = data[2]

        adj = adj.to(device=self.device)
        ent_ent = ent_ent.to(device=self.device)
        ent_rel = ent_rel.to(device=self.device)

        he_emb = self.ent_emb.weight
        hr_emb = self.rel_emb.weight  
        
        he = torch.matmul(ent_ent, he_emb)
        hr = torch.matmul(ent_rel, hr_emb)
        h = torch.cat([he,hr],-1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        hg = h 
        for i in range(self.depth-1):
            h = torch.matmul(ent_ent, h)            
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)            
            hg = torch.cat([hg,h],-1)           

        h_mul = hg 
        return h_mul

class Alignment_loss(nn.Module):
    def __init__(self,
                 gamma,
                 batch_size,
                 device
                 ):

        super(Alignment_loss, self).__init__()

        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

    def forward(self, outfeature, trainset):
        h = outfeature.to(device=self.device)
        set = torch.as_tensor(trainset).to(device=self.device)
        def _cosine(x):
            dot1 = torch.matmul(x[0], x[1], axes=1)
            dot2 = torch.matmul(x[0], x[0], axes=1)
            dot3 = torch.matmul(x[1], x[1], axes=1)
            max_ = torch.maximum(torch.sqrt(dot2 * dot3), torch.epsilon())
            return dot1 / max_
    
        def l1(ll,rr):
            return torch.sum(torch.abs(ll-rr),axis=-1,keepdims=True)
    
        def l2(ll,rr):
            return torch.sum(torch.square(ll-rr),axis=-1,keepdims=True)
        
        l,r,fl,fr = [h[set[:,0]],h[set[:,1]],h[set[:,2]],h[set[:,3]]]
        loss = F.relu(self.gamma + l1(l,r) - l1(l,fr)) + F.relu(self.gamma + l1(l,r) - l1(fl,r))
        loss_avg = torch.sum(loss,0,True) / self.batch_size
        return loss_avg
