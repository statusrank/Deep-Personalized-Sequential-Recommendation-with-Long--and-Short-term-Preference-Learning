from torch.nn.parameter import Parameter
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
class DPSR(nn.Module):
    
    def __init__(self, n_users,n_items, emb_size=1024, hidden_units=1024,output_dropout = 0.8,input_dropout = 0.6,device = 'cpu',
                    k = 4,w = 32,n = 10):
        super(self.__class__, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_units = hidden_units
        if emb_size == None:
            emb_size = hidden_units
        self.emb_size = emb_size
        self.device = device
        # vertical filter
        self.k = k
        # horizontal filter
        self.w = w
        self.n = n
        ## 
        self.user_emb = nn.Embedding(n_users,emb_size)
        self.item_emb = nn.Embedding(n_items,emb_size)
        self.grucell = nn.GRUCell(input_size = emb_size*2,hidden_size = hidden_units)
        torch.nn.init.orthogonal_(self.grucell.weight_ih)
        torch.nn.init.constant_(self.grucell.bias_ih,0)
        self.att_linear = nn.Linear(hidden_units+emb_size*2,emb_size)
        torch.nn.init.orthogonal_(self.att_linear.weight)
        torch.nn.init.constant_(self.att_linear.bias,0)
        self.linear = nn.Linear(self.n + hidden_units*2,n_items)
        self.output_dropout = nn.Dropout(output_dropout)
        self.input_dropout = nn.Dropout(input_dropout)
        self.hcov = nn.Conv1d(1,self.n,self.w)
        self.vcov = nn.Conv1d(self.k,1,1)
        #self.vcov = nn.Conv2d(1,1,(self.k,1),stride = (1,1))
        
    def forward(self, user_vectors, item_vectors):
        batch_size,_ = user_vectors.size()
        user_vectors = user_vectors
        item_vectors = item_vectors
        sequence_size = user_vectors.size()[1]
        
        #users = self.user_dropout(self.user_emb(user_vectors))#.view(-1,sequence_size,self.emb_size)
        users = self.input_dropout(self.user_emb(user_vectors))
        items = self.input_dropout(self.item_emb(item_vectors))#.view(-1,sequence_size,self.emb_size)
        
        h = torch.zeros(batch_size,self.hidden_units).to(self.device)
        h_t = h.unsqueeze(0)
        # horizontal results
        o_t = torch.zeros(1,batch_size,self.n).to(self.device)
        g_t = torch.zeros(1,batch_size,self.hidden_units).to(self.device)
        for i in range(sequence_size):
            attention = F.sigmoid(self.att_linear(torch.cat([users[:,i,:],items[:,i,:],h],dim = -1)))
            gru_input = torch.cat([attention*users[:,i,:],(1-attention)*items[:,i,:]],dim=-1)
            #gru_input = attention*users[:,i,:] + (1 - attention) * items[:,i,:]
            h = self.grucell(gru_input,h)
            #h = self.grucell(items[:,i,:],h)
            v_t = torch.sum(self.hcov(h.unsqueeze(1)),-1)
            o_t = torch.cat([o_t,v_t.unsqueeze(0)],dim = 0)
            h_t = torch.cat([h_t,h.unsqueeze(0)],dim = 0)
            if i - self.k + 1 >= 0:
                p_t = h_t[-self.k:].transpose(0,1)
                #q_t = torch.squeeze(self.vcov(p_t.unsqueeze(1)))
                q_t = torch.squeeze(self.vcov(p_t))
                l_t = h.mul(q_t)
                g_t = torch.cat([g_t,l_t.unsqueeze(0)],dim = 0)
            else:
                g_t = torch.cat([g_t,torch.zeros(1,batch_size,self.hidden_units).to(self.device)],dim = 0)
        hor_input = o_t[1:].transpose(0,1)
        voc_input = g_t[1:].transpose(0,1)
        gru_input = h_t[1:].transpose(0,1)
        ful_input = self.output_dropout(torch.cat([hor_input,gru_input,voc_input],dim = -1))
        #ln_input = self.dropout(h_t[1:].transpose(0,1))
        output_ln = self.linear(ful_input)
        output = F.log_softmax(output_ln, dim=-1)
        return output
