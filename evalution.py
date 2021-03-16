import numpy as np, scipy.stats as st
import scipy as sp
import scipy.stats
import torch
from torch.autograd import Variable
import gc
import math
'''
def mean_confidence_interval(data, confidence=0.95,num_parts = 5):
    part_len = len(data)//num_parts
    estimations = []
    for i in range(num_parts):
        est = np.mean(data[part_len*i:part_len*(i+1)])
        estimations.append(est)
    a = 1.0*np.array(estimations)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h
def validate_mrr(network,k,test_loader,device):
    network.eval()
    losses = []
    with torch.no_grad():
        for user_batch_ix,item_batch_ix, mask_batch_ix in test_loader:
            user_batch_ix = Variable(user_batch_ix).to(device)
            item_batch_ix = Variable(item_batch_ix).to(device)
            mask_batch_ix = Variable(mask_batch_ix).to(device)

            logp_seq = network(user_batch_ix, item_batch_ix)
            # compute loss
            predictions_logp = logp_seq[:,-2]
            _,ind = torch.topk(predictions_logp, k,dim=-1)
            mrr = torch.zeros(predictions_logp.size())
            #scatter_(input, dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中。
            mrr.scatter_(-1,ind.cpu(),1/torch.range(1,k).repeat(*ind.size()[:-1],1).type(torch.FloatTensor).cpu())
            actual_next_tokens = item_batch_ix[:, -1]
            # torch.gather(input, dim, index, out=None)
            logp_next = torch.gather(mrr.to(device)*mask_batch_ix[:, -2,None], dim=1, index=actual_next_tokens[:,None])
#             if mask_batch_ix[:,-2].sum() >0:
            loss = logp_next.sum()/mask_batch_ix[:,-2].sum()
            losses.append(loss.cpu().data.numpy())
            torch.cuda.empty_cache()
            gc.collect() 
    m, h = mean_confidence_interval(losses)
    return m, h

# HitRatio
def validate_recall(network,k,test_loader,device):
    torch.cuda.empty_cache()
    gc.collect() 
    
    network.eval()
    losses = []
    with torch.no_grad():
        for user_batch_ix,item_batch_ix, mask_batch_ix in test_loader:
            user_batch_ix = Variable(user_batch_ix).to(device)
            item_batch_ix = Variable(item_batch_ix).to(device)
            mask_batch_ix = Variable(mask_batch_ix).to(device)
            logp_seq = network(user_batch_ix, item_batch_ix)
            # compute loss
            predictions_logp = logp_seq[:, -2]
            minus_kth_biggest_logp,_ = torch.kthvalue(-predictions_logp.cpu(), k,dim=-1,keepdim=True)
            prediicted_kth_biggest = (predictions_logp>(-minus_kth_biggest_logp.to(device)))\
                                        .type(torch.FloatTensor).to(device)
            actual_next_tokens = item_batch_ix[:, -1]

            logp_next = torch.gather(prediicted_kth_biggest*mask_batch_ix[:, -2,None], dim=1, index=actual_next_tokens[:,None])
            loss = logp_next.sum()/mask_batch_ix[:,-2].sum()
            losses.append(loss.cpu().data.numpy())
    torch.cuda.empty_cache()
    gc.collect() 
    m, h = mean_confidence_interval(losses)
    return m, h
'''
def Metrics_at_k(network,k,test_loader,device):
    torch.cuda.empty_cache()
    gc.collect()
    network.eval()
    mrrs,hits,ndcgs = [],[],[]
    with torch.no_grad():
        for user_batch_ix, item_batch_ix, mask_batch_ix in test_loader:
            user_batch_ix = Variable(user_batch_ix).to(device)
            item_batch_ix = Variable(item_batch_ix).to(device)
            mask_batch_ix = Variable(mask_batch_ix).to(device)
            logp_seq = network(user_batch_ix,item_batch_ix)
            predictions_logp = logp_seq[:, -2]
            actual_next_tokens = item_batch_ix[:, -1]

            #######################
            ##    compute mrr    ##
            #######################
            
            _,ind = torch.topk(predictions_logp, k,dim=-1)
            mrr = torch.zeros(predictions_logp.size())
            #scatter_(input, dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中。
            mrr.scatter_(-1,ind.cpu(),1/torch.range(1,k).repeat(*ind.size()[:-1],1).type(torch.FloatTensor).cpu())
            # torch.gather(input, dim, index, out=None)
            mrr_next = torch.gather(mrr.to(device)*mask_batch_ix[:, -2,None], dim=1, index=actual_next_tokens[:,None])
            _mrr = mrr_next.sum()/mask_batch_ix[:,-2].sum()
            mrrs.append(_mrr.cpu().data.numpy())
            
            #######################
            ##    compute hit    ##
            #######################
            minus_kth_biggest_logp,_ = torch.kthvalue(-predictions_logp.cpu(), k,dim=-1,keepdim=True)
            prediicted_kth_biggest = (predictions_logp>(-minus_kth_biggest_logp.to(device)))\
                                        .type(torch.FloatTensor).to(device)
            hit_next = torch.gather(prediicted_kth_biggest*mask_batch_ix[:, -2,None], dim=1, index=actual_next_tokens[:,None])
            _hit = hit_next.sum() / mask_batch_ix[:,-2].sum()
            hits.append(_hit.cpu().data.numpy())

            #######################
            ##   compute ndcg   ##
            #######################

            ndcg = torch.zeros(predictions_logp.size())
            ndcg.scatter_(-1,ind.cpu(),torch.range(1,k).repeat(*ind.size()[:-1],1).type(torch.FloatTensor).cpu())
            ndcg_next = torch.gather(ndcg.to(device)*mask_batch_ix[:,-2,None],dim = 1,index=actual_next_tokens[:,None])
            #_np = torch.squeeze(ndcg_next).numpy()
            _np = ndcg_next.cpu().data.numpy()
            _ndcg,_len = 0,0
            for id in _np:
                _len += 1
                if id:
                    _ndcg += 1 / math.log(id + 1,2)
            ndcgs.append(float(_ndcg / _len))
    torch.cuda.empty_cache()
    gc.collect() 
    return np.mean(mrrs),np.mean(hits),np.mean(ndcgs)

def print_scores(model,name,test_loader,device,k = 20):
    network = torch.load(model).to(device)
    mrr, hit, ndcg = Metrics_at_k(network,20,test_loader,device)
    print("MRR@{} score for {}: {}".format(k,name,mrr))
    print("Recall@{} score for {}: {}".format(k,name,hit))
    print("NDCG@{} score for {}: {}".format(k,name,ndcg))