import gc
import os
import numpy as np
import torch, torch.nn as nn
from model import AttentionalLinearGRU
from utils import get_data,get_prepared_data
import torch.utils.data
from evalution import *
from torch.autograd import Variable
import visdom
from tqdm import tqdm

#from IPython.display import clear_output

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.empty_cache()
gc.collect()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
data_path = 'Foursquare/'
dname = 'data.txt'
batch_size = 32


print('loading data')
train_items = np.load(data_path + "lf_train_items.npy")
train_mask = np.load(data_path + "lf_train_mask.npy")
train_users = np.load(data_path + "lf_train_users.npy")
val_items = np.load(data_path + "lf_val_items.npy")
val_mask = np.load(data_path + "lf_val_mask.npy")
val_users = np.load(data_path + "lf_val_users.npy")
test_items = np.load(data_path + "lf_test_items.npy")
test_mask = np.load(data_path + "lf_test_mask.npy")
test_users = np.load(data_path + "lf_test_users.npy")


'''
data = get_data(data_path,dname)
(train_items, train_mask,train_users),\
(val_items, val_mask,val_users),\
(test_items, test_mask,test_users),\
(train_user_idx, train_item_idx, train_feedback) = get_prepared_data(data)
'''
n_users = int(np.max([train_users.max(),test_users.max(),val_users.max()])+1)
n_items = int(np.max([train_items.max() + 1,val_items.max() + 1,test_items.max() + 1]))


def validate_lce(network,val_loader):
        torch.cuda.empty_cache()
        gc.collect()
        network.eval()
        losses = []
        with torch.no_grad():
                for user_batch_ix,item_batch_ix,mask_batch_ix in val_loader:
                        user_batch_ix = Variable(user_batch_ix).to(device)
                        item_batch_ix = Variable(item_batch_ix).to(device)
                        mask_batch_ix = Variable(mask_batch_ix).to(device)
                        logp_seq = network(user_batch_ix,item_batch_ix)

                        # compute loss
                        predictions_logp = logp_seq[:,:-1] * mask_batch_ix[:,:-1,None]
                        actual_next_token = item_batch_ix[:,1:]

                        logp_next = torch.gather(predictions_logp,dim = 2,index = actual_next_token[:,:,None])

                        loss = -logp_next.sum() / mask_batch_ix[:,:-1].sum()
                        losses.append(loss.cpu().data.numpy())
        torch.cuda.empty_cache()
        gc.collect() 
        return np.mean(losses)

def train_network(network,train_loader,val_loader,w = 128,k = 3,emb_size = 1024,num_epoch = 16,opt = None):
        history = []
        path = data_path + "Out/w_" + str(w) + "_k_" + str(k) + '_emb_size_' + str(emb_size)
        # Visdom
        vis = visdom.Visdom(env = "Foursquare_w_" + str(w) + "_k_" + str(k) + '_emb_size_' + str(emb_size))
        vis.line([[0.,0.]],[0],win = 'train_val',opts = dict(title = 'train and validation loss',legend = ['trian_loss','val_loss']))
        vis.line([[0.,0.,0.,0.,0.,0.]],[0],win = 'Metrics',opts = dict(title = 'Metrics at k',legend = ['val_mrr','val_hit','val_ndcg','test_mrr','test_hit','test_ndcg']))
        global_step = 0
        for epoch in range(num_epoch):
                i = 0
                train_loss = []
                for user_batch_ix,item_batch_ix,mask_batch_ix in tqdm(train_loader):
                        network.train()
                        user_batch_ix = Variable(user_batch_ix).to(device)
                        item_batch_ix = Variable(item_batch_ix).to(device)
                        mask_batch_ix = Variable(mask_batch_ix).to(device)
                        logp_seq = network(user_batch_ix, item_batch_ix)
                        # compute loss
                        predictions_logp = logp_seq[:, :-1]*mask_batch_ix[:, :-1,None]
                        actual_next_tokens = item_batch_ix[:, 1:]
                        logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:,:,None])
                        
                        loss = -logp_next.sum()/mask_batch_ix[:, :-1].sum()
                        #print("epoch: {} training loss: {}".format(epoch,loss))
                        train_loss.append(loss.cpu().data.numpy())
                        # train with backprop
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(network.parameters(),5)
                        opt.step()

                        if (i + 1) % 100 == 0:
                                val_loss = validate_lce(network,val_loader)
                                history.append(val_loss)
                                vis.line([[np.mean(train_loss),val_loss]],[global_step],win = 'train_val',update = 'append')
                                global_step += 1
                                train_loss = []
                                #print("-" * 100)
                                #print("epoch: {} validation loss: {}".format(epoch,val_loss))
                                #print("-" * 100)
                        i += 1
                
                if epoch % 1 == 0:
                        f = open(path + '.txt','a')
                        kk = 20
                        val_mrr,val_hit,val_ndcg = Metrics_at_k(network,k = kk,test_loader = val_loader,device = device)
                        print("MRR@{}  for validation: {}".format(kk,val_mrr))
                        print("Hit@{}  for validation: {}".format(kk,val_hit))
                        print("NDCG@{} for validation: {}".format(kk,val_ndcg))
                        f.write('\n')
                        f.write("-"*100)
                        f.write('MRR for validation: ' + str(val_mrr) + '\n')
                        #print("MRR@{} score for validation: {}±{}".format(k,val_mrr,val_mrr_h),file = f)
                        #print("Recall@{} score for validation: {}±{}".format(k,val_recall,val_recall_h),file = f)
                        f.write("Hit for validation: " + str(val_hit) + '\n')
                        f.write("NDCG for validation: " + str(val_ndcg) + '\n')
                        f.write('\n')

                        test_mrr,test_hit,test_ndcg = Metrics_at_k(network,k = kk,test_loader = test_loader,device = device)
                        print("MRR@{}  for test: {}".format(kk,test_mrr))
                        print("Hit@{}  for test: {}".format(kk,test_hit))
                        print("NDCG@{} for test: {}".format(kk,test_ndcg))
                        
                        f.write('MRR for test: ' + str(test_mrr) + '\n')
                        f.write('Hit for test: ' + str(test_hit) + '\n')
                        f.write("NDCG for test:" + str(test_ndcg) + '\n')
                        f.write("-"*100)
                        f.write('\n')
                        f.close()
                        vis.line([[val_mrr,val_hit,val_ndcg,test_mrr,test_hit,test_ndcg]],[epoch],win = 'Metrics',update = 'append')
                        #time.sleep(0.5)
                torch.save(network, data_path + "Out/network_att_w" + str(w) + '_k_'+ str(k)  + '_emb_size' + str(emb_size) + "_" + 
                        str(epoch) + '.p')
        val_loss = validate_lce(network,val_loader)
        history.append(val_loss)
        return history

if __name__ == '__main__':
        
        train_loader = torch.utils.data.DataLoader(\
        torch.utils.data.TensorDataset(\
        *(torch.LongTensor(train_users),torch.LongTensor(train_items),torch.FloatTensor(train_mask))),\
        batch_size=batch_size,shuffle=True)

        val_loader = torch.utils.data.DataLoader(\
        torch.utils.data.TensorDataset(\
        *(torch.LongTensor(val_users),torch.LongTensor(val_items),torch.FloatTensor(val_mask))),\
        batch_size=batch_size,shuffle=True)

        test_loader = torch.utils.data.DataLoader(\
        torch.utils.data.TensorDataset(\
        *(torch.LongTensor(test_users),torch.LongTensor(test_items),torch.FloatTensor(test_mask))),\
        batch_size=batch_size,shuffle=True)
        
        #history = train_network(network.to(device),train_loader,val_loader)
        #torch.save(network, data_path + "network_bn_bn5mh_att_end.p")
        emb_size = [1024,512,256,128,64]
        w = [128,64,32,16,8]
        k = [2,3,4,5,6,7]
        for _emb in emb_size:
                for _k in k:
                        for _w in w:
                                network = AttentionalLinearGRU(n_users = n_users,n_items = n_items,emb_size = _emb,
                                                                w = _w,k = _,device = device).to(device)
                                opt = torch.optim.Adagrad(network.parameters(),lr = 0.01,weight_decay = 1e-5)
                                history = train_network(network.to(device),train_loader,val_loader,w = _w,k = _k,emb_size = emb_size,opt = opt)
        # loading model and test
        #kk = 20
        #print_scores(data_path + "network_bn_bn5mh_att_9.p", 'validation', val_loader, device, kk)

        #print_scores(data_path + "network_bn_bn5mh_att_9.p", 'test', test_loader, device, kk)
