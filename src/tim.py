from src.util import get_mi, get_cond_entropy, get_entropy, get_one_hot
from tqdm import tqdm
import torch
import time
import torch.nn.functional as F
import logging
import os
def config():
    temp = 15 # hyper-parameter 
    loss_weights = [0.7, 1.0, 0.1]  # [Xent, H(Y), H(Y|X)] # hyper-parameter 
    lr = 1e-4
    iter = 15
    alpha = 1.0

class TIM(object):
    def __init__(self, model,test_file,first):
        self.lr = 1e-5
        is_test = 0
        if is_test:
            self.test_init(test_file,first)
        else:
            self.eval_init(test_file,first)
        self.temp = 0.1 # different model may need different temp value
        self.loss_weights = [0.1, 0.1, 5] # [0.1, 0.1, 1]
        #self.loss_weights = [1, 1, 1] # [0.1, 0.1, 1]
        self.model = model
        self.init_info_lists()
        self.alpha = 1.0
    def test_init(self,test_file,first):
        self.iter = 20
    def eval_init(self,test_file,first):
        if first==0:
            if test_file=='BUK1_20181011_001004':
                self.iter = 30
            elif test_file=='BUK1_20181013_023504':
                self.iter= 30 # 15-->0
            elif test_file=='BUK4_20161011_000804':
                self.iter = 40 
            elif test_file=='BUK4_20171022_004304a' :
                self.iter = 7
            elif test_file=='BUK5_20161101_002104a':
                self.iter = 7
            elif test_file =='BUK5_20180921_015906a':
                self.iter = 15
            elif test_file =='a1':
                self.iter = 15
            elif test_file =='n1':
                self.lr = 3e-5
                self.iter = 35 # 25
        # self.iter = 10
        elif first==1:
            if test_file=='BUK1_20181011_001004':
                self.iter = 25
            elif test_file=='BUK1_20181013_023504':
                self.lr = 2e-5
                self.iter= 15 # 15-->0
            elif test_file=='BUK4_20161011_000804':
                self.iter = 25 
            elif test_file=='BUK4_20171022_004304a' :
                self.iter = 8
            elif test_file=='BUK5_20161101_002104a':
                self.iter = 5
            elif test_file =='BUK5_20180921_015906a':
                self.iter = 40
            elif test_file =='a1':
                self.iter = 35
            elif test_file =='n1':
                self.lr = 1e-4
                self.iter = 25 # 25
        else:
            #self.iter = 10
            if test_file=='BUK1_20181011_001004':
                self.iter = 25
            elif test_file=='BUK1_20181013_023504':
                self.lr = 2e-5
                self.iter= 15 # 15-->0
            elif test_file=='BUK4_20161011_000804':
                self.iter = 25  
            elif test_file=='BUK4_20171022_004304a' :
                self.iter = 8
            elif test_file=='BUK5_20161101_002104a':
                self.iter = 5
            elif test_file =='BUK5_20180921_015906a':
                self.iter = 40
            elif test_file =='a1':
                self.iter = 35
            elif test_file =='n1':
                self.lr = 1e-4
                self.iter = 25 # 25
    def init_info_lists(self):
        self.timestamps = []
        self.mutual_infos = []
        self.entropy = []
        self.cond_entropy = []
        self.test_probs = []
        self.losses = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = samples.matmul(self.weights.transpose(1,2)) # 
        cont = samples.norm(dim=2).unsqueeze(2).matmul(self.weights.norm(dim=2).unsqueeze(2).transpose(1,2))
        return logits

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]
        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds
    def compute_FB_param(self, features_q):
        logits_q = self.get_logits(features_q).detach() # logits: according to W, calculate results
        q_probs = logits_q.softmax(2) # predict probability
        #probas = self.get_probas(features_q).detach()
        b = q_probs[:,:,0]>0.5
        # print(1.0*b.sum(1)/a.shape[1])
        pos = 1.0*b.sum(1)/q_probs.shape[1]
        neg = 1.0 -pos
        pos = pos.unsqueeze(1)
        neg = neg.unsqueeze(1)
        self.FB_param2 = torch.cat([pos,neg],1)
        self.FB_param = (q_probs).mean(dim=1)
    def init_weights(self, support, query, y_s): 
        self.model.eval()
        t0 = time.time()
        n_tasks = support.size(0)
        # print('n_tasks ',n_tasks)
        one_hot = get_one_hot(y_s) # get one-hot vector
        counts = one_hot.sum(1).view(n_tasks, -1, 1) # 
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts
        self.record_info(new_time=time.time()-t0,
                         support=support,
                         query=query,
                         y_s=y_s)
        self.model.train()

    def compute_lambda(self, support, query, y_s): 
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]

        updates :
            self.loss_weights[0] : Scalar
        """
        self.N_s, self.N_q = support.size(1), query.size(1)
        self.num_classes = torch.unique(y_s).size(0) 
        if self.loss_weights[0] == 'auto':
            self.loss_weights[0] = (1 + self.loss_weights[2]) * self.N_s / self.N_q

    def record_info(self, new_time, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        logits_q = self.get_logits(query).detach() 
        logits_q = logits_q/self.temp
        preds_q = logits_q.argmax(2) 
        q_probs = logits_q.softmax(2) 
        self.timestamps.append(new_time) 
        self.mutual_infos.append(get_mi(probs=q_probs)) 
        self.entropy.append(get_entropy(probs=q_probs.detach())) # # H(Y_q)
        self.cond_entropy.append(get_cond_entropy(probs=q_probs.detach())) # # H(Y_q | X_q)
        self.test_probs.append(q_probs.view(-1,2)[:,0]) 

    def get_logs(self):
        self.test_probs = self.test_probs[-1].cpu().numpy() # use the last as results
        self.cond_entropy = torch.cat(self.cond_entropy, dim=1).cpu().numpy()
        self.entropy = torch.cat(self.entropy, dim=1).cpu().numpy()
        self.mutual_infos = torch.cat(self.mutual_infos, dim=1).cpu().numpy()
        self.W = self.weights
        return {'timestamps': self.timestamps, 'mutual_info': self.mutual_infos,
                'entropy': self.entropy, 'cond_entropy': self.cond_entropy, 'losses': self.losses,
                'test': self.test_probs,'W': self.W}

    def run_adaptation(self, support, query, y_s):
        pass


class TIM_GD(TIM):
    def __init__(self, model,test_file,first):
        super().__init__(model=model,test_file=test_file,first=first)

    def run_adaptation(self, support, query, y_s):
        t0 = time.time()
        self.weights.requires_grad_() # W
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        self.model.train()
        l3 = 0.2
        for i in tqdm(range(self.iter)): # 
            logits_s = self.get_logits(support)  #
            logits_q = self.get_logits(query) # 
            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0) 
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0) # H(Y|X)
            q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0) #H(Y)
            b = q_probs[:,:,0]>0.5
            pos = 1.0*b.sum(1)/q_probs.shape[1]
            neg = 1.0 -pos
            pos = pos.unsqueeze(1)
            neg = neg.unsqueeze(1)
            F2 = torch.cat([pos,neg],1)
            marginal = q_probs.mean(dim=1) # n_task,2
            div_kl = F.kl_div(marginal, self.FB_param, reduction='sum')
            div_kl2 = F.kl_div(F2,self.FB_param2,reduction='sum')
            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent) + l3*div_kl   
            # loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i > 2:
                self.compute_FB_param(query)
                l3 += 0.1
            t1 = time.time()
            self.model.eval()
            self.record_info(new_time=t1-t0,
                             support=support,
                             query=query,
                             y_s=y_s)
            self.model.train()
            t0 = time.time()

    def run_adaptation_model_w(self, support, query, y_s,nums):
        t0 = time.time()
        self.weights.requires_grad_() 
        optimizer = torch.optim.Adam([{'params': self.model.encoder.parameters()},{'params': self.weights}], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        self.model.train()
        l3 = 0.2
        if nums<self.iter:
            logits_s = self.get_logits(support)  
            logits_q = self.get_logits(query) 
            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0) 
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0) # H(Y|X)
            q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0) #H(Y)
            b = q_probs[:,:,0]>0.5
            pos = 1.0*b.sum(1)/q_probs.shape[1]
            neg = 1.0 -pos
            pos = pos.unsqueeze(1)
            neg = neg.unsqueeze(1)
            F2 = torch.cat([pos,neg],1)
            marginal = q_probs.mean(dim=1) # n_task,2
            div_kl = F.kl_div(marginal.softmax(dim=-1).log(), self.FB_param.softmax(dim=-1), reduction='sum')
            div_kl2 = F.kl_div(F2.softmax(dim=-1).log(),self.FB_param2.softmax(dim=-1),reduction='sum')
            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent) + l3*div_kl + div_kl2  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if nums>2:
                self.compute_FB_param(query)
                l3 += 0.1
            t1 = time.time()
            self.model.eval()
            self.record_info(new_time=t1-t0,
                             support=support,
                             query=query,
                             y_s=y_s)
            
            self.model.train()
            t0 = time.time()
        return self.model,nums+1,self.iter
            