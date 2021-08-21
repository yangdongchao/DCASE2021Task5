import numpy as np
from sacred import Ingredient
from src.util import warp_tqdm, compute_confidence_interval, load_checkpoint
from src.util import load_pickle, save_pickle
from src.util import save_checkpoint
import os
import torch
import collections
import torch.nn.functional as F
from src.tim import TIM, TIM_GD
from src.datasets.Datagenerator import Datagen_test
from src.datasets.batch_sampler import EpisodicBatchSampler
from src.util import warp_tqdm, get_metric, AverageMeter,euclidean_dist,save_plot_data
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn
import random
def config():
    number_tasks = 10000
    n_ways = 5
    query_shots = 15
    method = 'baseline'
    model_tag = 'best'
    target_data_path = None  # Only for cross-domain scenario
    target_split_dir = None  # Only for cross-domain scenario
    plt_metrics = ['accs']
    shots = [1, 5]
    used_set = 'test'  # can also be val for hyperparameter tuning
    fresh_start = False
seed = 2021
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.number_tasks = 10000
        self.n_ways = 2
        self.query_shots = 15
        self.method = 'tim_gd' #tim_gd
        self.model_tag = 'best'
        self.plt_metrics = ['accs']
        self.shots = [5]
        self.used_set = 'test'
        self.fresh_start = True

    def align_predict_and_query(self,pre_predict,num_query,k_q): # align predict and true the number of query
        if pre_predict.shape[0]>num_query: # deal predict length
            n_ = pre_predict.shape[0]//k_q
            prob_final = pre_predict[:(n_-1)*k_q]
            n_last = num_query - prob_final.shape[0]
            prob_final = np.concatenate((prob_final,pre_predict[-n_last:]))
        else:
            prob_final = pre_predict
        return prob_final

    def best_lower_bound_search(self,pos_num,x_query_labels):
        l = 0.0
        r=0.5
        iterate_num = 0
        ans = (l+r)/2.0
        while iterate_num<50:
            mid = (l+r)/2.0
            x_query_neg_index = torch.where(x_query_labels<mid,torch.ones(x_query_labels.shape[0]),torch.zeros(x_query_labels.shape[0])) # 选出预测为正的样本
            x_q_trian_label = x_query_labels[x_query_neg_index==1]
            if x_q_trian_label.shape[0]>pos_num:
                r = mid
            elif x_q_trian_label.shape[0]<pos_num:
                ans = mid
                l = mid
            else:
                return mid
            iterate_num +=1
        return ans
    def best_upper_bound_search(self,pos_num,x_query_labels):
        l = 0.5
        r=1.0
        iterate_num = 0
        ans = 0.5
        while iterate_num<50:
            mid = (l+r)/2.0
            x_query_pos_index = torch.where(x_query_labels>mid,torch.ones(x_query_labels.shape[0]),torch.zeros(x_query_labels.shape[0])) # 选出预测为正的样本
            x_q_trian_label = x_query_labels[x_query_pos_index==1]
            if x_q_trian_label.shape[0]>pos_num:
                ans = mid
                l = mid
            elif x_q_trian_label.shape[0]<pos_num:
                r = mid
            else:
                return mid
            iterate_num +=1
        return ans
        
    def from_teacher_to_student(self,student,save_dict,W,pre_predict,num_query,k_q,iter_num,test_file,loaders_dic): # control the KD learning
        prob_final = self.align_predict_and_query(pre_predict,num_query,k_q)
        x_query = save_dict['x_query']  
        x_pos_train = save_dict['x_pos'] # support sample, thoes sapmle are very small, just 5 shots
        x_query = torch.from_numpy(x_query)
        assert prob_final.shape[0]==x_query.shape[0]
        x_query_labels = torch.from_numpy(prob_final) 
        hyper_high_confident_num = 400 # this is a hyper-parameter, you can set it by yourself
        thres_pos = self.best_upper_bound_search(hyper_high_confident_num,x_query_labels) 
        x_query_pos_index = torch.where(x_query_labels>=thres_pos,torch.ones(x_query_labels.shape[0]),torch.zeros(x_query_labels.shape[0])) # 选出预测为正的样本
        x_query_tr_pos = x_query[x_query_pos_index==1] # the query predict as positive
        x_query_tr_pos_label = x_query_labels[x_query_pos_index==1] # get thier predict probability
        thresh_neg = self.best_lower_bound_search(x_query_tr_pos_label.shape[0]+x_pos_train.shape[0],x_query_labels)
        x_query_neg_index = torch.where(x_query_labels<thresh_neg,torch.ones(x_query_labels.shape[0]),torch.zeros(x_query_labels.shape[0])) # 选出预测为正的样本
        x_q_trian = x_query[x_query_neg_index==1] # get exaplem by index,note x_query represent mel spectrum
        x_q_fake_label = x_query_labels[x_query_neg_index==1] # thier predict label by previous model
        x_q_trian = torch.cat([x_q_trian,x_query_tr_pos],0) # mix up pos sample and random sample
        x_q_fake_label = torch.cat([x_q_fake_label,x_query_tr_pos_label],0) # label
        assert x_q_trian.shape[0]==x_q_fake_label.shape[0] # judge the number of label and train number is same
        x_pos_train = torch.from_numpy(x_pos_train) # convert numpy to tensor
        x_pos_label = torch.ones(x_pos_train.shape[0]) # thier label is certainty, 1
        assert x_pos_train.shape[0]==x_pos_label.shape[0] # judege
        x_train = torch.cat([x_q_trian,x_pos_train],0) # add support sample, we need those label,because thier label is true
        y_train = torch.cat([x_q_fake_label,x_pos_label],0) # finally, we have get all the sample for student to train.
        model_path = '/home/ydc/DACSE2021/sed-tim-base/check_point/' + str(iter_num)
        student = self.train_student(x_train,y_train,student,W,x_pos_train,x_pos_label,model_path) #  get student model. note, we need x_pos to updata student according to W.
        # torch.save(student,'/home/ydc/DACSE2021/task5/sed-tim/check_point/model/best_55per.pth')
        model_path = '/home/ydc/DACSE2021/sed-tim-base/check_point/' + str(iter_num)+'/'
        extracted_features_dic = self.extract_features(model=student, model_path=model_path, model_tag='student',
                                    used_set=test_file, fresh_start=True,loaders_dic=loaders_dic,test_student=1) # use student model to extract feature

        predict = None
        for shot in self.shots: # 5 shot
            tasks = self.generate_tasks(extracted_features_dic=extracted_features_dic,k_q=k_q)  
            logs = self.run_task(task_dic=tasks,
                                 model=student,test_file=test_file,first=iter_num)
            # l2n_mean, l2n_conf = compute_confidence_interval(logs['acc'][:, -1])
            predict = logs['test']
            W = logs['W']
        return predict,W,student
        
    def run_full_evaluation(self,test_file, model, model_path,student_model,hdf_eval,conf,k_q,iter_num):
        """
        Run the evaluation over all the tasks in parallel
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            method : Which method to use for inference ("baseline", "tim-gd" or "tim-adm")
            shots : Number of support shots to try

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        print("=> Runnning full evaluation with method: {}".format(self.method))
        load_checkpoint(model=model, model_path=model_path, type=self.model_tag)
        load_checkpoint(model=student_model,model_path=model_path,type=self.model_tag)
        # Get loaders
        loaders_dic,save_dict = self.get_loaders(hdf_eval=hdf_eval,conf=conf) 
        # Extract features (just load them if already in memory)
        extracted_features_dic = self.extract_features(model=model, model_path=model_path, model_tag=self.model_tag,
                                    used_set=test_file, fresh_start=self.fresh_start,loaders_dic=loaders_dic)
        results = []
        predict = None
        for shot in self.shots: # 5 shot
            tasks = self.generate_tasks(extracted_features_dic=extracted_features_dic,k_q=k_q)  
            logs = self.run_task(task_dic=tasks,
                                 model=model,test_file=test_file,first=0)
            # l2n_mean, l2n_conf = compute_confidence_interval(logs['acc'][:, -1])
            predict = logs['test']
            W = logs['W']
            results.append(predict)
        if iter_num ==0:
            return results, self.number_tasks
        results = []
        for i in range(iter_num):
            predict,W,student_model= self.from_teacher_to_student(student_model,save_dict,W,predict,self.number_tasks,k_q,i+1,test_file,loaders_dic)
            if i == iter_num-1:
                results.append(predict)
        
        return results, self.number_tasks
    def run_full_evaluation_model_w(self,test_file, model, model_path,student_model,hdf_eval,conf,k_q,iter_num):
        print("=> Runnning full evaluation with method: {}".format(self.method))
        # Load pre-trained model
        load_checkpoint(model=model, model_path=model_path, type=self.model_tag)
        load_checkpoint(model=student_model,model_path=model_path,type=self.model_tag)
        # Get loaders
        loaders_dic,save_dict = self.get_loaders(hdf_eval=hdf_eval,conf=conf) 
        # Extract features (just load them if already in memory)
        extracted_features_dic = self.extract_features(model=model, model_path=model_path, model_tag=self.model_tag,
                                    used_set=test_file, fresh_start=self.fresh_start,loaders_dic=loaders_dic)
        results = []
        predict = None
        for shot in self.shots: # 5 shot
            tasks = self.generate_tasks(extracted_features_dic=extracted_features_dic,k_q=k_q)  
            logs = self.run_task_model_w(task_dic=tasks,
                                 model=model,test_file=test_file,first=0,loaders_dic=loaders_dic,k_q=k_q,model_path=model_path)
            # l2n_mean, l2n_conf = compute_confidence_interval(logs['acc'][:, -1])
            predict = logs['test']
            W = logs['W']
            results.append(predict)
        if iter_num ==0:
            return results, self.number_tasks
        results = []
        for i in range(iter_num):
            predict,W,student_model= self.from_teacher_to_student(student_model,save_dict,W,predict,self.number_tasks,k_q,i+1,test_file,loaders_dic)
            if i == iter_num-1:
                results.append(predict)
        
        return results, self.number_tasks
    def train_student(self,train_data,label,student,W,x_pos_train,x_pos_label,model_path):
        losses = AverageMeter()
        top1 = AverageMeter()
        device = 'cuda'
        lr = 0.00001
        fc = nn.Linear(1024, 2)
        student.cuda()
        fc.cuda()
        train_dataset = torch.utils.data.TensorDataset(train_data, label)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=None,batch_size=128,shuffle=True) 
        student.train()
        fc.train()
        optimizer = torch.optim.Adam([{'params': student.encoder[3:4].parameters()},{'params': fc.parameters(),'lr': lr*50}], lr=lr) # {'params': student.encoder[2].parameters()},
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5,step_size=10)
        epoches = 1
        best_loss = 1000.0
        best_student = student
        for epoch in range(epoches):
            loss = self.do_epoch(epoch,lr_scheduler,student,train_loader,optimizer,W,fc,x_pos_train,x_pos_label)
            is_best = loss.get_avg() < best_loss
            print('loss.get_avg() ',loss.get_avg())
            if is_best:
                best_student = student
            best_loss = min(loss.get_avg(), best_loss)
            # Save checkpoint
            save_checkpoint(state={'epoch': epoch + 1,
                                'arch': 'Protonet',
                                'state_dict': student.state_dict(),
                                'best_prec1': best_loss,
                                'optimizer': optimizer.state_dict()},
                            is_best=is_best,
                            folder=model_path)
        return best_student

    def cross_entropy(self, logits, one_hot_targets, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        return - (one_hot_targets * logsoftmax).sum(1).mean()
    def do_epoch(self, epoch, scheduler, model,train_loader,optimizer,W,fc,x_pos_train,x_pos_label,disable_tqdm=False,device='cuda'):  # 可以看做基类训练，不需要划分支持集，查询集
        batch_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        W_mean = W.mean(0)
        x_pos_train = x_pos_train.cuda()
        tqdm_train_loader = warp_tqdm(train_loader, disable_tqdm) 
        for i, (input, target) in enumerate(tqdm_train_loader):
            input, target = input.to(self.device), target.to(device, non_blocking=True) # move to cuda
            feature,_ = model(input,True)
            pos_feature,p_ = model(x_pos_train,True)
            choose = torch.where(target>0.5,torch.ones(target.shape[0]).cuda(),torch.zeros(target.shape[0]).cuda())
            assert choose.shape[0] == target.shape[0]
            neg_w = feature[choose==0] # 68
            neg_mul = (1-target[choose==0]).view(-1,1)
            neg_mul = neg_mul.repeat(1,neg_w.shape[1])
            neg_w_wi = torch.mul(neg_w,neg_mul) # 

            pos_w = pos_feature.mean(0) # positive samples
            neg_w_wi = neg_w_wi.mean(0)
            target_neg = 1-target
            target_pos = target.view(-1,1)
            target_neg = target_neg.view(-1,1)
            target_one_hot = torch.cat([target_pos,target_neg],1)
            logits = fc(feature)
            loss_ce = self.cross_entropy(logits,target_one_hot)
            p_t = torch.cosine_similarity(W_mean[0],pos_w,dim=0)
            ls = []
            for k in range(neg_w.shape[0]):
                ls.append(torch.cosine_similarity(W_mean[0],neg_w[k],dim=0))
            T_t = 1
            fenmu = 0
            for t in ls:
                fenmu+= torch.exp(t/T_t)
            loss_clr = -torch.log(torch.exp(p_t/T_t)/fenmu)
            loss_w = 0.6*torch.cosine_similarity(W_mean[0],pos_w,dim=0) - 0.4*torch.cosine_similarity(W_mean[0],neg_w_wi,dim=0)
            loss = 0.7*loss_ce + 0.3*loss_w # thoese hyper parameter you can set by your self
            #loss = 0.5*loss_ce + 0.5*loss_clr # thoese hyper parameter you can set by your self
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 20== 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       loss=losses))
        return losses
            
    def run_task(self, task_dic, model,test_file,first):
        # Build the TIM classifier builder
        tim_builder = self.get_tim_builder(model,self.method,test_file,first) # choose the update methods
        # Extract support and query
        y_s = task_dic['y_s']  # n_task*?*?
        z_s, z_q = task_dic['z_s'], task_dic['z_q']
        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device) 
        # Perform normalizations required
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)

        # Initialize weights
        tim_builder.compute_lambda(support=support, query=query, y_s=y_s) # lambda
        print('tim_builder.loss_weights',tim_builder.loss_weights[0])
        print('self.number_task ',self.number_tasks)
        tim_builder.init_weights(support=support, y_s=y_s, query=query) # init W
        tim_builder.compute_FB_param(query)
        # Run adaptation
        tim_builder.run_adaptation(support=support, query=query, y_s=y_s) # update
        # Extract adaptation logs
        logs = tim_builder.get_logs()
        return logs
    def run_task_model_w(self, task_dic, model,test_file,first,loaders_dic,k_q,model_path):
        # Build the TIM classifier builder
        tim_builder = self.get_tim_builder(model,self.method,test_file,first) 

        # Extract support and query
        y_s = task_dic['y_s']  # n_task*?*?
        z_s, z_q = task_dic['z_s'], task_dic['z_q']

        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        # print('y_s ',y_s.shape)
        y_s = y_s.long().squeeze(2).to(self.device) #
        # Perform normalizations required
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)
        # Initialize weights
        tim_builder.compute_lambda(support=support, query=query, y_s=y_s) 
        print('tim_builder.loss_weights',tim_builder.loss_weights[0])
        print('self.number_task ',self.number_tasks)
        tim_builder.init_weights(support=support, y_s=y_s, query=query) 
        tim_builder.compute_FB_param(query)
        # Run adaptation
        model_new,nums,iters = tim_builder.run_adaptation_model_w(support=support, query=query, y_s=y_s,nums=1) # update
        while nums<iters:
            print('nums ',nums)
            extracted_features_dic = self.extract_features(model=model_new, model_path=model_path, model_tag=self.model_tag,
                                    used_set=test_file, fresh_start=1,loaders_dic=loaders_dic)
            tasks = self.generate_tasks(extracted_features_dic=extracted_features_dic,k_q=k_q)  # generate task
            y_s = tasks['y_s']  # n_task*?*?
            z_s, z_q = tasks['z_s'], tasks['z_q']
            # Transfer tensors to GPU if needed
            support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
            query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
            # print('y_s ',y_s.shape)
            y_s = y_s.long().squeeze(2).to(self.device) # 
            model_new,nums,iters = tim_builder.run_adaptation_model_w(support=support, query=query, y_s=y_s,nums=nums) 
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)
        # Extract adaptation logs
        logs = tim_builder.get_logs()
        return logs
    def get_tim_builder(self, model, method,test_file,first):
        # Initialize TIM classifier builder
        tim_info = {'model': model,'test_file': test_file,'first': first}
        if method == 'tim_adm':
            tim_builder = TIM_ADM(**tim_info)
        elif method == 'tim_gd':
            tim_builder = TIM_GD(**tim_info)
        elif method == 'baseline':
            tim_builder = TIM(**tim_info)
        else:
            raise ValueError("Method must be in ['tim_gd', 'tim_adm', 'baseline']")
        return tim_builder

    def get_loaders(self, hdf_eval,conf):
        # First, get loaders
        loaders_dic = {}
        gen_eval = Datagen_test(hdf_eval,conf)
        X_pos, X_neg,X_query = gen_eval.generate_eval()
        save_dict = {}
        save_dict['x_pos'] = X_pos
        save_dict.update({'x_query': X_query})
        self.number_tasks = X_query.shape[0] #
        X_pos = torch.tensor(X_pos)
        Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0])) # init as 0
        X_neg = torch.tensor(X_neg)
        Y_neg = torch.LongTensor(np.ones(X_neg.shape[0]))
        X_query = torch.tensor(X_query)
        Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

        num_batch_query = len(Y_query) // conf.eval.query_batch_size  # len // 8
        # print('num_batch_query ',num_batch_query) # 8821
        query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
        q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=conf.eval.query_batch_size,shuffle=False) # 按顺序来
        loaders_dic['query'] = q_loader

        neg_indices = torch.randperm(len(X_neg))[:conf.eval.samples_neg] 
        X_neg = X_neg[neg_indices]
        Y_neg = Y_neg[neg_indices]
        batch_size_neg = conf.eval.negative_set_batch_size # 16
        neg_dataset = torch.utils.data.TensorDataset(X_neg, Y_neg)
        negative_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None, batch_size=batch_size_neg)
        loaders_dic.update({'neg_loader': negative_loader})

        # batch_samplr_pos = EpisodicBatchSampler(Y_pos, num_batch_query + 1, 1, conf.train.n_shot*4)
        pos_dataset = torch.utils.data.TensorDataset(X_pos, Y_pos)
        #pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=batch_samplr_pos)
        pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset,batch_sampler=None, batch_size=25,shuffle=False)
        loaders_dic.update({'pos_loader': pos_loader})

        return loaders_dic,save_dict

    def extract_features(self, model, model_path, model_tag, used_set, fresh_start, loaders_dic,test_student=0):
        """
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            used_set : Set used between 'test' and 'val'
            n_ways : Number of ways for the task

        returns :
            extracted_features_dic : Dictionnary containing all extracted features and labels
        """

        # Load features from memory if previously saved ...
        save_dir = os.path.join(model_path, model_tag, used_set)
        filepath = os.path.join(save_dir, 'output.plk')
        if os.path.isfile(filepath) and (not fresh_start):
            extracted_features_dic = load_pickle(filepath)
            print(" ==> Features loaded from {}".format(filepath))
            return extracted_features_dic

        # ... otherwise just extract them
        else:
            print(" ==> Beginning feature extraction")
            os.makedirs(save_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            all_features = []
            all_labels = []
            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['query'], False)):
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs, True)
                all_features.append(outputs.cpu())
                all_labels.append(labels)
            all_features = torch.cat(all_features, 0)
            all_labels = torch.cat(all_labels, 0)
            extracted_features_dic = {'query_features': all_features,
                                      'query_labels': all_labels}
            all_features = []
            all_labels = []
            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader'], False)):
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs, True)
                all_features.append(outputs.cpu())
                all_labels.append(labels)
            all_features = torch.cat(all_features, 0)
            all_labels = torch.cat(all_labels, 0)
            extracted_features_dic.update({'pos_features': all_features,
                                      'pos_labels': all_labels})
            all_features = []
            all_labels = []
            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader'], False)):
                inputs = inputs.to(self.device)
                outputs, _ = model(inputs, True)
                all_features.append(outputs.cpu())
                all_labels.append(labels)
            all_features = torch.cat(all_features, 0)
            all_labels = torch.cat(all_labels, 0)
            extracted_features_dic.update({'neg_features': all_features,
                                      'neg_labels': all_labels})
        print(" ==> Saving features to {}".format(filepath))
        save_pickle(filepath, extracted_features_dic)
        return extracted_features_dic

    def get_task(self, extracted_features_dic,index,k_q):
        """
        inputs:
            extracted_features_dic : Dictionnary containing all extracted features and labels
            shot : Number of support shot per class
            n_ways : Number of ways for the task

        returns :
            task : Dictionnary : z_support : torch.tensor of shape [n_ways * shot, feature_dim]
                                 z_query : torch.tensor of shape [n_ways * query_shot, feature_dim]
                                 y_support : torch.tensor of shape [n_ways * shot]
                                 y_query : torch.tensor of shape [n_ways * query_shot]
        """
        query_features = extracted_features_dic['query_features']
        
        pos_features = extracted_features_dic['pos_features']
        pos_labels = extracted_features_dic['pos_labels']

        neg_features = extracted_features_dic['neg_features']
        neg_labels = extracted_features_dic['neg_labels']

        support_samples = []
        query_samples = []

        pos_indices = torch.randperm(len(pos_features))[:min(2000,len(pos_features))] # 5 shot
        X_pos = pos_features[pos_indices]
        num_neg = 50
        neg_indices = torch.randperm(len(neg_features))[:max(num_neg,len(pos_features))] # 5 shot
        X_neg = neg_features[neg_indices]
        support_samples.append(X_pos)
        support_samples.append(X_neg)
        
        query_size = query_features.shape[0]
        if index+k_q > query_size:
            X_query = query_features[-k_q:]
        else:
            X_query = query_features[index:index+k_q]
        query_samples.append(X_query)

        y_support_pos = torch.zeros(min(2000,len(pos_features)))
        y_support_neg = torch.ones(max(num_neg,len(pos_features)))
        y_support = torch.cat([y_support_pos,y_support_neg],0) # 
        z_support = torch.cat(support_samples, 0)
        z_query = torch.cat(query_samples, 0)

        task = {'z_s': z_support, 'y_s': y_support,
                'z_q': z_query}
        return task

    def generate_tasks(self, extracted_features_dic,k_q):
        """
        inputs:
            extracted_features_dic :
            shot : Number of support shot per class
            number_tasks : Number of tasks to generate

        returns :
            merged_task : { z_support : torch.tensor of shape [number_tasks, n_ways * shot, feature_dim]
                            z_query : torch.tensor of shape [number_tasks, n_ways * query_shot, feature_dim]
                            y_support : torch.tensor of shape [number_tasks, n_ways * shot]
                            y_query : torch.tensor of shape [number_tasks, n_ways * query_shot] }
        """
        print(f" ==> Generating {self.number_tasks//k_q} tasks ...")
        tasks_dics = []
        index = 0
        while index < self.number_tasks: 
            task_dic = self.get_task(extracted_features_dic,index,k_q)
            index += k_q
            tasks_dics.append(task_dic)
        # Now merging all tasks into 1 single dictionnary
        merged_tasks = {}
        n_tasks = len(tasks_dics)
        for key in tasks_dics[0].keys(): # z_s,y_s,z_q,y_q
            n_samples = tasks_dics[0][key].size(0)
            merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks, n_samples, -1)
        return merged_tasks


