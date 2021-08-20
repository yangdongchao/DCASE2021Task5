import yaml
import argparse
import pandas as pd
import random
import csv
import os
import h5py
import pandas as pd
from glob import glob
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from tqdm import tqdm
from collections import Counter
from src.datasets.batch_sampler import EpisodicBatchSampler
from src.trainer import Trainer
from src.util import warp_tqdm, save_checkpoint
from src.models import __dict__
from src.datasets.Feature_extract import feature_transform
from src.datasets.Datagenerator import Datagen
from src.eval import Evaluator
import time
def get_model(arch, num_classes):
    if arch == 'resnet10' or arch == 'resnet18':
        return __dict__[arch](num_classes=num_classes)
    else:
        return __dict__[arch]()

def train_protonet(model,train_loader,valid_loader,conf):
    arch = 'Protonet'
    alpha = 0.0  
    disable_tqdm = False 
    ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/check_point'
    pretrain = False
    resume = False
    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optim = torch.optim.Adam(model.parameters(), lr=conf.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=conf.train.scheduler_gamma,
                                                   step_size=conf.train.scheduler_step_size)
    num_epochs = conf.train.epochs

    if pretrain: 
        pretrain = os.path.join(pretrain, 'checkpoint.pth.tar')
        if os.path.isfile(pretrain):
            print("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            print('[Warning]: Did not find pretrained model {}'.format(pretrain))

    if resume:
        resume_path = ckpt_path + '/checkpoint.pth.tar'
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print('[Warning]: Did not find checkpoint {}'.format(resume_path))
    else:
        start_epoch = 0
        best_prec1 = -1

    #cudnn.benchmark = True
    model.to(device) # cuda
    trainer = Trainer(device=device,num_class=19, train_loader=train_loader,val_loader=valid_loader,conf=conf)
    time_start=time.time()
    for epoch in range(num_epochs):
        trainer.do_epoch(epoch=epoch,scheduler=lr_scheduler,disable_tqdm=disable_tqdm,model=model,alpha=alpha,optimizer=optim)
        # Evaluation on validation set
        prec1 = trainer.meta_val(epoch=epoch,model=model, disable_tqdm=disable_tqdm)
        print('Meta Val {}: {}'.format(epoch, prec1))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if not disable_tqdm:
            print('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # Save checkpoint
        save_checkpoint(state={'epoch': epoch + 1,
                               'arch': arch,
                               'state_dict': model.state_dict(),
                               'best_prec1': best_prec1,
                               'optimizer': optim.state_dict()},
                        is_best=is_best,
                        folder=ckpt_path)
        if lr_scheduler is not None:
            lr_scheduler.step()
    time_end=time.time()
    print('totally cost',time_end-time_start)
    print('model_paramiter...............')
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
        # if epoch == num_epochs-1:
        #     trainer.met_plot(epoch, model, disable_tqdm)
def eval_only(model,train_loader,valid_loader,conf):  # what this function means?
    arch = 'Protonet'
    alpha = 0.0  # mixpu
    disable_tqdm = False # 
    ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/check_point'
    pretrain = True
    resume = False
    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optim = torch.optim.Adam(model.parameters(), lr=conf.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=conf.train.scheduler_gamma,
                                                   step_size=conf.train.scheduler_step_size)
    num_epochs = conf.train.epochs

    if pretrain:
        pretrain = os.path.join(ckpt_path, 'checkpoint.pth.tar')
        if os.path.isfile(pretrain):
            print("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            print('[Warning]: Did not find pretrained model {}'.format(pretrain))

    if resume:
        resume_path = ckpt_path + '/checkpoint.pth.tar'
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print('[Warning]: Did not find checkpoint {}'.format(resume_path))
    else:
        start_epoch = 0
        best_prec1 = -1

    #cudnn.benchmark = True
    model.to(device) # 
    trainer = Trainer(device=device,num_class=19, train_loader=train_loader,val_loader=valid_loader,conf=conf)
    for epoch in range(num_epochs):
        #trainer.do_epoch(epoch=epoch,scheduler=lr_scheduler,disable_tqdm=disable_tqdm,model=model,alpha=alpha,optimizer=optim)
        # Evaluation on validation set
        prec1 = trainer.met_plot(epoch=epoch,model=model, disable_tqdm=disable_tqdm)
        print('Meta Val {}: {}'.format(epoch, prec1))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if not disable_tqdm:
            print('Best Acc {:.2f}'.format(best_prec1 * 100.))

@hydra.main(config_name="config")
def main(conf : DictConfig):
    seed = 2021
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
    if not os.path.isdir(conf.path.feat_path):
        os.makedirs(conf.path.feat_path)

    if not os.path.isdir(conf.path.feat_train):
        os.makedirs(conf.path.feat_train)

    if not os.path.isdir(conf.path.feat_eval):
        os.makedirs(conf.path.feat_eval)

    if conf.set.features:

        print(" --Feature Extraction Stage--")
        Num_extract_train,data_shape = feature_transform(conf=conf,mode="train") # train data
        print("Shape of dataset is {}".format(data_shape))
        print("Total training samples is {}".format(Num_extract_train))

        Num_extract_eval = feature_transform(conf=conf,mode='eval')
        print("Total number of samples used for evaluation: {}".format(Num_extract_eval)) # validate data
        print(" --Feature Extraction Complete--")

        Num_extract_test = feature_transform(conf=conf,mode='test')
        print("Total number of samples used for evaluation: {}".format(Num_extract_test)) # test data
        print(" --Feature Extraction Complete--")



    if conf.set.train: # train
        meta_learning = False # wether use meta learing ways to train
        if meta_learning:
            gen_train = Datagen(conf) 
            X_train,Y_train,X_val,Y_val = gen_train.generate_train() # 
            X_tr = torch.tensor(X_train) 
            Y_tr = torch.LongTensor(Y_train)
            X_val = torch.tensor(X_val)
            Y_val = torch.LongTensor(Y_val)

            samples_per_cls =  conf.train.n_shot * 2 

            batch_size_tr = samples_per_cls * conf.train.k_way # the batch size of training 
            batch_size_vd = batch_size_tr # 

            num_batches_tr = len(Y_train)//batch_size_tr # num of batch
            num_batches_vd = len(Y_val)//batch_size_vd


            samplr_train = EpisodicBatchSampler(Y_train,num_batches_tr,conf.train.k_way,samples_per_cls) # batch_size_tr = samples_per_cls * conf.train.k_way
            samplr_valid = EpisodicBatchSampler(Y_val,num_batches_vd,conf.train.k_way,samples_per_cls)

            train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr) # 利用torch 的 dataset,整合X,Y
            valid_dataset = torch.utils.data.TensorDataset(X_val,Y_val)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_sampler=samplr_train,shuffle=False)
            # batch_sampler 批量采样，默认设置为None。但每次返回的是一批数据的索引,每次输入网络的数据是随机采样模式，这样能使数据更具有独立性质
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_sampler=samplr_valid,shuffle=False)

        else:
            gen_train = Datagen(conf) 
            X_train,Y_train,X_val,Y_val = gen_train.generate_train() 
            X_tr = torch.tensor(X_train) 
            Y_tr = torch.LongTensor(Y_train)
            X_val = torch.tensor(X_val)
            Y_val = torch.LongTensor(Y_val)
            # print('X_tr ',X_tr.shape)
            # print('X_val ',X_val.shape)
            samples_per_cls =  conf.train.num_query+conf.train.n_shot 

            batch_size_tr = 64 # the batch size of training 
            batch_size_vd = (conf.train.num_query+conf.train.n_shot) * conf.train.k_way # 

            #num_batches_tr = len(Y_train)//batch_size_tr # num of batch
            num_batches_vd = len(Y_val)//batch_size_vd
            #samplr_train = EpisodicBatchSampler(Y_train,num_batches_tr,conf.train.k_way,samples_per_cls) # batch_size_tr = samples_per_cls * conf.train.k_way
            samplr_valid = EpisodicBatchSampler(Y_val,num_batches_vd,conf.train.k_way,samples_per_cls)

            train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr) # 利用torch 的 dataset,整合X,Y
            valid_dataset = torch.utils.data.TensorDataset(X_val,Y_val)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size_tr,num_workers=0,pin_memory=True,shuffle=True)
            # batch_sampler 批量采样，默认设置为None。但每次返回的是一批数据的索引,每次输入网络的数据是随机采样模式，这样能使数据更具有独立性质
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_sampler=samplr_valid,num_workers=0,pin_memory=True,shuffle=False)

        #model = get_model('Protonet',19)
        model = get_model('Protonet',19)
        #model = get_model('resnet10',19)
        #train_protonet(model,train_loader,valid_loader,conf)
        train_protonet(model,train_loader,valid_loader,conf)

    if conf.set.eval: # eval

        device = 'cuda'

        # init_seed()
        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])
        all_feat_files = sorted([file for file in glob(os.path.join(conf.path.feat_eval,'*.h5'))])
        evaluator = Evaluator(device=device)
        model = get_model('Protonet',19).cuda()
        student_model = get_model('Protonet',19).cuda()
        #ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/check_point'
        ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/pre_best'
        hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel) # 0.05*22050//256 == 4
        k_q = 128
        iter_num = 1 # change wether use ML framework
        for feat_file in all_feat_files:
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5','wav')
            print("Processing audio file : {}".format(audio_name))

            hdf_eval = h5py.File(feat_file,'r')
            strt_index_query =  hdf_eval['start_index_query'][:][0]
            
            result, num= evaluator.run_full_evaluation(test_file=audio_name[:-4],model=model,student_model=student_model,
                                                        model_path=ckpt_path,hdf_eval=hdf_eval,conf=conf,k_q=k_q,iter_num=iter_num) # only update W
            # result, num= evaluator.run_full_evaluation_model_w(test_file=audio_name[:-4],model=model,student_model=student_model,
            #                                             model_path=ckpt_path,hdf_eval=hdf_eval,conf=conf,k_q=k_q,iter_num=iter_num) # updata feature extractor and W
            # num indicate the length of return sequence
            predict = result[0]
            if predict.shape[0]>num:
                n_ = predict.shape[0]//k_q
                print('n_ ',n_)
                prob_final = predict[:(n_-1)*k_q]
                n_last = num - prob_final.shape[0]
                print('n_last ',n_last)
                prob_final = np.concatenate((prob_final,predict[-n_last:]))
                print('prob_final ',prob_final.shape)
            else:
                prob_final = predict
            
            assert num == prob_final.shape[0]
            krn = np.array([1, -1])
            prob_thresh = np.where(prob_final > 0.5, 1, 0) # 70572
            prob_pos_final = prob_final * prob_thresh
            changes = np.convolve(krn, prob_thresh) # 70573
            # print('changes ',changes.shape)
            onset_frames = np.where(changes == 1)[0]
            print('onset_frames ',onset_frames.shape)
            offset_frames = np.where(changes == -1)[0]

            str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr # 
            print('str_time_query ',str_time_query) # 322.5
            onset = (onset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
            onset = onset + str_time_query
            # print('onset ',onset)
            offset = (offset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
            offset = offset + str_time_query
            # print('offset ',offset)
            assert len(onset) == len(offset)
            
            name = np.repeat(audio_name,len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        csv_path = os.path.join(conf.path.work_path,'Eval_out_tim.csv')
        df_out.to_csv(csv_path,index=False)

    if conf.set.test: # It only be used when test the final dataset of DCASE2021 task5

        device = 'cuda'

        # init_seed()
        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])
        all_feat_files = sorted([file for file in glob(os.path.join(conf.path.feat_test,'*.h5'))])
        evaluator = Evaluator(device=device)
        model = get_model('Protonet',19).cuda()
        student_model = get_model('Protonet',19).cuda()
        ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/check_point'
        #ckpt_path = '/home/ydc/DACSE2021/task5/best2'
        #ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/pre_best'
        hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel) # 0.05*22050//256 == 4
        k_q = 128
        iter_num = 1
        for feat_file in all_feat_files:
            print('file name ',feat_file)
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5','wav')
            print("Processing audio file : {}".format(audio_name))
            hdf_eval = h5py.File(feat_file,'r')
            strt_index_query =  hdf_eval['start_index_query'][:][0]
            result, num= evaluator.run_full_evaluation(test_file=audio_name[:-4],model=model,student_model=student_model,
                                                        model_path=ckpt_path,hdf_eval=hdf_eval,conf=conf,k_q=k_q,iter_num=iter_num) # only update W
            # result, num= evaluator.run_full_evaluation_model_w(test_file=audio_name[:-4],model=model,student_model=student_model,
            #                                             model_path=ckpt_path,hdf_eval=hdf_eval,conf=conf,k_q=k_q,iter_num=iter_num) # updata model and W
            # num 返回query 的长度
            predict = result[0]
            if predict.shape[0]>num:
                n_ = predict.shape[0]//k_q
                print('n_ ',n_)
                prob_final = predict[:(n_-1)*k_q]
                n_last = num - prob_final.shape[0]
                print('n_last ',n_last)
                prob_final = np.concatenate((prob_final,predict[-n_last:]))
                print('prob_final ',prob_final.shape)
            else:
                prob_final = predict
            
            assert num == prob_final.shape[0]
            krn = np.array([1, -1])
            prob_thresh = np.where(prob_final > 0.5, 1, 0) # 70572
            prob_pos_final = prob_final * prob_thresh
            changes = np.convolve(krn, prob_thresh) # 70573
            onset_frames = np.where(changes == 1)[0]
            print('onset_frames ',onset_frames.shape)
            offset_frames = np.where(changes == -1)[0]
            str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr # 转时间？
            print('str_time_query ',str_time_query) # 322.5
            onset = (onset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
            onset = onset + str_time_query
            offset = (offset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
            offset = offset + str_time_query
            assert len(onset) == len(offset)
            name = np.repeat(audio_name,len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        csv_path = os.path.join(conf.path.work_path,'Eval_out_tim_test.csv')
        df_out.to_csv(csv_path,index=False)






if __name__ == '__main__':
    main()


