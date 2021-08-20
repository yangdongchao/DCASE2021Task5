import torch
import time
import torch.nn as nn
import numpy as np
from src.util import warp_tqdm, get_metric, AverageMeter,prototypical_loss
from sklearn import manifold, datasets
import h5py
# setting
print_freq = 1000
meta_val_way = 10
meta_val_shot = 5
meta_val_metric = 'cosine'  # ('euclidean', 'cosine', 'l1', l2')
meta_val_iter = 500
meta_val_query = 15
alpha = - 1.0
label_smoothing = 0.
class Trainer:
    def __init__(self, device,num_class,train_loader,val_loader, conf):
        self.train_loader,self.val_loader = train_loader,val_loader
        self.device = device
        self.num_classes = num_class # 
        self.alpha = -1.0
        self.label_smoothing = 0.1
        self.meta_val_metric = 'cosine'

    def cross_entropy(self, logits, one_hot_targets, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        return - (one_hot_targets * logsoftmax).sum(1).mean()

    def do_epoch(self, epoch, scheduler, disable_tqdm, model,
                 alpha, optimizer):  
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()
        steps_per_epoch = len(self.train_loader) 
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm) 
        for i, (input, target) in enumerate(tqdm_train_loader):

            input, target = input.to(self.device), target.to(self.device, non_blocking=True) 
            smoothed_targets = self.smooth_one_hot(target,self.label_smoothing) 
            # assert (smoothed_targets.argmax(1) == target).float().mean() == 1.0
            # Forward pass
            if self.alpha > 0:  # Mixup augmentation
                # generate mixed sample and targets
                lam = np.random.beta(self.alpha, self.alpha)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                mixed_input = lam * input + (1 - lam) * input[rand_index]

                output = model(mixed_input)
                loss = self.cross_entropy(output, target_a) * lam + self.cross_entropy(output, target_b) * (1. - lam)
            else:
                output = model(input)
                loss = self.cross_entropy(output, smoothed_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = (output.argmax(1) == target).float().mean()
            top1.update(prec1.item(), input.size(0))
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            # Measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    def do_epoch_meta_learning(self, epoch, scheduler, disable_tqdm, model,
                 alpha, optimizer): 
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()
        steps_per_epoch = len(self.train_loader) 
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm) 
        for i, (input, target) in enumerate(tqdm_train_loader):

            input, target = input.to(self.device), target.to(self.device, non_blocking=True) 
            # print('target ',target.shape)
            smoothed_targets = self.smooth_one_hot(target,self.label_smoothing) 
            # assert (smoothed_targets.argmax(1) == target).float().mean() == 1.0
            # Forward pass
            if self.alpha > 0:  # Mixup augmentation
                # generate mixed sample and targets
                lam = np.random.beta(self.alpha, self.alpha)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                mixed_input = lam * input + (1 - lam) * input[rand_index]

                output = model(mixed_input)
                loss = self.cross_entropy(output, target_a) * lam + self.cross_entropy(output, target_b) * (1. - lam)
            else:
                output, feature = model(input,feature = True)
                loss_val, acc_val = prototypical_loss(feature,target,5)
                loss = loss_val

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = acc_val
            top1.update(prec1.item(), input.size(0))
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            # Measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    def smooth_one_hot(self, targets, label_smoothing):
        assert 0 <= label_smoothing < 1
        with torch.no_grad():
            new_targets = torch.empty(size=(targets.size(0), self.num_classes), device=self.device)
            new_targets.fill_(label_smoothing / (self.num_classes-1))
            new_targets.scatter_(1, targets.unsqueeze(1), 1. - label_smoothing)
        return new_targets

    def get_feature_by_y(self,feature_x,target_y,idx):
        new_feature_x = []
        for i  in range(feature_x.shape[0]):
            if target_y[i]==idx:
                new_feature_x.append(feature_x[i,:])
        new_feature_x = np.array(new_feature_x)
        new_feature_x = new_feature_x.mean(0)
        return new_feature_x
    def meta_val(self,epoch, model, disable_tqdm):
        top1 = AverageMeter()
        model.eval() 

        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.val_loader, disable_tqdm)
            for i, (inputs, target) in enumerate(tqdm_test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
                classes = torch.unique(target)
                n_classes = len(classes)

                def supp_idxs(c):
                    return target.eq(c).nonzero()[:meta_val_shot].squeeze(1) 
                output = model(inputs, feature=True)[0].cuda(0)
                n_query = meta_val_query 
                support_idxs = list(map(supp_idxs,classes)) 
                query_idxs = torch.stack(list(map(lambda c:target.eq(c).nonzero()[meta_val_shot:],classes))).view(-1) 
                train_out = torch.cat([output[idx_list] for idx_list in support_idxs]) #
                train_label = torch.cat([target[idx_list] for idx_list in support_idxs])
                test_out = output[query_idxs]
                test_label = target[query_idxs]
                train_out = train_out.reshape(meta_val_way, meta_val_shot, -1).mean(1) # 
                train_label = train_label[::meta_val_shot] # 0,1,2,3...
                prediction = self.metric_prediction(train_out, test_out, train_label,meta_val_metric ='cosine')
                acc = (prediction == test_label).float().mean()
                top1.update(acc.item(),prediction.shape[0])
                if not disable_tqdm:
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg * 100))
        return top1.avg
    def save_plot_data(self,feature_x,target_y,model):
        feature_x = np.array(feature_x)
        target_y = np.array(target_y)
        new_feature_x = []
        new_feature_y = []
        for i in range(19):
            new_feature_x.append(self.get_feature_by_y(feature_x, target_y, i))
            new_feature_y.append(i)
        for i in range(model.state_dict()['fc.weight'].shape[0]):
            new_feature_x.append(model.state_dict()['fc.weight'][i,:].detach().cpu().numpy())
            new_feature_y.append(i+19)
        new_feature_x = np.array(new_feature_x)
        new_feature_y = np.array(new_feature_y)
        print('new_feature_x ',new_feature_x.shape)
        print('new_feature_y ',new_feature_y.shape)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(new_feature_x)
        hf = h5py.File('/home/ydc/DACSE2021/sed-tim-base/check_point/plot/visual_transductive.h5', 'w')
        X_shape = X_tsne.shape[1]
        hf.create_dataset(
                name='feature', 
                shape=(new_feature_y.shape[0], X_shape), 
                dtype=np.float32)
        hf.create_dataset(
                name='target', 
                shape=(new_feature_y.shape[0],), 
                dtype=np.float32)
        for n,u in enumerate(X_tsne):
            hf['feature'][n] = u
        for n,u in enumerate(new_feature_y):
            hf['target'][n] = u
        hf.close()
    def met_plot(self,epoch, model, disable_tqdm):
        top1 = AverageMeter()
        model.eval() 
        feature_x = []
        target_y = []
        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.train_loader, disable_tqdm)
            for i, (inputs, target) in enumerate(tqdm_test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
                classes = torch.unique(target) 
                n_classes = len(classes)

                def supp_idxs(c):
                    return target.eq(c).nonzero()[:meta_val_shot].squeeze(1) 
                output = model(inputs, feature=True)[0].cuda(0)
                for o in output:
                    feature_x.append(o.detach().cpu().numpy())
                for t in target:
                    target_y.append(t.detach().cpu().numpy())
        
        self.save_plot_data(feature_x,target_y,model)

    def metric_prediction(self, support, query, train_label, meta_val_metric): # meta_val_metric--> consin?
        support = support.view(support.shape[0], -1) # n_way* 
        # print('support ',support.shape)
        query = query.view(query.shape[0], -1) # n_query * ?
        # print('query ',query.shape)
        distance = get_metric(meta_val_metric)(support, query) #
        # print('distance ',distance.shape)
        predict = torch.argmin(distance, dim=1) 
        # print('predict ',predict)
        predict = torch.take(train_label, predict) 
        # print('predict2 ',predict)
        return predict