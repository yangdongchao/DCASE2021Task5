import torch
import numpy as np
import shutil
from tqdm import tqdm
import logging
import os
import pickle
import torch.nn.functional as F
from sklearn import manifold, datasets
import h5py
def prototypical_loss(input, target,n_support):
    '''
    Adopted from https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
    Compute the prototypes by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      bprototypes, for each one of the current classes
    '''
    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    classes = torch.unique(target_cpu) 
    n_classes = len(classes)
    p = n_classes * n_support
    n_query = target.eq(classes[0].item()).sum().item() - n_support 
    support_idxs = list(map(supp_idxs,classes)) 
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs]) 
    query_idxs = torch.stack(list(map(lambda c:target.eq(c).nonzero()[n_support:],classes))).view(-1) 
    query_samples = input.cpu()[query_idxs] 
    dists = euclidean_dist(query_samples, prototypes) 
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1) 
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long() 
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() 
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)

    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)
def get_one_hot(y_s):  # 
    num_classes = torch.unique(y_s).size(0)
    eye = torch.eye(num_classes).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot


def get_logs_path(model_path, method, shot):
    exp_path = '_'.join(model_path.split('/')[1:])
    file_path = os.path.join('tmp', exp_path, method)
    os.makedirs(file_path, exist_ok=True)
    return os.path.join(file_path, f'{shot}.txt')


def get_features(model, samples):
    features, _ = model(samples, True)
    features = F.normalize(features.view(features.size(0), -1), dim=1)
    return features


def get_loss(logits_s, logits_q, labels_s, lambdaa):
    Q = logits_q.softmax(2)
    y_s_one_hot = get_one_hot(labels_s) #
    ce_sup = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1)  # Taking the mean over samples within a task, and summing over all samples
    # ce_sup
    ent_q = get_entropy(Q)
    cond_ent_q = get_cond_entropy(Q)
    loss = - (ent_q - cond_ent_q) + lambdaa * ce_sup
    return loss


def get_mi(probs):
    q_cond_ent = get_cond_entropy(probs)
    q_ent = get_entropy(probs)
    return q_ent - q_cond_ent


def get_entropy(probs): # H(Y_q)
    q_ent = - (probs.mean(1) * torch.log(probs.mean(1) + 1e-12)).sum(1, keepdim=True) # H(Y_q)
    return q_ent


def get_cond_entropy(probs): # H(Y_q | X_q)
    q_cond_ent = - (probs * torch.log(probs + 1e-12)).sum(2).mean(1, keepdim=True) # H(Y_q | X_q)
    return q_cond_ent


def get_metric(metric_type): 
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): # 
        self.val = val # 
        self.sum += val * n # 
        self.count += n 
        self.avg = self.sum / self.count 
    def get_avg(self):
        return self.avg

def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def warp_tqdm(data_loader, disable_tqdm):
    if disable_tqdm: # 
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data): #
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file): #
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'): # check_point
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')


def load_checkpoint(model, model_path, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path))
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(model_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    state_dict = checkpoint['state_dict']
    names = []
    for k, v in state_dict.items():
        names.append(k)
    model.load_state_dict(state_dict)


def compute_confidence_interval(data, axis=0): 
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    return m, pm
def save_plot_data(feature_x,target_y,save_file_name):
        feature_x = np.array(feature_x)
        target_y = np.array(target_y)
        print('feature_x ',feature_x.shape)
        print('target_y ',target_y.shape)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(feature_x)
        path = '/home/ydc/DACSE2021/sed-tim-base/check_point/plot/'+save_file_name
        hf = h5py.File(path, 'w')
        X_shape = X_tsne.shape[1]
        hf.create_dataset(
                name='feature', 
                shape=(target_y.shape[0], X_shape), 
                dtype=np.float32)
        hf.create_dataset(
                name='target', 
                shape=(target_y.shape[0],), 
                dtype=np.float32)
        for n,u in enumerate(X_tsne):
            #hf['feature'].resize((n + 1, X_shape))
            hf['feature'][n] = u
        for n,u in enumerate(target_y):
            hf['target'][n] = u
        hf.close()