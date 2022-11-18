from gen_factor import gen_factor
from evaluate_ori import rank, getcentroid, load_data
import argparse
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.distance import correlation
import math
import torch
import torch.nn.functional as F
random.seed(18520819)
np.random.seed(18520819)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='cosine', help='metric')
    parser.add_argument('--backbone', type=str,
                        default='r18', help='model.yaml path')
    parser.add_argument('--combine', action='store_true',
                        help='existing project/name ok, do not increment')
    return parser.parse_args()


def main(opt):
    m = torch.nn.Sigmoid()
    backbone_name = opt.backbone
    metric = opt.metric

    query_pid, gallery_pid, query_feature, gallery_feature = load_data(
        backbone_name)
    a, b, c = gen_factor(gallery_feature, gallery_pid, metric)
    query_feature = F.normalize(query_feature, p=2, dim=1)
    gallery_feature = F.normalize(gallery_feature, p=2, dim=1)
    #----------------------center--------------------
    centers = []
    for pid in gallery_pid:
        index_arr = np.where(gallery_pid == pid)[0]
        centers.append(torch.mean(gallery_feature[index_arr], 0))
    centers = torch.stack(centers, dim=0)
    centers = F.normalize(centers, p=2, dim=1)
    if metric == 'cosine':
        similarity1 = torch.matmul(query_feature, gallery_feature.t())
        similarity2 = torch.matmul(query_feature, centers.t())
        all_cmc, mAP, _ = rank(similarity1, query_pid,
                               gallery_pid, topk=[1, 5, 10])
        print("instance :", all_cmc[0])
    else:
        similarity1 = torch.cdist(query_feature, gallery_feature, p=2)
        similarity2 = torch.cdist(query_feature, centers, p=2)
        all_cmc, mAP, _ = rank(-similarity1, query_pid,
                               gallery_pid, topk=[1, 5, 10])
        print("instance :", all_cmc[0])

    similarity = a*similarity1 + b*similarity2 + c

    all_cmc, mAP, _ = rank(m(similarity), query_pid,
                           gallery_pid, topk=[1, 5, 10])
    print("adaptive :", all_cmc[0])

    pid_unique = torch.unique(gallery_pid)
    centers = []
    for pid in pid_unique:
        index_arr = np.where(gallery_pid == pid)[0]
        centers.append(torch.mean(gallery_feature[index_arr], 0))
    centers = torch.stack(centers, dim=0)
    centers = F.normalize(centers, p=2, dim=1)
    similarity = torch.matmul(query_feature, centers.t())

    all_cmc, mAP, _ = rank(similarity, query_pid, pid_unique, topk=[1, 5, 10])
    print("centroid :", all_cmc[0])


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
