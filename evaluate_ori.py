import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.distance import correlation
import math
import torch
import torch.nn.functional as F




def getcentroid(vecs):
    # print(vecs.size())
    lengh = vecs.size(0)
    centroid = torch.mean(gallery_feature[index_arr], 0)

    distance = torch.cdist(centroid.view(1, -1), vecs, p=2)
    distance = torch.sum(distance)-distance
    centroid_w = torch.matmul(distance/torch.sum(distance), vecs)
    # print(centroid_w.size())
    return centroid_w.view(-1)


def rank(similarity, q_pids, g_pids, topk=[1], get_mAP=True):
    # print(torch.min(similarity),torch.max(similarity))
    topk = torch.tensor(topk)
    max_rank = max(topk)
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100
    return all_cmc, mAP, indices


def load_data(backbone_name):
    gallery_path = "gallery_emd_{}".format(backbone_name)

    query_path = "query_emd_{}".format(backbone_name)

    files = os.listdir(gallery_path)

    gallery_pid = []
    gallery_feature = []
    for file in files:
        np_arrays = np.load(os.path.join(gallery_path, file))
        for i in range(np_arrays.shape[0]):
            # print(np_arrays[i].reshape(-1).shape)
            gallery_feature.append(torch.from_numpy(np_arrays[i].reshape(-1)))
            gallery_pid.append(int(file.split(".")[0]))

    gallery_pid = torch.from_numpy(np.array(gallery_pid))
    gallery_feature = torch.stack(gallery_feature, dim=0)

    files = os.listdir(query_path)
    query_pid = []
    query_feature = []
    for file in files:
        np_arrays = np.load(os.path.join(query_path, file))
        for i in range(np_arrays.shape[0]):
            # print(np_arrays[i].reshape(-1).shape)
            query_feature.append(torch.from_numpy(np_arrays[i].reshape(-1)))
            query_pid.append(int(file.split(".")[0]))
    query_pid = torch.from_numpy(np.array(query_pid))
    query_feature = torch.stack(query_feature, dim=0)
    return query_pid, gallery_pid, query_feature, gallery_feature
