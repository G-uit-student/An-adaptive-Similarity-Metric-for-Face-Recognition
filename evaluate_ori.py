import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.distance import correlation
import math
import torch
import torch.nn.functional as F

use_center=True
gallery_path = "gallery_emd_r18"
query_path = "query_emd_r18"

def getcentroid(vecs):
    # print(vecs.size())
    lengh=vecs.size(0)
    centroid=torch.mean(gallery_feature[index_arr], 0)
    
    distance = torch.cdist(centroid.view(1, -1), vecs, p=2)
    distance = torch.sum(distance)-distance
    centroid_w = torch.matmul(distance/torch.sum(distance),vecs)
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
    # print(matches.size(), q_pids.size(), g_pids.size(), indices.size())
    positive=[]
    negative=[]
    # indices_pos = torch.stack([torch.where(matches == True)[1], torch.where(matches == True)[0]], dim=1)
    # negative=similarity[indices_pos]
    # print(similarity.size())
    # positive=similarity[matches == True]
    # negative=similarity[matches == False]
    # for i in tqdm(range(matches.size(0))):
    #      for j in range(matches.size(1)):
    #         if matches[i][j]:
    #             positive.append(similarity[i][j])
    #         else:
    #             negative.append(similarity[i][j])

    # import matplotlib.pyplot as plt


    # bins = np.linspace(0, 1, 200)

    # plt.hist(positive, bins, density=True,
    #          alpha=0.5, label='1', facecolor='red')
    # plt.hist(negative, bins, density=True,
    #          alpha=0.5, label='0', facecolor='blue')
    # plt.savefig('combine.png')





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




files=os.listdir(gallery_path)
gallery_pid=[]
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
if use_center:
    centers = []
    for pid in gallery_pid:
        index_arr = np.where(gallery_pid == pid)[0]
        # centers.append(getcentroid(gallery_feature[index_arr]))
        centers.append(torch.mean(gallery_feature[index_arr], 0))
        # getcentroid(gallery_feature[index_arr])
    centers = torch.stack(centers, dim=0)


query_feature = F.normalize(query_feature, p=2, dim=1)
gallery_feature = F.normalize(gallery_feature, p=2, dim=1)


if use_center:
    centers = F.normalize(centers, p=2, dim=1)

    # similarity1 = torch.matmul(query_feature, gallery_feature.t())
    # similarity2 = torch.matmul(query_feature, centers.t())
    similarity1 = torch.cdist(query_feature, gallery_feature, p=2)
    # print(query_feature.size(), centers.size())
    similarity2 = torch.cdist(query_feature, centers, p=2)

    #norm 2 resnet100: -7.27890612 -7.40575791] [17.2733069
    #norm 2 resnet34: -7.53953658 -7.64612592] [18.17252602
    #norm 2 resnet50: -7.2438548  -7.37516197] [17.30471655

    #norm2: -7.83016114 - 7.66531965][18.31706869
    #norm with get:-7.83190859 - 7.65431508][18.30676881
    #cosine: 7.16154676 6.87027761][-4.02421888

    #re r50: -4.70326541 -9.81360128] [17.34664495  
    similarity = -2.95810494*similarity1 - 6.90354989*similarity2+11.60246772
    m = torch.nn.Sigmoid()
    all_cmc,mAP, _ = rank(m(similarity), query_pid, gallery_pid, topk=[1, 5, 10])
    print(all_cmc,mAP)
else:
    similarity = torch.matmul(query_feature, gallery_feature.t())
    all_cmc,mAP, _ = rank(similarity, query_pid, gallery_pid, topk=[1, 5, 10])
    print(all_cmc,mAP)
