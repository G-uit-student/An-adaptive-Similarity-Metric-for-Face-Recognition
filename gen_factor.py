import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.distance import correlation
import math
import torch
import torch.nn.functional as F
import random
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

random.seed(18520819)
np.random.seed(18520819)
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def getcentroid(vecs):
    # print(vecs.size())
    lengh=vecs.size(0)
    centroid=torch.mean(gallery_feature[index_arr], 0)
    distance = torch.cdist(centroid.view(1, -1), vecs, p=2)
    centroid_w = torch.matmul(torch.flip(distance/torch.sum(distance),dims=(1,)),vecs)
    # print(centroid_w.size())
    return centroid_w.view(-1)

files = os.listdir("gallery_emd_r18")
gallery_pid = []
gallery_feature = []
for file in files:
    np_arrays = np.load(os.path.join("gallery_emd_r18", file))
    for i in range(np_arrays.shape[0]):
        # print(np_arrays[i].reshape(-1).shape)
        gallery_feature.append(torch.from_numpy(np_arrays[i].reshape(-1)))
        gallery_pid.append(int(file.split(".")[0]))

gallery_pid = torch.from_numpy(np.array(gallery_pid))
gallery_feature = torch.stack(gallery_feature, dim=0)


# files = os.listdir("query_emd_r50")
# query_pid = []
# query_feature = []
# for file in files:
#     np_arrays = np.load(os.path.join("query_emd_r50", file))
#     for i in range(np_arrays.shape[0]):
#         # print(np_arrays[i].reshape(-1).shape)
#         query_feature.append(torch.from_numpy(np_arrays[i].reshape(-1)))
#         query_pid.append(int(file.split(".")[0]))
# query_pid = torch.from_numpy(np.array(query_pid))
# query_feature = torch.stack(query_feature, dim=0)


positives=[]
negatives=[]
for ii in tqdm(range(200)):
    pid = np.unique(gallery_pid)
    m=np.random.choice(pid, 1)[0]
    index_arr=np.where(gallery_pid == m)[0]
    random.shuffle(index_arr)
    # print(gallery_feature[index_arr[1:]].shape)
    centers = torch.mean(gallery_feature[index_arr[1:]], 0)
    point1 = F.normalize(gallery_feature[index_arr[0]].view(1, -1), p=2, dim=1)
    point2 = F.normalize(gallery_feature[index_arr[1]].view(1, -1), p=2, dim=1)
    centers = F.normalize(centers.view(1, -1), p=2, dim=1)

    cos_sim = float(torch.matmul(point1, point2.t())[0][0])
    cos_sim_c = float(torch.matmul(point1, centers.t())[0][0])

    norm_sim = float(torch.cdist(point1, point2, p=2)[0][0])
    norm_sim_c = float(torch.cdist(point1, centers, p=2)[0][0])
    
    positives.append([norm_sim, norm_sim_c])
    # positives.append([norm_sim,norm_sim_c])
for ii in tqdm(range(200)):
    pid = np.unique(gallery_pid)
    m,n = np.random.choice(pid, 2)
    index_arr_m=np.where(gallery_pid == m)[0]
    index_arr_n=np.where(gallery_pid == n)[0]

    random.shuffle(index_arr_m)
    random.shuffle(index_arr_n)
    centers = torch.mean(gallery_feature[index_arr_n[1:]], 0)

    point1 = F.normalize(gallery_feature[index_arr_m[0]].view(1, -1), p=2, dim=1)
    point2 = F.normalize(gallery_feature[index_arr_n[0]].view(1, -1), p=2, dim=1)
    centers = F.normalize(centers.view(1, -1), p=2, dim=1)


    cos_sim = float(torch.matmul(point1, point2.t())[0][0])
    cos_sim_c = float(torch.matmul(point1, centers.t())[0][0])
    norm_sim = float(torch.cdist(point1, point2, p=2)[0][0])
    norm_sim_c = float(torch.cdist(point1, centers, p=2)[0][0])

    negatives.append([norm_sim, norm_sim_c])
    # negatives.append([norm_sim,norm_sim_c])


print(len(positives),len(negatives))

positives=np.array(positives)
negatives=np.array(negatives)
# # print(positives.shape, negatives.shape)
Y=np.concatenate((np.ones(len(positives)), np.zeros(len(negatives))), axis=0)
X=np.concatenate((positives, negatives), axis=0)
# Y_t=np.concatenate((np.ones(len(old_positives)), np.zeros(len(old_negatives))), axis=0)
# X_t=np.concatenate((old_positives, old_negatives), axis=0)
logistic_regression = LogisticRegression(class_weight="balanced",solver='lbfgs')
model = logistic_regression.fit(X, Y)
print(model.score(X, Y), model.score(positives, np.ones(len(positives))),model.score(negatives, np.zeros(len(negatives))))
print(model.coef_[0], model.intercept_)

pos_=np.zeros((len(positives)))
neg_=np.zeros((len(negatives)))
for i in range(model.coef_[0].shape[0]):
  pos_+=positives[:,i]*model.coef_[0][i]
  neg_+=negatives[:,i]*model.coef_[0][i]
pos_+=model.intercept_[0]
neg_+=model.intercept_[0]
bins = np.linspace(-10,10, 100)

plt.hist(pos_, bins, density=True,
         alpha=0.5, label='1', facecolor='blue')
plt.hist(neg_, bins, density=True,
         alpha=0.5, label='0', facecolor='red')
# plt.xlabel('Euclidean distance')
# plt.ylabel('Y-Axis')

# plt.savefig('combine4/combineresnet101.png')
# from plot import LogisticRegression
# lr = LogisticRegression(server=True)
# lr.exec(positives,negatives)
