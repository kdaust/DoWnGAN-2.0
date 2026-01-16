#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:31:16 2023

@author: kiridaust
"""

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

def bourgain_embedding(data, p, m, distmat):
    """
    bourgain embedding main function.
    Args:
        data (ndarray): Input data for embedding. Shape must be nxm, 
                        where n is the number of data points, m is the data dimension.
        p, m (float): bourgain embedding hyperparameters.
        distmat (ndarray): Distance matrix for data, shape must be nxn.
    
    Returns:
        ans (ndarray): results for bourgain embedding, shape must be nxk, where k is the
            latent space dimension.

    """
    assert(p>0 and p<1)
    assert(isinstance(m, int))
    n = data.shape[0]
    K = np.ceil(np.log(n)/np.log(1/p))
    S={}
    for j in range(int(K)):
        for i in range(m):
            S[str(i)+str('_')+str(j)]=[]
            prob = np.power(p, j+1)
            rand_num = np.random.rand(n)
            good = rand_num<prob
            good = np.argwhere(good==True).reshape((-1))
            S[str(i)+str('_')+str(j)].append(good)


    ans = np.zeros((n, int(K)*m))

    for (c, x) in enumerate(data):
        fx = np.zeros((m, int(K)))
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                fx[i, j] = mindist(c, S[str(i)+str('_')+str(j)], distmat)

        fx = fx.reshape(-1)
        ans[c, :] = fx

    ans = ans - np.mean(ans, axis=0)
    dists = np.linalg.norm(ans, ord='fro')/np.sqrt(ans.shape[0])
    ans = ans/dists * np.sqrt(ans.shape[1])
    return ans

def mindist(x_id, idxset, distmat):
    """
    helper function to find the minimal distance in a given point set
    Args:
        x_id (int): id for reference point
        idxset (list): ids for the point set to test
        distmat (ndarray): distance matrix for all points
    
    Returns:
        mindist (float): minimal distance
    """
    d = distmat[x_id, idxset[0]]
    if d.shape[0] == 0:
        return 0
    else:
        return torch.min(d)
    
def pairwise_dist(data, dist, p=None):
    if not isinstance(dist, str):
        raise ValueError('dist must be str')
    
    if dist == 'l2':
        return squareform(pdist(data, metric='euclidean'))
    elif dist == 'lp':
        return squareform(pdist(data, metric='minkowski', p=p))


####Main sampler
class BourgainSampler(object):
    def __init__(self, data, path=None, dist='l2'):
        if path is not None:
            self.load(path)
            return

        self.name = "bourgain"
        p = 0.7
        m = 20
        self.eps = 0.01
        self.origin_data = data
        batch_size = data.size(0)
        reshaped_tensor = data.view(batch_size, 1, data.size(1), data.size(2))
        expanded_tensor = reshaped_tensor.expand(batch_size, batch_size, data.size(1), data.size(2))
        distmat = torch.pow(expanded_tensor - expanded_tensor.transpose(0, 1), 2).mean(dim=(2, 3))
        #distmat = torch.cdist(data,data)
        embed_dat = bourgain_embedding(data, p=p, m=m, distmat=distmat)
        self.embedded_data = torch.from_numpy(embed_dat).to("cuda:0")
        #the data stored in Sampler are numpy array
        self.scale = float(self.get_scale(embed_dat, distmat))
        print('scale factor:', self.scale)

        
    def sampling(self, n):   #bourgain sampling
        num_data = self.embedded_data.shape[0]
        sampled_idx = torch.randint(0,num_data,[n]) ##np.random.choice(num_data, n)
        sampled_data = self.embedded_data[sampled_idx, :]
        noise = torch.normal(0, self.eps, sampled_data.shape, device="cuda:0") ##np.random.normal(scale=self.eps, size=sampled_data.shape)
        sampled_data = sampled_data + noise
        sampled_data = sampled_data[:,0:256]
        #sampled_data = torch.from_numpy(sampled_data).to("cuda:0").float()
        sampled_data = torch.reshape(sampled_data,(n,1,16,16)).float() ##shouldn't hard code       
        #return torch.Tensor
        return sampled_data


    def get_scale(self, embedded_data, distmat):
        l2 = pairwise_dist(embedded_data, 'l2')
        for i in range(l2.shape[0]):
            l2[i, i] = 1
        l2 = torch.from_numpy(l2)
        div1 = torch.sqrt(torch.divide(distmat, l2))
        return torch.max(div1)

    def save(self, path):
        np.savez(path, eps=self.eps, embed=self.embedded_data, scale=self.scale, origin_data=self.origin_data)

    def load(self, path):
        ff = np.load(path)
        self.embedded_data = ff['embed']
        self.eps = ff['eps']
        self.scale = ff['scale']
        self.origin_data = ff['origin_data']
        self.name = "bourgain"