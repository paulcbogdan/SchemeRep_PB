# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:16:22 2020
modified 2021/04/15
@author: ld178
"""

# %%
import os

#import urllib3
import numpy as np
from PIL import Image
from cv2 import resize

from tqdm import tqdm
from scipy.stats import spearmanr
from scipy import io

import pandas as pd
#import wget
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

#import imageio
# %%
import sys
sys.path.insert(0,'D:\Research_local\SchemRep\RSAmodel\cornet_models\CORnet-master')
from torchvision.models import vgg16
# %%
model = vgg16(pretrained=True)
# %%
Ts = T.Compose([ T.ToTensor(),
                 T.Resize((224,224)),
                 #T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
               ])


for p in model.parameters():
    p.requires_grad = False

model.eval()

# %%
tblStim=pd.read_csv(r"/SchemRep_tasks/PTBtasks/fullStimList.csv")
tblStim.head()

filelist=tblStim['ObjectFile'].to_list()
padImagePath=r"C:\PycharmProjects_C\SchemeRep\SchemRep_tasks\PTBtasks\updatedObjectsResampled"+chr(92)



# %%
pattern_layer1 = []


for i in tqdm(range( len(filelist) )):
    x = Image.open( padImagePath + filelist[i] ) 
    x = np.array(x, dtype=np.uint8)
    
    x = Ts(x).unsqueeze(0)
    
    x = model.features[0](x)
    x = model.features[1](x)
    
    pattern_layer1.append(x.numpy())
    
# %% get the patterns and put them into a m x n matrix.
# m: number of neurons, n: number of pics

print(pattern_layer1[0].shape)

layer1dat = np.zeros((64*224*224,len(filelist)))

for i,pat in enumerate(pattern_layer1):
    layer1dat[:,i] = pat.flatten()
# %%
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(layer1dat.T)
# %%
tmat_skl = pca.transform(layer1dat.T)


RSM_skl = np.corrcoef(tmat_skl.T)
pd.DataFrame(RSM_skl).to_csv('RSM_skl_lifu_backup.csv')
quit()

tmat_manual = layer1dat.T @ pca.components_.T

RSM_manual = np.corrcoef(tmat_manual.T)
RSM_orig = np.corrcoef(layer1dat.T)


# %%
# method 3: calculate the std of each neuron, rank the neurons from high to low

# first, estimate the proportion of white area

all_img = np.zeros((300*300*3,114))
for i in tqdm(range( len(filelist) )):
    x = Image.open( padImagePath + filelist[i] ) 
    x = np.array(x, dtype=np.uint8)
    all_img[:,i] = x.flatten()
    
all_img_pixstd = np.mean(all_img,axis=1,keepdims=True)

all_img_pixstd = np.reshape(all_img_pixstd,newshape=(300,300,3))

plt.imshow(all_img_pixstd/max(all_img_pixstd.flatten()))
# %%

layer1neuro_std = np.std(layer1dat, axis=(1),keepdims=True)

print(layer1neuro_std.shape)

layer1mask = layer1neuro_std>(0.5*max(layer1neuro_std.flatten()))

print(sum(layer1mask))


layer1dat_highvar = layer1dat[layer1mask.flatten()==1,:]
RSM_highvar = np.corrcoef(layer1dat_highvar.T)

# %%
# weighted correlation
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#RSM_weightedcorr = np.cov(m=layer1dat.T,aweights=layer1neuro_std.flatten())
RSM_weightedcorr = np.zeros((114,114))
for i in range(114):
    for j in range(114):
        a=np.cov(m = layer1dat[:,i].reshape(1,-1),
                aweights = layer1neuro_std.flatten())
        b=np.cov(m = layer1dat[:,j].reshape(1,-1),
                aweights = layer1neuro_std.flatten())
        
        RSM_weightedcorr[i,j] = (np.cov(
            m = layer1dat[:,i].reshape(1,-1),
            y = layer1dat[:,j].reshape(1,-1),
            aweights = layer1neuro_std.flatten())[0,1])/np.sqrt(a*b)
# %%
# covariance
RSM_cov = np.cov(layer1dat.T)
RSM_weighted_cov = np.cov(layer1dat.T,aweights=layer1neuro_std.flatten())
# %%

stim_ID_num = np.arange(114).reshape((1,114)) + 1
#io.savemat('RSM_VGG16_cov.mat',{'R':RSM_cov, 'stim_ID_num':stim_ID_num})
#io.savemat('RSM_VGG16_weighted_cov.mat',{'R':RSM_weighted_cov, 'stim_ID_num':stim_ID_num})
io.savemat('RSM_VGG16_PCA.mat',{'R':RSM_skl, 'stim_ID_num':stim_ID_num})
#io.savemat('RSM_VGG16_highvar50.mat',{'R':RSM_highvar, 'stim_ID_num':stim_ID_num})
#io.savemat('RSM_VGG16_highvar_weightedcorr.mat',{'R':RSM_weightedcorr, 'stim_ID_num':stim_ID_num})
#io.savemat('RSMcornetV1.mat',{'R':RDM_cornetV1, 'stim_ID_num',stim_ID_num})x
#io.savemat('RSMcornetV2.mat',{'R':RDM_cornetV2, 'stim_ID_num',stim_ID_num})
#io.savemat('RSMcornetV4.mat',{'R':RDM_cornetV4, 'stim_ID_num',stim_ID_num})
#io.savemat('RSMcornetIT.mat',{'R':RDM_cornetIT, 'stim_ID_num',stim_ID_num})
#io.savemat('RSMcornet_decoder.mat',{'R':RDM_cornet_decoder, 'stim_ID_num',stim_ID_num})
