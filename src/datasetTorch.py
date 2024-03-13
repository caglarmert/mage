from __future__ import print_function
from torch.utils.data import Dataset 
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
cmap = plt.get_cmap("Pastel1")
##############################################################################

class structureData(Dataset):
    def __init__(self,attributes, phase, **kwargs):
        
        # Get the 
        self.attributes = attributes
        self.phase = phase
        self.aug = kwargs.get('augmentation_flag')
        self.freq = kwargs.get('freq_flag')
        self.aug_param = kwargs.get('Augmentation parameter')
        self.signaltime =  kwargs.get('signaltime')
        self.fs = kwargs.get("fs")
        self.startP = attributes["tpga_ind"]
        self.gtnorm = kwargs.get("gtnorm")
        self.window_size = kwargs.get("window_size")
        
        if self.freq == True:
            
            if self.phase == "training":
                self.signals = [attributes["Signal"][i] for i in attributes["trind"]]
                self.groundTruth = [attributes["groundTruth"][i] for i in attributes["trind"]]
                self.stat_info = [attributes["stat_info"][i] for i in attributes["trind"]]
            
            if self.phase == "validation":
                self.signals = [attributes["Signal"][i] for i in attributes["vlind"]]
                self.groundTruth = [attributes["groundTruth"][i] for i in attributes["vlind"]]
                self.stat_info = [attributes["stat_info"][i] for i in attributes["vlind"]]
            
            if self.phase == "test":
                self.signals = [attributes["Signal"][i] for i in attributes["tsind"]]
                self.groundTruth = [attributes["groundTruth"][i] for i in attributes["tsind"]]
                self.stat_info = [attributes["stat_info"][i] for i in attributes["tsind"]]
                
        else:
            if self.phase == "training":
                self.signals = [attributes["Signal"][i] for i in attributes["trind"]]
                self.groundTruth = [attributes["groundTruth"][i] for i in attributes["trind"]]
                self.stat_info = [attributes["stat_info"][i] for i in attributes["trind"]]
            if self.phase == "validation":
                self.signals = [attributes["Signal"][i] for i in attributes["vlind"]]
                self.groundTruth = [attributes["groundTruth"][i] for i in attributes["vlind"]]
                self.stat_info = [attributes["stat_info"][i] for i in attributes["vlind"]]
            if self.phase == "test":
                self.signals = [attributes["Signal"][i] for i in attributes["tsind"]]
                self.groundTruth = [attributes["groundTruth"][i] for i in attributes["tsind"]]                
                self.stat_info = [attributes["stat_info"][i] for i in attributes["tsind"]]
                
                
    def __getitem__(self, index):
        
        gt = self.groundTruth[index].clone()
        k = self.fs
        l = self.signaltime * self.fs
        gtnorm = self.gtnorm
        
        if self.freq:
            k = self.window_size*2 #overlap constant
            l = 2*self.signaltime - 1
                 
        self.startP = self.attributes["tpga_ind"][index]
        
        if self.phase=="training" and (self.aug):
            self.startP = random.randint(0,(2*self.aug))
              
        sig_to_return = torch.tensor(self.signals[index][self.startP * k : self.startP * k + l][:][:]).clone()# 19x51x3
        # sig_to_return = torch.from_numpy(self.signals[index][self.startP * k : self.startP * k + l,:,:]).clone() # 19x51x3

        sig_to_return = np.transpose(sig_to_return,(2,0,1)) #(3, 19, 51)
        
        sig_to_return[0,:,:] = sig_to_return[0,:,:] - self.attributes["trMeanR"]
        sig_to_return[1,:,:] = sig_to_return[1,:,:] - self.attributes["trMeanG"]
        sig_to_return[2,:,:] = sig_to_return[2,:,:] - self.attributes["trMeanB"]
        
        
        # Normalize station information, i.e., latitude (deg), longitude (deg), altitude (km) info of station.
        stat_info = self.stat_info[index].clone()
        stat_info[0] = stat_info[0] - self.attributes["MeanStatLat"]
        stat_info[1] = stat_info[1] - self.attributes["MeanStatLon"]
       # stat_info[2] = stat_info[2] - self.attributes["MeanStatAlt"]
        
        if gtnorm:            
            for channel in range(len(gt)):          
                gt[channel] = (gt[channel] - self.attributes["trMeans"][channel]) / self.attributes["trStds"][channel]   
                
        return sig_to_return, stat_info, gt

        
    def __len__(self):

        return len(self.signals) ## len(self.images)

##############################################################################