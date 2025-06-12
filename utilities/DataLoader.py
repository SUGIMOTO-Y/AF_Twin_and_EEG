import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class get_dataset:
    def __init__(self, args):
        sub = args.TargetSub
        self.device = args.device
        self.sub = '0'+str(sub) if sub < 10 else str(sub) 
        self.dir = os.path.join('Preprocesssed_ThingEEGDataset')
        self.eeg = self.load_eeg()
        self.ChannelSelect()
        self.label = self.load_label()
        self.TestBatchSize = int(self.label.shape[0])
        self.img = self.load_img()
        
    def load_eeg(self):
        preprocessed_eeg = np.load(os.path.join(self.dir, f'sub-{self.sub}',  'preprocessed_eeg_test.npy'), allow_pickle=True).item()['preprocessed_eeg_data']
        eeg_per_repet = [preprocessed_eeg[:, i] for i in range(preprocessed_eeg.shape[1])]
        del preprocessed_eeg
        eeg_per_repet = np.mean(eeg_per_repet, 0)
        return eeg_per_repet   
    
    def ChannelSelect(self):
        SelectedChannels = [1, 6, 11, 13, 12, 4, 3, 5]
        sig_ = np.zeros([self.eeg.shape[0], len(SelectedChannels), self.eeg.shape[2]])
        for i, ii in enumerate(SelectedChannels):
            sig_[:,  i, :] = self.eeg[:, ii, :] 
        self.eeg = sig_
    
    def load_label(self):
        return np.load(os.path.join(self.dir, 'TestLabel.npy'))[:,np.newaxis]
        
    def load_img(self):
        Img = np.load(os.path.join(self.dir, 'TestImg128.npy'))        
        return np.transpose(Img, (0, 3, 2, 1))
        
    @property
    def get_eeg(self):       
        test_dataloader = Torch_Dataloader(self.eeg, self.label, self.device)
        return DataLoader(test_dataloader, batch_size = self.TestBatchSize, drop_last=True, shuffle=False)
    
    @property
    def get_img(self):  
        return self.img
                   
class Torch_Dataloader(Dataset):
    def __init__(self, Sig, Lab, device):
        self.Sig = Sig
        self.Lab = Lab
        self.device = device
        self.annotations = self.Sig.shape[0]

    def __len__(self):
        return self.annotations        
        
    def __getitem__(self, index):
        Sig = torch.Tensor(self.Sig[index, :]).to(self.device, torch.float32)
        Sig = Sig[None,:]
        Lab = torch.Tensor(self.Lab[index]).to(self.device, torch.float32)
        return (Sig, Lab)
    
