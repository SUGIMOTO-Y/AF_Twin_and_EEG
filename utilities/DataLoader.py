import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class get_dataset:
    def __init__(self, args):
        sub = args.TargetSub
        self.device = args.device
        self.sub = '0'+str(sub) if sub < 10 else str(sub) 
        self.test_eeg = self.load_eeg()
        self.train_eeg = self.load_eeg(True)
        self.test_img, self.train_img = self.load_img()
        self.test_label, self.train_label = self.load_label()
        self.test_bacth = int(self.test_label.shape[0])
        self.train_bacth = int(self.train_label.shape[0])
        
    def load_eeg(self, train=False):
        mode = 'training' if train else 'test'
        preprocessed_eeg = np.load(os.path.join('Preprocesssed_ThingEEGDataset', f'sub-{self.sub}',  f'preprocessed_eeg_{mode}.npy'), allow_pickle=True).item()['preprocessed_eeg_data']
        eeg_per_repet = [preprocessed_eeg[:, i] for i in range(preprocessed_eeg.shape[1])]
        del preprocessed_eeg
        eeg_per_repet = np.mean(eeg_per_repet, 0)
        return self.ChannelSelect(eeg_per_repet)
    
    def ChannelSelect(self, eeg):
        SelectedChannels = [1, 6, 11, 13, 12, 4, 3, 5]
        sig_ = np.zeros([eeg.shape[0], len(SelectedChannels), eeg.shape[2]])
        for i, ii in enumerate(SelectedChannels):
            sig_[:,  i, :] = eeg[:, ii, :] 
        eeg = sig_
        return eeg
    
    def load_label(self):
        # load_label_ = lambda x : np.load(os.path.join(self.dir, x))[:,np.newaxis]
        load_label_ =  lambda x : np.array([i for i in range(x.shape[0])])[:,np.newaxis]
        return load_label_(self.test_img),  load_label_(self.train_img) 
        
    def load_img(self):
        load_img_ = lambda x : np.transpose(np.load(os.path.join('PretrainingWeights', x)), (0, 3, 2, 1))   
        return load_img_('TestImg128.npy'),  load_img_('TrainImg128.npy')
        
    @property
    def get_test(self):       
        dataloader = Torch_Dataloader(self.test_eeg, self.test_label, self.device)
        return DataLoader(dataloader, batch_size = self.test_bacth, drop_last=True, shuffle=False)
    
    @property
    def get_train(self):       
        dataloader = Tree_modalities_Dataloader((self.train_eeg, self.train_label), self.train_img, self.device)
        return DataLoader(dataloader, batch_size = self.train_bacth, drop_last=True, shuffle=True)
    
    @property
    def get_img(self):  
        return self.test_img
                   
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
    
class Tree_modalities_Dataloader(Dataset):
    def __init__(self, Sig, Img, device):
        self.TX, self.Y = Sig
        self.IX = np.transpose(Img, (0, 3, 2, 1))
        self.device = device
        self.annotations = self.TX.shape[0]

    def __len__(self):
        return self.annotations        
        
    def __getitem__(self, index):
        Sig = torch.Tensor(self.TX[index, :]).to(self.device, torch.float32)
        label = torch.Tensor(self.Y[index]).to(self.device, torch.float32)
        ImgIndex = label.cpu().numpy().astype(int)
        Img = torch.Tensor(self.IX[ImgIndex[0], :]).to(self.device, torch.float32)
        Sig = Sig[None,:]
        return (Sig, Img, label)