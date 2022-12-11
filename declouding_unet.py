#!/usr/bin/env python
# coding: utf-8

# In[26]:


import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST,StanfordCars
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import math
import pandas as pd
from PIL import Image
from tqdm import tqdm


# In[2]:


import wandb


# In[3]:


wandb.init()


# In[4]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[5]:


# Defining the device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[6]:


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(480,720)):
        super().__init__()
        self.out_size    =out_sz
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.sigmoid = nn.Sigmoid()
        self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        out      = self.sigmoid(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_size)
        return out


# In[7]:


# dims=1024
model = UNet(num_class=4,retain_dim=True,out_sz=(480,720)).to(device)
test=torch.ones(1,3,480,720).to(device)
model(test)


# In[8]:



# Setting the optimiser

learning_rate = 1e-3*5

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# In[9]:


# Reconstruction + KL divergence losses summed over all elements and batch

def loss_function(ỹ, y):
    BCE = nn.functional.binary_cross_entropy(
        ỹ, y,reduction='sum'
    )
#     KLD = (-0.5 * torch.mean(-logvar1.exp() + logvar1 + 1.0 - mu1.pow(2)))
    return BCE


# In[10]:


class MyDataset(Dataset):
    def __init__(self, train_path,transform_x=None,transform_y=None):
        self.df = pd.read_csv(train_path, sep=',', usecols=['input', 'output'])
        self.transform_x=transform_x
        self.transform_y=transform_y
    def __getitem__(self, index):
#         print(self.df.iloc[index, 1])
#         print(self.df.iloc[index, 0])
        x = Image.open(self.df.iloc[index, 1])
        y = Image.open(self.df.iloc[index, 0])
        if self.transform_x is not None:
            x=self.transform_x(x)
            y=self.transform_y(y)
        else:
            x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y
    def __len__(self):
#         return len(self.df)
        return 3000


# In[11]:


class MyDataset_np(Dataset):
    def __init__(self, train_path,transform_x=None,transform_y=None):
        self.df = pd.read_csv(train_path, sep=',', usecols=['input', 'output'])
        self.transform_x=transform_x
        self.transform_y=transform_y
    def __getitem__(self, index):
#         print(self.df.iloc[index, 1])
#         print(self.df.iloc[index, 0])
        x = np.array(Image.open(self.df.iloc[index, 1]))
        y = np.array(Image.open(self.df.iloc[index, 0]))
        if self.transform_x is not None:
            x=self.transform_x(x)
            y=self.transform_y(y)
        else:
            x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y
    def __len__(self):
#         return len(self.df)
        return 10


# In[12]:


epochs = 1000
batch_size = 4


# In[13]:


wandb.config = {
  "learning_rate": learning_rate,
  "epochs": epochs,
  "batch_size": batch_size,
}


# In[14]:


wandb.init(project="AerialPoseEstimator")


# In[15]:


train_loader=MyDataset_np("./dataset_train.csv")
test_loader=MyDataset_np("./dataset_test.csv")
train_loader=DataLoader(train_loader, batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_loader, batch_size=batch_size,shuffle=True)


# In[16]:


wandb.watch(model)


# In[17]:


def batch_mean_x(loader):
    cnt=0
    fst_moment=torch.empty(3)
    snd_moment=torch.empty(3)
    for images,_ in loader:
        # c h w b
#         print(images.shape)
        images=images/255
        b,h,w,c = images.shape
        nb_pixels=b * h * w
        sum_ =  torch.sum(images,dim=[0,1,2])
        sum_of_square = torch.sum(images**2,dim=[0,1,2])
        
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / ( cnt + nb_pixels)
        
        cnt+=nb_pixels
    mean,std=fst_moment,torch.sqrt(snd_moment - fst_moment ** 2)
    return mean,std


# In[18]:


def batch_mean_y(loader):
    cnt=0
    fst_moment=torch.empty(4)
    snd_moment=torch.empty(4)
    for _,images in loader:
        images=images/255
        b,h,w,c = images.shape
        nb_pixels=b * h * w
        sum_ =  torch.sum(images,dim=[0,1,2])
        sum_of_square = torch.sum(images**2,dim=[0,1,2])
        
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / ( cnt + nb_pixels)
        
        cnt+=nb_pixels
    mean,std=fst_moment,torch.sqrt(snd_moment - fst_moment ** 2)
    return mean,std


# In[19]:


mean_x,std_x=batch_mean_x(train_loader)
mean_y,std_y=batch_mean_y(train_loader)


# In[20]:


train_loader=MyDataset("./dataset_train.csv")
test_loader=MyDataset("./dataset_test.csv")
train_loader=DataLoader(train_loader, batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_loader, batch_size=batch_size,shuffle=True)


# In[21]:


transform_img_normal_x = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = mean_x,std= std_x)
])
transform_img_normal_y = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = 0,std = 1)
])
train_loader=MyDataset("./dataset_train.csv",
                       transform_x=transform_img_normal_x,
                       transform_y=transform_img_normal_y)
test_loader=MyDataset("./dataset_test.csv",
                      transform_x=transform_img_normal_x,
                      transform_y=transform_img_normal_y)
train_loader=DataLoader(train_loader, batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_loader, batch_size=batch_size,shuffle=True)


# In[ ]:





# In[27]:


# Training and testing the VAE
codes = dict(μ=list(), logσ2=list(), x=list())
for epoch in tqdm(range(0, epochs + 1)):
    # Training
    if epoch > 0:  # test untrained net first
        model.train()
        train_loss = 0
        for x,y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            x=x.view(-1,3,480,720)             
            y=y.view(-1,4,480,720)
            x=torch.div(x,255)
            y=torch.div(y,255)
            y_bar=model(x)
            loss = loss_function(y_bar, y)
            train_loss += loss
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================

    # Testing
        print(train_loss)
        print()
        wandb.log({"train_loss":train_loss /len(train_loader.dataset)})
        means, logvars, labels = list(), list(), list()
        if epoch%10==0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,}, 
                       "./Weights/resnet_unet.pt")
    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for x,y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            x=x.view(-1,3,480,720)
            y=y.view(-1,4,480,720)
            x=torch.div(x,255)
            y=torch.div(y,255)
            # ===================forward=====================
            ỹ = model(x)
            loss = loss_function(ỹ, y)
            test_loss+=loss.item()
    test_loss /= len(test_loader.dataset)
    print(test_loss)
    print()
    wandb.log({"test_loss":test_loss /len(test_loader.dataset)})




