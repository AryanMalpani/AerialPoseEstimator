#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torch.nn as nn
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


def _create_res_block(tiny, num_gn_channel, ch_down_factor=1):
    """Create residual block"""
    num_ch = (512, 128)[tiny] // ch_down_factor
    res_block = nn.Sequential(nn.Conv2d(num_ch, num_ch, 3, 1, 1),
                              nn.GroupNorm(min(num_gn_channel, num_ch), num_ch),
                              nn.ReLU(),
                              nn.Conv2d(num_ch, num_ch, 1, 1, 0),
                              nn.GroupNorm(min(num_gn_channel, num_ch), num_ch),
                              nn.ReLU(),
                              nn.Conv2d(num_ch, num_ch, 3, 1, 1),
                              nn.GroupNorm(min(num_gn_channel, num_ch), num_ch),
                              nn.ReLU()
                              )
    return res_block


# In[7]:


class Encoder(nn.Module):
    def __init__(self,tiny,enc_add_res_block=0,num_gn_channel=32):
        super(Encoder,self).__init__()
        self.tiny=tiny
        self.enc=enc_add_res_block
        self.gn_channel=num_gn_channel
        self.num_gn_channel = num_gn_channel
        self.conv1 = nn.Conv2d(3, num_gn_channel, 3, 1, 1)
        self.norm1 = nn.GroupNorm(num_gn_channel, num_gn_channel)
        self.conv2 = nn.Conv2d(num_gn_channel, 64, 3, 2, 1)
        self.norm2 = nn.GroupNorm(num_gn_channel, 64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.norm3 = nn.GroupNorm(num_gn_channel, 128)
        self.conv4 = nn.Conv2d(128, (256, 128)[tiny], 3, 2, 1)
        self.norm4 = nn.GroupNorm(num_gn_channel, (256, 128)[tiny])

        self.res1_conv1 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 3, 1, 1)
        self.res1_norm1 = nn.GroupNorm(num_gn_channel, (256, 128)[tiny])
        self.res1_conv2 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 1, 1, 0)
        self.res1_norm2 = nn.GroupNorm(num_gn_channel, (256, 128)[tiny])
        self.res1_conv3 = nn.Conv2d((256, 128)[tiny], (256, 128)[tiny], 3, 1, 1)
        self.res1_norm3 = nn.GroupNorm(num_gn_channel, (256, 128)[tiny])

        self.res2_conv1 = nn.Conv2d((256, 128)[tiny], (512, 128)[tiny], 3, 1, 1)
        self.res2_norm1 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.res2_conv2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res2_norm2 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.res2_conv3 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 3, 1, 1)
        self.res2_norm3 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])

        if not tiny:
            self.res2_skip = nn.Conv2d(256, 512, 1, 1, 0)
            self.res2_skip_norm = nn.GroupNorm(num_gn_channel, 512)
        self.enc_add_res_block_ls = [_create_res_block(tiny, num_gn_channel) for _ in range(enc_add_res_block)]
        for i, block in enumerate(self.enc_add_res_block_ls):
            self.add_module('enc_add_res_block{:d}'.format(i+1), block)
    def forward(self, inputs):
        x = inputs
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        res = F.relu(self.norm4(self.conv4(x)))

        x = F.relu(self.res1_norm1(self.res1_conv1(res)))
        x = F.relu(self.res1_norm2(self.res1_conv2(x)))
        x = F.relu(self.res1_norm3(self.res1_conv3(x)))

        res = F.relu(res + x)

        x = F.relu(self.res2_norm1(self.res2_conv1(res)))
        x = F.relu(self.res2_norm2(self.res2_conv2(x)))
        x = F.relu(self.res2_norm3(self.res2_conv3(x)))

        if not self.tiny:
            res = self.res2_skip_norm(self.res2_skip(res))

        res = F.relu(res + x)

        # additional residual block
        for i in range(len(self.enc_add_res_block_ls)):
            x = self.enc_add_res_block_ls[i](res)
            res = F.relu(res + x)

        return res
       


# In[8]:


class DenseUpsamplingConvolution(nn.Module):
    def __init__(self, down_sampling_rate, in_channel, num_classes, num_gn_channel=32):
        super(DenseUpsamplingConvolution, self).__init__()
        up_sampling_channel = (down_sampling_rate ** 2) * num_classes
        self.conv = nn.Conv2d(in_channel, up_sampling_channel, 3, 1, 1)
        self.norm = nn.GroupNorm(num_gn_channel, up_sampling_channel)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_sampling_rate)

    def forward(self, x):
        x = self.relu(self.norm(self.conv(x)))
        x = self.pixel_shuffle(x)
        return x


# In[9]:


class Decoder(nn.Module):
    """A modular decoder for TransPose network."""
    def __init__(self, tiny, dec_add_res_block=0, num_gn_channel=32, full_size_output=False):
        super(Decoder, self).__init__()

        # learned output relative to its mean (e.g. center of the scene)
        self.tiny = tiny
        self.dec_add_res_block = dec_add_res_block
        self.num_gn_channel = num_gn_channel
        self.full_size_output = full_size_output

        # Additional residual block could be added on top of the vanilla decoder.
        self.dec_add_res_block_ls = [_create_res_block(tiny, num_gn_channel) for _ in range(dec_add_res_block)]
        for i, block in enumerate(self.dec_add_res_block_ls):
            self.add_module('dec_add_res_block{:d}'.format(i+1), block)

        self.res3_conv1 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res3_norm1 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.res3_conv2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res3_norm2 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])
        self.res3_conv3 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.res3_norm3 = nn.GroupNorm(num_gn_channel, (512, 128)[tiny])

        self.fc1 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.fc1_norm = nn.GroupNorm(min((512, 128)[tiny], num_gn_channel), (512, 128)[tiny])
        self.fc2 = nn.Conv2d((512, 128)[tiny], (512, 128)[tiny], 1, 1, 0)
        self.fc2_norm = nn.GroupNorm(min((512, 128)[tiny], num_gn_channel), (512, 128)[tiny])
        if full_size_output:
            # upsampling for semantics task
            self.duc_upsample = DenseUpsamplingConvolution(down_sampling_rate=8, in_channel=(512, 128)[tiny],
                                                           num_classes=3)
            self.fc3 = nn.Conv2d(3, 3, 1, 1, 0)
        else:
            self.fc3 = nn.Conv2d((512, 128)[tiny], 4, 1, 1, 0)

    def forward(self, inputs, up_height=None, up_width=None):
        """
        Forward pass.
        @param inputs           4D data tensor (BxCxHxW)
        @param up_height        Scalar, up-sampling target tensor height
        @param up_width         Scalar, up-sampling target tensor width
        """

        res = inputs

        # additional residual block
        # self.dec_add_res_block_ls[0][0] or self.res3_conv1 layer input is the intermediate activation [feature vec.].
        for i in range(len(self.dec_add_res_block_ls)):
            x = self.dec_add_res_block_ls[i](res)
            res = F.relu(res + x)

        x = F.relu(self.res3_norm1(self.res3_conv1(res)))
        x = F.relu(self.res3_norm2(self.res3_conv2(x)))
        x = F.relu(self.res3_norm3(self.res3_conv3(x)))

        res = F.relu(res + x)

        sc = F.relu(self.fc1_norm(self.fc1(res)))
        sc = F.relu(self.fc2_norm(self.fc2(sc)))
        if self.full_size_output:
            # upsampling for semantics task
            sc = self.duc_upsample(sc)  # [B, C, H', W']
            sc = F.interpolate(sc, (up_height, up_width), mode='bilinear', align_corners=False)  # trim dimensions

        sc = self.fc3(sc)
        sc=F.sigmoid(sc)
        return sc


# In[10]:


def _create_mlr_concatenator(num_mlr, tiny, num_gn_channel):
    """Create activation concatenation block for MLR."""
    in_channel = (512, 128)[tiny] * num_mlr
    out_channel = (512, 128)[tiny]
    mlr_block = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                              nn.GroupNorm(num_gn_channel, out_channel),
                              nn.ReLU(),
                              nn.Conv2d(out_channel, out_channel, 1, 1, 0),
                              nn.GroupNorm(num_gn_channel, out_channel),
                              nn.ReLU(),
                              nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                              nn.GroupNorm(num_gn_channel, out_channel),
                              nn.ReLU()
                              )
    return mlr_block


def _create_mlr_skip_layer(num_mlr, tiny, num_gn_channel):
    """Create skip layer for MLR"""
    in_channel = (512, 128)[tiny] * num_mlr
    out_channel = (512, 128)[tiny]
    skip_block = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1, 0),
                               nn.GroupNorm(num_gn_channel, out_channel))
    return skip_block


# In[11]:


class Net(nn.Module):
    """
    Flexible FCN architecture for various regression tasks.
    The output is sub-sampled by a factor of 8 compared to the image input.
    Contents of changes:
    - Added non-grayscale RGB image input.
    - Added group normalization.
    - Added encoder/decoder separation and supported an arbitrary number of residual blocks.
    - Added support for arbitrary-channel regression task output and positive-value uncertainty output.
    """

    def __init__(self,tiny, enc_add_res_block=0, dec_add_res_block=0,num_gn_channel=32,
                 num_mlr=0, num_unfrozen_encoder=0, full_size_output=False):
        """
        Constructor.
        @param mean                 Mean offset for task output.
        @param tiny                 Flag for tiny network.
        @param grayscale            Flag for grayscale image input.
        @param enc_add_res_block    Number of additional DSAC* style residual block for encoder.
        @param dec_add_res_block    Number of additional DSAC* style residual block for decoder.
        @param num_task_channel     Number of channels for underlying task.
        @param num_pos_channel      Number of channels for additional task w/ positive values, e.g., uncertainty.
        @param num_gn_channel       Number of group normalization channels, a hyper-parameter.
        @param num_mlr              Number of homogeneous mid-level representations encoders.
        @param num_unfrozen_encoder Number of encoders that are not frozen.
        @param full_size_output     Flag for full-size network output (by using DUC-style layers).
        Note: if enc_add_res_block == dec_add_res_block == 0 && num_task_channel == 3 && num_pos_channel = 0,
        the model become DSAC* net + group normalization only.
        """
        super(Net, self).__init__()

        """Init"""
        # learned output relative to its mean (e.g. center of the scene)
        self.tiny = tiny
        self.enc_add_res_block = enc_add_res_block
        self.dec_add_res_block = dec_add_res_block
        self.num_gn_channel = num_gn_channel
        self.num_mlr = num_mlr
        self.full_size_output = full_size_output

        self.OUTPUT_SUBSAMPLE = 1 if full_size_output else 8

        """Vanilla encoder"""
        if num_mlr == 0:
            self.encoder = Encoder(tiny,enc_add_res_block, num_gn_channel)
            self.encoder_ls = [self.encoder]
        else:
            self.encoder = nn.Identity()
            self.encoder_ls = [self.encoder]

        """MLR encoders"""
        if num_mlr > 0 and isinstance(num_mlr, int):
            assert 0 <= num_unfrozen_encoder <= num_mlr
            self.mlr_encoder_ls = [Encoder(tiny, enc_add_res_block, num_gn_channel) for _ in range(num_mlr)]
            # Freeze gradients of the re-used encoder
            for i, block in enumerate(self.mlr_encoder_ls):
                if i >= num_unfrozen_encoder:  # the first few encoders **may** be reused for training
                    for param in block.parameters():
                        param.requires_grad = False
                self.add_module('mlr_encoder_{:d}'.format(i + 1), block)
            self.mlr_norm = nn.GroupNorm(num_gn_channel, (512, 128)[tiny] * num_mlr)
            self.mlr_forward = _create_mlr_concatenator(num_mlr, tiny, num_gn_channel)
            self.mlr_skip = _create_mlr_skip_layer(num_mlr, tiny, num_gn_channel)  # normalization is included
        else:
            self.mlr_encoder_ls = [nn.Identity()]
            self.mlr_norm = nn.Identity()
            self.mlr_forward = nn.Identity()
            self.mlr_skip = nn.Identity()
        self.mlr_ls = self.mlr_encoder_ls + [self.mlr_norm, self.mlr_forward, self.mlr_skip]

        """Decoder"""
        # we always have a decoder regardless of the #MLR
        self.decoder = Decoder(tiny, dec_add_res_block, num_gn_channel, full_size_output)
        self.decoder_ls = [self.decoder]

    def forward(self, inputs):
        """
        Forward pass.
        @param inputs           4D data tensor (BxCxHxW)
        """

        x = inputs
        up_height, up_width = inputs.size()[2:4]

        """Vanilla encoder"""
        if self.num_mlr == 0:
            res = self.encoder(x)
        else:
            res = None

        """MLR encoder"""
        if self.num_mlr:
            # inference
            mlr_activation_ls = [mlr_enc(inputs) for mlr_enc in self.mlr_encoder_ls]

            # activation concatenation
            mlr = torch.cat(mlr_activation_ls, dim=1)  # [B, C * #MLR, H, W]

            # forward
            res = self.mlr_skip(mlr)
            mlr = self.mlr_norm(mlr)
            mlr = self.mlr_forward(mlr)
            res = F.relu(res + mlr)

        """Decoder"""
        if self.full_size_output:
            sc = self.decoder(res, up_height, up_width)
        else:
            sc = self.decoder(res)

        return sc


# In[12]:


model=Net(tiny=1,enc_add_res_block=3,dec_add_res_block=3,num_gn_channel=32,num_mlr=2,num_unfrozen_encoder=2,full_size_output=True).to(device)


# In[13]:



# Setting the optimiser

learning_rate = 1e-3*5

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# In[14]:


# Reconstruction + KL divergence losses summed over all elements and batch

def loss_function(ỹ, y):
    BCE = nn.functional.binary_cross_entropy(
        ỹ, y,reduction='sum'
    )
#     KLD = (-0.5 * torch.mean(-logvar1.exp() + logvar1 + 1.0 - mu1.pow(2)))
    return BCE


# In[15]:


class MyDataset(Dataset):
    def __init__(self, train_path,transform_x=None,transform_y=None):
        self.df = pd.read_csv(train_path, sep=',', usecols=['input', 'output'])
        self.transform_x=transform_x
        self.transform_y=transform_y
    def __getitem__(self, index):
#         print(self.df.iloc[index, 1])
#         print(self.df.iloc[index, 0])
        x = np.array(Image.open(self.df.iloc[index, 1]).convert("RGB"))
        y = np.array(Image.open(self.df.iloc[index, 0]).convert("RGB"))
        if self.transform_x is not None:
            x=self.transform_x(x)
            y=self.transform_y(y)
        else:
            x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y
    def __len__(self):
#         return len(self.df)
        return 4000


# In[16]:


class MyDataset_np(Dataset):
    def __init__(self, train_path,transform_x=None,transform_y=None):
        self.df = pd.read_csv(train_path, sep=',', usecols=['input', 'output'])
        self.transform_x=transform_x
        self.transform_y=transform_y
    def __getitem__(self, index):
#         print(self.df.iloc[index, 1])
#         print(self.df.iloc[index, 0])
        x = np.array(Image.open(self.df.iloc[index, 1]))[:,:,:3]
        y = np.array(Image.open(self.df.iloc[index, 0]))[:,:,:3]
        if self.transform_x is not None:
            x=self.transform_x(x)
            y=self.transform_y(y)
        else:
            x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y
    def __len__(self):
#         return len(self.df)
        return 3000


# In[17]:


epochs = 1000
batch_size = 4


# In[18]:


wandb.config = {
  "learning_rate": learning_rate,
  "epochs": epochs,
  "batch_size": batch_size,
}


# In[19]:


wandb.init(project="AerialPoseEstimator")


# In[20]:


# train_loader=MyDataset("./dataset_train.csv")
# test_loader=MyDataset("./dataset_test.csv")
# train_loader=DataLoader(train_loader, batch_size=batch_size,shuffle=True)
# test_loader=DataLoader(test_loader, batch_size=batch_size,shuffle=True)


# In[21]:


wandb.watch(model)


# In[22]:


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


# In[23]:


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


# In[24]:


# mean_x,std_x=batch_mean_x(train_loader)
# mean_y,std_y=batch_mean_y(train_loader)


# In[25]:


train_loader=MyDataset("./dataset_train.csv")
test_loader=MyDataset("./dataset_test.csv")
train_loader=DataLoader(train_loader, batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_loader, batch_size=batch_size,shuffle=True)

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
            y=y.view(-1,3,480,720)
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
            y=y.view(-1,3,480,720)
            x=torch.div(x,255)
            y=torch.div(y,255)
            # ===================forward=====================
            ỹ = model(x)
            loss = loss_function(ỹ, y)
            test_loss+=loss.item()
    test_loss /= len(test_loader.dataset)
    print(test_loss)
    wandb.log({"test_loss":test_loss /len(test_loader.dataset)})
    print(epoch)


# In[ ]:





# In[ ]:


# test_loader=MyDataset("./dataset_test.csv")


# In[ ]:


# temp =np.array(Image.open("./Datasets/Input/Echendens-LHS_09620.png_6.png"), dtype = float)/255.0


# In[ ]:


# tem = torch.from_numpy(temp).view(-1,3,480,720)


# In[ ]:


# tem=tem.to(device,dtype=torch.float32)


# In[ ]:


# ans=(model(tem))


# In[ ]:


# ans=(ans[0]*255).detach().cpu().numpy()


# In[ ]:


# ans.shape


# In[ ]:


# img=Image.frtomarray(ans)


# In[ ]:




