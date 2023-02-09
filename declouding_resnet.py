#!/usr/bin/env python
# coding: utf-8
 
# In[1]:
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import math
from PIL import Image
from tqdm import tqdm
import os
import cv2
import pandas as pd
import wandb
import torch
from PIL import ImageFile
import res_encoder as enc
import res_decoder as dec
 
latent_size=256
 
class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.enc=enc.ResNet(enc.Bottleneck,[3,6,4,3])
		self.mlp=nn.Linear(2048,latent_size)
		self.mlp2=nn.Linear(latent_size,2048)
		self.dec=dec.ResNet(dec.Bottleneck,[3,6,4,3])
	def forward(self,x):
		x=self.enc(x)[0]
		x=x.view((-1,2048))
		x=self.mlp(x)
		x=self.mlp2(x)
		x=x.view(-1,2048,1,1)
		x=self.dec(x)
		return x
model=Net()
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
 
learning_rate = 1e-3*5
 
optimizer = torch.optim.Adam(
	model.parameters(),
	lr=learning_rate,
)
# Reconstruction + KL divergence losses summed over all elements and batch
 
def loss_function(ỹ, y):
	BCE = nn.functional.binary_cross_entropy(
		ỹ, y,reduction='sum'
	)
	return BCE
import cv2
class MyDataset(Dataset):
	def __init__(self, train_path,transform_x=None,transform_y=None):
		self.df = pd.read_csv(train_path, sep=',', 
usecols=['input', 'output'])
		self.transform_x=transform_x
		self.transform_y=transform_y
	def __getitem__(self, index):
		x = cv2.imread(self.df.iloc[index, 1])
		y = cv2.imread(self.df.iloc[index, 1])
		if self.transform_x is not None:
			x=self.transform_x(x)
			y=self.transform_y(y)
		else:
			x, y = torch.from_numpy(x), torch.from_numpy(y)
		return x, y,self.df.iloc[index,1],self.df.iloc[index,0]
	def __len__(self):
		return 4000
class MyDataset_np(Dataset):
	def __init__(self, train_path,transform_x=None,transform_y=None):
		self.df = pd.read_csv(train_path, sep=',', 
usecols=['input', 'output'])
		self.transform_x=transform_x
		self.transform_y=transform_y
	def __getitem__(self, index):
		x = np.array(Image.open(self.df.iloc[index, 
1]).convert("RGB"))
		y = np.array(Image.open(self.df.iloc[index, 
0]).convert("RGB"))
		if self.transform_x is not None:
			x=self.transform_x(x)
			y=self.transform_y(y)
		else:
			x, y = torch.from_numpy(x), torch.from_numpy(y)
		return x, y,self.df.iloc[index,1],self.df.iloc[index,0]
	def __len__(self):
		return 3000
epochs = 30
batch_size = 4
# wandb.config = {
#   "learning_rate": learning_rate,
#   "epochs": epochs,
#   "batch_size": batch_size,
# }
# wandb.init(project="AerialPoseEstimator")
# wandb.watch(model)
train_loader=MyDataset("./dataset_train2.csv")
test_loader=MyDataset("./dataset_test.csv")
train_loader=DataLoader(train_loader, batch_size=batch_size,shuffle=False)
test_loader=DataLoader(test_loader, batch_size=batch_size,shuffle=False)
 
# path="./Weights/resnet.pt"
# isExist = os.path.exists(path)
# if isExist:
# 	checkpoint=torch.load("./Weights/crossloc.pt")
# 	model.load_state_dict(checkpoint["model_state_dict"])
# 	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# Training and testing the VAE
model=model.to(device)
codes = dict(μ=list(), logσ2=list(), x=list())
T=transforms.ToPILImage()
for epoch in tqdm(range(0, epochs + 1)):
	# Training
	if epoch > 0:  # test untrained net first
		model.train()
		train_loss = 0
		for x,y,_,_ in tqdm(train_loader):
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
		print(train_loss)
# 		wandb.log({"train_loss":train_loss 
# /len(train_loader.dataset)})
		means, logvars, labels = list(), list(), list()
		if epoch%2==0:
			torch.save({'epoch': epoch,
						'model_state_dict': 
model.state_dict(),
						'optimizer_state_dict': 
optimizer.state_dict(),
						'loss': train_loss,}, 
					   "./Weights/resnet_crossloc.pt")
	torch.cuda.empty_cache()
	with torch.no_grad():
		model.eval()
		test_loss = 0
		counter=0
		if not os.path.exists("./outputs/"+str(epoch)):
			os.mkdir("./outputs/"+str(epoch))
		for x,y,xname,yname in tqdm(test_loader):
			x = x.to(device)
			y = y.to(device)
			x=x.view(-1,3,480,720)
			y=y.view(-1,3,480,720)
			x=torch.div(x,255)
			y=torch.div(y,255)
			# ===================forward=====================
			ỹ = model(x)
			if counter%10==0 and epoch%2==0:
				image_y=ỹ[0]
				image_x=x[0]
				print(image_y.shape)
				print(image_x.shape)
				image_y = torch.mul(image_y,255)
				image_x = torch.mul(image_x,255)
				image_y=image_y.view(480,720,3)
				image_x=image_x.view(480,720,3)
				path="./outputs/"+str(epoch)
				isExist = os.path.exists(path)
				if not isExist:
					os.makedirs(path)
				# cv2.imshow("hello",image_y)
				# cv2.imshow("hello",image_x)
				
				print(image_y.cpu().detach().numpy().shape)
				image_y = Image.fromarray(image_y.cpu().detach().numpy().astype(np.uint8))
				image_y = image_y.convert('RGB')
				xname=xname[0]
				xname=xname.split("/")
				image_y.save("./outputs/"+xname[-1])
				image_x = Image.fromarray(image_x.cpu().detach().numpy().astype(np.uint8))
				image_x=image_x.convert("RGB")
				yname=yname[0]
				yname=yname.split("/")
				image_x.save("./outputs/"+yname[-1])
			loss = loss_function(ỹ, y)
			test_loss+=loss.item()
			counter=counter+1
	test_loss /= len(test_loader.dataset)
	print(test_loss)
	# wandb.log({"test_loss":test_loss /len(test_loader.dataset)})
	print(epoch)


 
 
