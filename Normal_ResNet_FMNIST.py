#=====================================================
# Centralized (normal) learning: ResNet18 on HAM10000
# Single program
# ====================================================
import os
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame

import math
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import random
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

    
#===================================================================    
program = "Normal Learning ResNet18 on FMNIST"
print(f"---------{program}----------")              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#=============================================================================
#                         Data loading 
#=============================================================================    
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    

#=============================================================================
#                         Data preprocessing
#=============================================================================  
# Data preprocessing: Transformation 
dataset_train = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
dataset_test = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))  
train_iterator = DataLoader(dataset_train, shuffle = True, batch_size = 256)
test_iterator = DataLoader(dataset_test, batch_size = 256)


for x, y in train_iterator:
    print("shape of x = ", x.shape)
    print(type(x))
    break

#=============================================================================
#                    Model definition: ResNet18
#============================================================================= 

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet18(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

       
net_glob = ResNet18(BasicBlock, [2, 2, 2, 2], 10) # Class labels for HAM10000 = 7 

if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob = nn.DataParallel(net_glob)   # to use the multiple GPUs 

net_glob.to(device)
print(net_glob)        


#=============================================================================
#                    ML Training and Testing
#============================================================================= 

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

#==========================================================================================================================     
def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    ell = len(iterator)
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad() # initialize gradients to zero
        
        # ------------- Forward propagation ----------
        fx = model(x)
        loss = criterion(fx, y)
        acc = calculate_accuracy (fx , y)
        
        # -------- Backward propagation -----------
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / ell, epoch_acc / ell
        
def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    ell = len(iterator)
    
    with torch.no_grad():
        for (x,y) in iterator:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            fx = model(x)       
            loss = criterion(fx, y)
            acc = calculate_accuracy (fx , y)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss/ell, epoch_acc/ell
 

# =======================================================================================
epochs = 200
LEARNING_RATE = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_glob.parameters(), lr = LEARNING_RATE)

loss_train_collect = []
loss_test_collect = []
acc_train_collect = []
acc_test_collect = []
        
start_time = time.time()    
for epoch in range(epochs):
    train_loss, train_acc = train(net_glob, device, train_iterator, optimizer, criterion)
    #print(f'Train completed - {epoch} Epoch")
    test_loss, test_acc = evaluate(net_glob, device, test_iterator, criterion)
    #print(f'Test completed - {epoch} Epoch")
    
    loss_train_collect.append(train_loss)
    loss_test_collect.append(test_loss)
    acc_train_collect.append(train_acc)
    acc_test_collect.append(test_acc)
    
    
    prRed(f'Train => Epoch: {epoch} \t Acc: {train_acc*100:05.2f}% \t Loss: {train_loss:.3f}')
    prGreen(f'Test =>               \t Acc: {test_acc*100:05.2f}% \t Loss: {test_loss:.3f}')
  
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
file_name = program+".xlsx"    
df.to_excel(file_name, sheet_name= "v1_test", index = False)     

#=============================================================================
#                         Program Completed
#============================================================================= 








    

