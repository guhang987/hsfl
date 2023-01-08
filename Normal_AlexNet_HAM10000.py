#=====================================================
# Centralized (normal) learning: ResNet18 on HAM10000
# Single program
# ====================================================
import os
import torch
from torch import nn
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
program = "Normal Learning AlexNet on HAM10000"
print(f"---------{program}----------")              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#=============================================================================
#                         Data loading 
#=============================================================================    
df = pd.read_csv('data/HAM10000_metadata.csv')
print(df.head())
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    

lesion_type = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# merging both folders of HAM1000 dataset -- part1 and part2 -- into a single directory
imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                for x in glob(os.path.join("data", '*', '*.jpg'))}


df['path'] = df['image_id'].map(imageid_path.get)
df['cell_type'] = df['dx'].map(lesion_type.get)
df['target'] = pd.Categorical(df['cell_type']).codes
print(df['cell_type'].value_counts())
print(df['target'].value_counts())


#==============================================================
# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        
        if self.transform:
            X = self.transform(X)
        
        return X, y


#=============================================================================
# Train-test split    
train, test = train_test_split(df, test_size = 0.2)

train = train.reset_index()
test = test.reset_index()
#=============================================================================
#                         Data preprocessing
#=============================================================================  
# Data preprocessing: Transformation 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                        transforms.RandomVerticalFlip(),
                        transforms.Pad(3),
                        transforms.RandomRotation(10),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
                        ])
    
test_transforms = transforms.Compose([
                        transforms.Pad(3),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
                        ])    


dataset_train = SkinData(train, transform = train_transforms)
dataset_test = SkinData(test, transform = test_transforms)

train_iterator = DataLoader(dataset_train, shuffle = True, batch_size = 256)
test_iterator = DataLoader(dataset_test, batch_size = 256)



print(f'Number of training examples: {len(train)}')
print(f'Number of testing examples: {len(test)}')

for x, y in train_iterator:
    print("shape of x = ", x.shape)
    print(type(x))
    break

#=============================================================================
#                    Model definition: ResNet18
#============================================================================= 

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):   
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  #打包
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] 自动舍去小数点后
            nn.ReLU(inplace=True), #inplace 可以载入更大模型
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27] kernel_num为原论文一半
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            #全链接
            nn.Linear(128, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) #展平   或者view()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  #正态分布赋值
                nn.init.constant_(m.bias, 0)

       
net_glob = AlexNet() # Class labels for HAM10000 = 7 

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
  
    test_loss, test_acc = evaluate(net_glob, device, test_iterator, criterion)
    
    
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








    

