import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os


data = pd.read_csv("/home/saibalaji/Music/dataset/test.csv")
vals = data['image_ID'].tolist()
# Paths for image directory and model
IMDIR="test/"
MODEL='model.pth'

# Load the model for testing
model = torch.load(MODEL,"cpu")
model.eval()

# Class labels for prediction
class_names=['Badminton','Cricket','Karate','Soccer','Swimming','Tennis','Wrestling']

# Retreive 9 random images from directory
files=os.listdir(IMDIR)
#print(files)

# Configure plots
fig = plt.figure(figsize=(9,9))
rows,cols = 3,3

# Preprocessing transformations
preprocess=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Perform prediction and write in csv file
result={}
image_id=[]
ground=[]        
with torch.no_grad():
    for i in vals:
        if i in files:
            i='/home/saibalaji/Music/dataset/test'+'/'+i
            name = i.split('/')[-1]
            img=Image.open(i).convert('RGB')
            
            inputs=preprocess(img).unsqueeze(0).to(device)
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            image_id.append(name)

        
            label=class_names[preds]
            ground.append(label)
            result[name]=label

df = pd.DataFrame(
    {'image_ID': image_id,
     'label': ground
    })
print(df)
df.to_csv("test.csv",index=False)

