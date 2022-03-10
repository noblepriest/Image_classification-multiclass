import numpy as np
import pandas as pd
import os
import shutil
from pathlib import Path

df = pd.read_csv("/home/saibalaji/pytorch-image-classification/test.csv")

store = df[df['label']=='Wrestling']
sports=store.head(50)

vals = sports['image_ID'].tolist()
new_files=[]
for i in vals:
    i = '/home/saibalaji/pytorch-image-classification/test'+'/'+i
    #print(i)
    new_files.append(i)
src = '/home/saibalaji/pytorch-image-classification/test'
trg = '/home/saibalaji/pytorch-image-classification/eval_ds/6'
 
files=os.listdir(src)
 
# iterating over all the files in
# the source directory
for fname in new_files:
     
    # copying the files to the
    # destination directory
    shutil.copy2(os.path.join(src,fname), trg)