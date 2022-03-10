from PIL import Image
import os 
from pathlib import Path
import shutil
import cv2
import numpy as np
import glob

src="dataset/train/"
folders = os.listdir(src)
print(folders)
trg = "/home/saibalaji/Music/dataset/New_data/"
count=0
for file in glob.glob(src):
    name = file.split("/")[-1]
    print(name)
    name = name.split(".")
    count+=1
    image = Image.open(file)
    img = image.convert('RGB')
    img.save(trg+"/"+name[0]+".jpg")
print(count)
