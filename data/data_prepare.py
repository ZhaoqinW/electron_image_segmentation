import pandas as pd
import h5py
import numpy as np
import os, sys
from PIL import Image


file_dir = './image_toy'

suffix = '_image.hdf'
file_list = os.listdir(file_dir)
truth_suffix = '_densegranule.hdf'

whole_images = [f for f in file_list if f.endswith(suffix)]
record = []
for f in whole_images:
    prefix = f.split(suffix)[0]
    truth = prefix+truth_suffix
    record.append([f,truth])

data = pd.DataFrame(record)
data.columns = ['image','target']
data.to_csv('image_pairs.csv',index=False)