from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import torch
import h5py
import numpy as np
import os, sys

#need flexibility to update hdf hierarchy
def hdf_2_array(filename):
    hdf = h5py.File(filename,'r')
    data = hdf['MDF']['images']['0']['image']
    arr = np.array(data)
    return arr

class ImgDataset(Dataset):
    """CryoEM Img dataset."""

    def __init__(self, image_path, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file filename pairs.
            root_dir (string): Directory with all the images.
        """
        self.image_path = image_path
        self.csv_map = csv_file
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path,self.df.iloc[idx,]['image'])
        target_path = os.path.join(self.image_path,self.df.iloc[idx,]['target'])
        image = hdf_2_array(img_path)
        target = hdf_2_array(target_path)
        image = torch.from_numpy(image)
        target= torch.from_numpy(target)
        #need revisit 
        image = torch.reshape(image, [1,960,960])
        target = torch.reshape(target, [1,960,960])
        return image, target

    
def get_loader(data_dir, csv_file, split, batch_size,
               shuffle, num_workers,max_num_samples=-1):

    dataset = ImgDataset(image_path=data_dir, csv_file=csv_file)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader, dataset