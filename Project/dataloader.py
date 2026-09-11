import pandas as pd
import cv2 as cv
from torch.utils import data
from torch.utils.data import DataLoader

from dependency import *
import torch
import numpy as np
from label_encoding import encode_labels, encode_metadata_labels

from albumentations import (
HorizontalFlip,
VerticalFlip,
Compose,
RandomRotate90,
RandomBrightnessContrast,
ShiftScaleRotate
)

aug= Compose(
    [
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(p=0.5)
    ],p=0.5)

def load_img(pth, shape):
    img= cv.imread(pth)
    img= cv.resize(img, (shape[0], shape[1]))
    return img

class SkinLesionDataset(data.Dataset):
    def __init__(self, img_dir, img_info, file_list, shape, is_test=False, num_class=1, type='clinic'):
        self.is_test= is_test
        self.img_dir= img_dir
        self.img_info = img_info
        self.file_list= file_list
        self.shape= shape
        self.num_class= num_class
        self.total_img_info= img_info
        self.type= type

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id= self.file_list[index]
        sub_img_info= self.total_img_info[file_id:file_id+1] # 'clinic', 'derm'

        clinic_img_pth= sub_img_info['clinic']
        derm_img_pth= sub_img_info['derm']
        clinic_img= load_img(self.img_dir + clinic_img_pth[file_id], self.shape)
        derm_img= load_img(self.img_dir + derm_img_pth, shape=self.shape)

        total_labels= encode_labels(sub_img_info, file_id)
        clinic_img= torch.from_numpy(np.transpose(clinic_img, (2,0,1)).astype('float32')/255) # (H,W,C) to (C,W,H)
        derm_img = torch.from_numpy(np.transpose(derm_img, (2, 0, 1)).astype('float32') / 255)

        meta_data_onehot, meta_data= encode_metadata_labels(sub_img_info, file_id)
        meta_data_onehot= torch.tensor(np.array([meta_data_onehot]))
        meta_data= torch.tensor(np.array(meta_data))

        if self.type=='clinic':
            return clinic_img, [total_labels[0],total_labels[1],total_labels[2],total_labels[3],
                                total_labels[4],total_labels[5],total_labels[6],total_labels[7]]
        elif self.type=='derm':
            return derm_img, [total_labels[0],total_labels[1],total_labels[2],total_labels[3],
                              total_labels[4],total_labels[5],total_labels[6],total_labels[7]]
        else:
            return clinic_img, derm_img, meta_data_onehot, [total_labels[0],total_labels[1],total_labels[2],total_labels[3],
                                                            total_labels[4],total_labels[5],total_labels[6],total_labels[7]]

def dataloader(shape, batch_size, num_workers, data_mode, type):
    test_index= pd.read_csv(test_index_path)
    train_index= pd.read_csv(train_index_path)
    val_index= pd.read_csv(val_index_path)

    test_index_list = list(test_index['indexes'])
    train_index_list= list(train_index['indexes'])
    val_index_list= list(val_index['indexes'])

    train_index_list_1= train_index_list[0:206]

    df= pd.read_csv(img_info_path)
    if data_mode=='self_evaluated':
        data_mode= 'SP'
        train_skin_Dataset= SkinLesionDataset(img_dir=source_dir,img_info=df,file_list=train_index_list_1, shape=shape, type=type)
        train_Dataloader= DataLoader(dataset=train_skin_Dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

        val_skin_Dataset= SkinLesionDataset(img_dir=source_dir,img_info=df,file_list=val_index_list, shape=shape, is_test=True, type=type)
        val_Dataloader= DataLoader(dataset=val_skin_Dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

    else:
        train_skin_Dataset = SkinLesionDataset(img_dir=source_dir, img_info=df, file_list=train_index_list, shape=shape, type=type)
        train_Dataloader = DataLoader(dataset=train_skin_Dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

        val_skin_Dataset = SkinLesionDataset(img_dir=source_dir, img_info=df, file_list=val_index_list, shape=shape, is_test=True, type=type)
        val_Dataloader = DataLoader(dataset=val_skin_Dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

    return train_Dataloader, val_Dataloader
