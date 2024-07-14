from torch.utils.data import Dataset, DataLoader
from constants.Constants import SEVERITIES, PATH_ROUTES
import pandas as pd
import os
import numpy as np
import torch
from PIL import Image
import pydicom

class CustomDataset(Dataset):
    def __init__(self, train_index, train_size=None, transform=None):
        self.transform = transform

        self.classes = []
        for header in train_index.columns[1:]:
            for severity in SEVERITIES:
                classname = header + '_' + severity
                self.classes.append(classname)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.labels = []
        self.file_paths = []
        
        sampled_train_index = train_index
        if (train_size != None):
            sampled_train_index = sampled_train_index.sample(n=train_size)

        
        for index, row in sampled_train_index.iterrows():
            study_id=row['study_id']
            study_path = os.path.join(PATH_ROUTES.TRAIN_IMAGES_FOLDER, str(study_id))
            label = self.createLabelFromDataRow(train_index, row, self.class_to_idx)
            for series in os.scandir(study_path):
                image_series_path = os.path.join(study_path, series)
                for images in os.scandir(image_series_path):
                    self.file_paths.append(os.path.join(image_series_path, images))
                    self.labels.append(label)
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        img_path = self.file_paths[index]
        label = self.labels[index]

        dicom_data = pydicom.dcmread(img_path)
        image_data = dicom_data.pixel_array
        image_data = (image_data / np.max(image_data) * 255).astype(np.uint8)
        image = Image.fromarray(image_data)

        if (self.transform):
            image = self.transform(image)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image, label_tensor
    


    def createLabelFromDataRow(self, train_index, row, class_to_idx):
        label = np.zeros(len(class_to_idx), dtype=int)
        for header in train_index.columns[1:]:
            if (not pd.isna(row[header])):
                classname = header + '_' + row[header]
                label[class_to_idx[classname]] = 1
        return label




            




