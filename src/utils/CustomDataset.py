from torch.utils.data import Dataset, DataLoader
from constants.Constants import SEVERITIES, PATH_ROUTES
import pandas as pd
import os

class CustomDataset(Dataset):
    def __init__(self, train_index, train_size=None, transform=None):
        self.transform = transform

        self.classes = []
        for header in train_index.columns[1:]:
            for severity in SEVERITIES:
                classname = header + '_' + severity
                self.classes.append(classname)

        count = 0
        for index, row in train_index.iterrows():
            if  (count == 0):
                study_id=row[study_id]
                study_path = os.path.join(PATH_ROUTES.TRAIN_IMAGES_FOLDER, study_id)
                for series in os.scandir(study_path):
                    image_series_path = os.path.join(study_path, series)
                    for images in os.scandir(image_series_path):


            count=count+1




            




