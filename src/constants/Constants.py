import os

class PATH_ROUTES:
    RELATIVE_PATH = ".."
    PROJECT_ABSOLUTE_PATH = os.path.abspath(RELATIVE_PATH)
    TRAIN_CSV_INDEX_PATH = os.path.join(PROJECT_ABSOLUTE_PATH, 'data', 'train.csv')
    TRAIN_IMAGES_FOLDER = os.path.join(PROJECT_ABSOLUTE_PATH, 'data','train_images')


SEVERITIES = ['Normal/Mild','Moderate', 'Severe']
    