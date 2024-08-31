import os
import shutil
import glob
from tqdm import tqdm
from PIL import Image
import nibabel as nib
import numpy as np
import pandas as pd
import pickle as pkl
import cv2 as cv
import random as rn
from multiprocessing import Pool,Process
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU, Activation, AveragePooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from keras.layers.convolutional import Conv2D
import tensorflow as tf
from tensorflow.keras.metrics import TruePositives,TrueNegatives,FalsePositives,FalseNegatives,AUC,Recall,Precision
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,TensorBoard
from tensorflow.keras.regularizers import l2
import time
import shutil
import sys

patients = 150
modalities = ['T1']
classifications = ['MGMT_positive', 'MGMT_negative']
block_size = (64,64)
stride = 2
inter_dim = (110, 90)
dataset_path = "D:/MGMT research project/NIfTI-files/"
main_dir_path = "D:/mgmt-patch-model/"

class Trainer():
    def __init__(self, main_dir_path, min_size) -> None:
        self.block_h, self.block_w = min_size
        self.work_dir = main_dir_path + 'Data/'

    def interpolation(self, df):
        the_y = []
        the_x = []
        for item in tqdm(range(len(df))):
            img = cv.imread(df['pat_path'].iloc[item],cv.IMREAD_GRAYSCALE)
            if img.shape[0]<self.block_h or img.shape[1]<self.block_w: continue
            # img = cv.resize(img,self.inter_dim)
            normalized_img = img.astype(np.float32) / 255.0  # Scaling pixel values to [0, 1]
            the_x.append(normalized_img)
            the_y.append(df['copd'].iloc[item])

        return the_x, the_y
    
    def spliting(self):
        pat_dict = {
            'pat_path':[],
            'copd':[],
        }
        workdir = os.listdir(self.work_dir)
        for copd in self.workdir:
            copdpath = os.path.join(self.work_dir, copd)
            copd_folder = os.listdir(copdpath)
            for patient in copd_folder:
                file_path = os.path.join(copdpath, patient + '/')
                filepath = os.listdir(file_path)
                for dicom in filepath:
                    pat_dict['pat_path'].append(os.path.join(file_path, dicom))
                    pat_dict['copd'].append(1 if copd == 'copd' else 0)

        pat_df = pd.DataFrame(pat_dict)
        # Splitting Data into train and test
        train_df,test_df=train_test_split(pat_df[['pat_path','copd']], stratify=pat_df['copd'], random_state=57, test_size=0.2)
        print(f'Shape of train_data {train_df.shape}')
        print(f'Shape of test_data {test_df.shape}')

        del pat_df, pat_dict
        
        return train_df, test_df
    
    def model(self):
        model = Sequential()

        # model.add(Conv2D(16, (5,5), padding='same',input_shape=(90,110,1),kernel_initializer=HeNormal()))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        model.add(Conv2D(64, (5, 5), padding='same',input_shape=(self.block_h,self.block_w,1),
                         kernel_initializer=HeNormal(),kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.1))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.1))

        # model.add(Conv2D(8, (3, 3), padding='same',kernel_initializer=HeNormal()))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.))

        model.add(Conv2D(32, (3, 3), padding='same',kernel_initializer=HeNormal(),kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.1))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.1))
        
        model.add(Conv2D(16, (3, 3), padding='same',kernel_initializer=HeNormal(),kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(8, (3, 3), padding='same',kernel_initializer=HeNormal(),kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        # model.add(Dropout(0.1))


        # model.add(Conv2D(48, (3, 3), padding='same'))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.1))

        model.add(Flatten())  # Convert 3D feature map to 1D feature vector.

        # model.add(Dense(10,kernel_initializer=HeNormal()))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.3))

        # model.add(Dense(10,kernel_initializer=HeNormal()))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.2))
        
        # model.add(Dense(10,kernel_initializer=HeNormal()))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(Dropout(0.1))

        model.add(Dense(10,kernel_initializer=HeNormal(),kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        # model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                    optimizer='adam', 
                    metrics=['accuracy',TruePositives(),
                             TrueNegatives(),FalsePositives(),
                             FalseNegatives(),AUC(),Recall(),Precision()])
        return model

    def train_model(self, model, train_x, train_y, test_x, test_y):
        model.fit(np.array(train_x),np.array(train_y), 
            validation_data=(np.array(test_x),np.array(test_y)),
            batch_size= 32,
            epochs = 10,
            shuffle= True
        )
        
        

    def main(self):
        train_df, test_df = self.spliting()
        train_x, train_y = self.interpolation(train_df)
        test_x, test_y = self.interpolation(test_df)
        model = self.model()
        self.train_model(model, train_x, train_y, test_x, test_y)

if __name__ == "__main__":
    Trainer_obj = Trainer(modalities, classifications, main_dir_path, (30,30), (64,64))
    Trainer_obj.main()