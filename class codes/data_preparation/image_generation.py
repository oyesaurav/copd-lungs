import os
import shutil
import glob
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import cv2 as cv
import numpy as np
from data_preparation.file_structuring import FolderStructure
import pydicom
from skimage import exposure
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input


class GenerateImages(FolderStructure):

    def __init__(self, block_size: Tuple[int, int], stride: int, main_dir_path: str):
        """ 
        :param block_size: Tuple[int, int], 
        :param stride: int,
        :param main_dir_path: str
        """
        super().__init__(block_size, stride, main_dir_path)
        
    def preprocess(self):
        print("preprocessing dicom images...")
        workdir = os.listdir(self.work_dir)
        for copd in workdir:
            copdpath = os.path.join(self.work_dir, copd)
            copd_folder = os.listdir(copdpath)
            for patient in copd_folder:
                file_path = os.path.join(copdpath, patient + '/')
                filepath = os.listdir(file_path)
                for dicom in filepath:
                    
                    dicom_data = pydicom.dcmread(filepath + dicom)
                    pixel_array = dicom_data.pixel_array
                    normalized_image = exposure.rescale_intensity(pixel_array, in_range='image', out_range=(0, 1))
                    if len(normalized_image.shape) == 2:
                        normalized_image = np.expand_dims(normalized_image, axis=-1)
                    img_array = image.img_to_array(image.array_to_img(normalized_image).resize(self.block_h,self.block_w))
                    # img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                    img_array = preprocess_input(img_array)
                    # save the image in png
                    cv.imwrite(os.path.join(file_path, dicom + '.png'), img_array)

    def generate_patches(self):
        print('Creating patches...')
        try:
            workdir = os.listdir(self.work_dir)
            for copd in workdir:
                copdpath = os.path.join(self.work_dir, copd)
                copd_folder = os.listdir(copdpath)
                os.chdir(copdpath)
                for patient in copd_folder:
                    # chdir to the patient folder
                    os.chdir(os.path.join(copdpath, patient + '/'))
                    for png in tqdm(glob.glob('*.png')):
                        img = Image.open(png)
                        img_w, img_h = img.size

                        file_name, extension = os.path.splitext(png)

                        save_path = os.path.join(copdpath, patient + '/')

                        frame_num = 0
                        count_row = 0

                        for row in range(0, img_h, self.stride):
                            if img_h - row >= self.block_h:
                                count_row += 1
                                count_col = 0

                                for col in range(0, img_w, self.stride):
                                    if (img_w - col >= self.block_w):
                                        count_col += 1
                                        frame_num += 1

                                        box = (col, row, col + self.block_w, row + self.block_h)
                                        a = img.crop(box)
                                        if self.check_image(a):
                                            a.save(save_path + file_name + '_row_' + str(count_row) + '_col_' + str(count_col) + '.png')

                        img.close()
                        os.remove(png)

        except Exception as e:
            print('Error in Creating_patches() function')
            print(e)
            
    def check_image(self, img):
        """
        :param img: PIL.Image
        :return: bool
        """
        try:
            cropped_img_gray = img.convert("L")
            img_array = np.array(cropped_img_gray)
            variance = np.var(img_array)
            # Check if the variance is less than 1
            if variance < 1:
                return False
            else:
                return True
        except Exception as e:
            print('Error in check_image() function')
            print(e)
            return False

    def main(self):
        self.preprocess()
        self.generate_patches()