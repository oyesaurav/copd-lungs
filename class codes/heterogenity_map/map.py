#  Import libraries and define variables
import tensorflow as tf 
import pickle
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from tqdm import tqdm
import pydicom
from skimage import exposure
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

class Heterogeneity_map():
    def init(self, main_dir_path, block_size, pat_img_path):
        self.main_dir_path = main_dir_path
        self.block_h, self.block_w = block_size
        self.img_path = pat_img_path
        
        self.model = tf.keras.models.load_model(main_dir_path + 'trained_model.h5')
        
    def preprocess(self):
        dicom_data = pydicom.dcmread(self.img_path)
        pixel_array = dicom_data.pixel_array
        normalized_image = exposure.rescale_intensity(pixel_array, in_range='image', out_range=(0, 255)).astype(np.float32)
        normalized_img = normalized_image.astype(np.float32) / 255.0 
        
        return normalized_img
    
    def Generate_matrix(self, img):
        
        map_arr = np.zeros((img.shape[0],img.shape[1]))
        for a in range (100,401):
            for b in range (100,401):
                map_arr[a,b] = self.fun(a,b,img)
        
        #Storing Wiegted Average Matrix of the patient
        pickle_out = open(self.main_dir_path + 'weigted_matrix.pickle', 'wb')
        pickle.dump(map_arr, pickle_out)
        pickle_out.close()

    def window(self, p, q, img):
        W = img[p:p+self.block_h, q:q+self.block_w]
        return W


    def predict(self, Windows):
        count = 0
        X = np.array(Windows).reshape(self.block_h, self.block_w, 1)
        predicted_classes = np.argmax(self.model(X), axis=1)
        for val in predicted_classes:
            if (val == 1):
                count = count + 1
        return count


    def fun(self, a, b, img):
        if (a < 0 or a > 511 or b < 0 or b > 511):
            return 0
        else:
            Windows = []
            for x in range(a - self.block_h - 1, a+1):
                for y in range(b - self.block_w - 1, b+1):
                    Windows.append(self.window(x, y,img))
            count_1 = self.predict(Windows)
        return (count_1/(self.block_h*self.block_w))
    
    def show(self):
        matrix = pickle.load(open(self.main_dir_path + 'weigted_matrix.pickle', 'rb'))

        plt.imshow(matrix, cmap='Reds')
        plt.show()

    def main(self):
        img = self.preprocess()
        self.Generate_matrix(img)
        
        self.show()
        return 