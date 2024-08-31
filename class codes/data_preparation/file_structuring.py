import os
import shutil
import glob
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
import nibabel as nib
import numpy as np


class FolderStructure:

    def __init__(self,
                 block_size: Tuple[int, int],
                 stride: int,
                 main_dir_path: str):
        """
        :param block_size: Tuple[int, int],
        :param stride: int,
        :param main_dir_path: str
        """
        self.block_h, self.block_w = block_size
        self.stride = stride
        self.work_dir = self.main_dir_path + 'Data/'

    def create_modality_folders(self):
        """
        Create the modality folders needed for the training.
        :return:
        """
        print("Copying files into the corresponding modality folders...")
        train_folder_1 = os.listdir(self.work_dir)
        for pos_neg in tqdm(train_folder_1):
            patient_folders = os.listdir(os.path.join(self.work_dir, pos_neg))
            for patient_folder in patient_folders:
                for modality in self.modalities:
                    modality_folder_path = os.path.join(self.work_dir, pos_neg, modality)
                    # print(modality_folder_path)
                    modality_patient_folder_path = os.path.join(modality_folder_path, patient_folder)
                    # print(modality_patient_folder_path)
                    if not os.path.exists(modality_patient_folder_path):
                        os.makedirs(modality_patient_folder_path)

                    modality_file_path = os.path.join(self.work_dir, pos_neg, patient_folder,
                                                      '{}_{}.nii.gz'.format(patient_folder, modality))
                    # print(modality_file_path)
                    seg_file_path = os.path.join(self.work_dir, pos_neg, patient_folder,
                                                 '{}_seg.nii.gz'.format(patient_folder))
                    # print(seg_file_path)
                    if os.path.exists(modality_file_path) and os.path.exists(seg_file_path):
                        shutil.copy(modality_file_path, modality_patient_folder_path)
                        shutil.copy(seg_file_path, modality_patient_folder_path)
                    else:
                        print("Either {} or {} does not exist".format(modality_file_path, seg_file_path))
                # delete patient folder after copying files
                shutil.rmtree(os.path.join(self.work_dir, pos_neg, patient_folder))

        print("Modality folders done.")