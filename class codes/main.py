from data_preparation.data_sorting import DataPreparation
from data_preparation.image_generation import GenerateImages
from train_model.training import Trainer
from heterogenity_map.map import Heterogeneity_map


block_size = (64,64)
stride = 2
inter_dim = (64,64)
main_dir_path = "E:/ML projects/copd lungs/"
test_pat_img_path = 


GenerateImages_obj = GenerateImages(block_size, stride, main_dir_path)
Trainer_obj = Trainer(main_dir_path, inter_dim)
MapObj = Heterogeneity_map(main_dir_path,stride, block_size, test_pat_img_path)


GenerateImages_obj.main()
Trainer_obj.main()
MapObj.main()