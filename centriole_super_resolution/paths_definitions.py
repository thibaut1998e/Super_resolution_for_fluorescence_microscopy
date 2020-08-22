import numpy as np

myHome = "/home/eloy"
metrics_folder = f'{myHome}/metrics_record' #json file containing losses and metrics are saved here
training_sets = myHome + "/training_sets"
models = myHome + "/models" #location in which models architecture are saved
test_results_dir = myHome + "/test_results" #results are saved in this folder
relative_path_type_channel = "deconv/c2"
relative_path_raw = 'raw/c2'
train_code_location = myHome + "/PSSR-master2/train.py"
test_code_location = myHome + "/PSSR-master2/inference.py"
wide_field_3D = f'{myHome}/wide_field/wide_field_resized' #wide field images used for testing the model

LR = "LR"
HR = "HR"
HR_bilinear = "HR_bilinear"
HR_predicted = "HR_predicted"
test_on_training = "test_on_training"
test_on_wide_field = "test_on_wide_field"

do_not_save = "do_not_save" #if this model name is given in the train argument parameters the model trained is not saved



