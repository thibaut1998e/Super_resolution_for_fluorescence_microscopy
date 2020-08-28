import numpy as np

"""constant string variables used through the project"""

"""path constant : modify accordingly"""
myHome = "/home/eloy"
metrics_folder = f'{myHome}/metrics_record' #json file containing losses and metrics are saved here
training_sets = myHome + "/training_sets"
models = myHome + "/models" #location in which models architecture are saved
test_results_dir = myHome + "/test_results" #results are saved in this folder
train_code_location = myHome + "/PSSR-master2/train.py" #location of the python file train.py


"""some other constants"""

do_not_save = "do_not_save" #if this model name is given in the train argument parameters the model trained is not saved



