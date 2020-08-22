import json as js
import matplotlib.pyplot as plt
import numpy as np
import os
import paths_definitions as pth

folder = pth.metrics_folder
#folder = 'C:/Users/Thibaut/Documents/_Stage_Image_Super_Resolution/PSSR-master/losses_and_val_losses'

def open_json(file_path):

    with open(file_path) as json_file:
        data = js.load(json_file)
        data = np.array(data).astype(np.float32)
    return data

def open_js(file_name):
    return open_json(f'{folder}/{file_name}')


def plot_loss(model_name):
    """plot losses, validation losses and metrics SSIM and PSNR and save figures. model_name is the name of the model
    without extension"""
    def add_legend_etc(y_name):
        plt.xlabel("epochs")
        plt.ylabel(y_name)
        plt.xticks(range(1, nb_epoch + 1, 5))
        plt.yscale('log')
        plt.legend()
        plt.title(model_name)

    val_losses_json_file = f'val_losses_{model_name}'
    train_losses_json_file = f'losses_{model_name}'
    losses = open_js(train_losses_json_file)
    val_losses = open_js(val_losses_json_file)
    print(losses[-1])
    nb_epoch = len(val_losses)
    losses_without_first_epoch = losses[len(losses) // nb_epoch:]
    plt.plot(range(1, nb_epoch + 1), val_losses, label='validation losses')
    plt.plot(np.linspace(1, nb_epoch, len(losses_without_first_epoch)), losses_without_first_epoch,
             label='train losses (without first epoch)')

    add_legend_etc("loss validation and train (log scale)")


    test_results_path = f'{pth.test_results_dir}/{model_name}'
    plt.savefig(f'{test_results_path}/losses')
    plt.close()

    for metric in ['psnr', 'ssim']:
        json = f'{metric}_{model_name}'
        plt.plot(range(1, nb_epoch + 1), open_js(json), label=metric)
        add_legend_etc(f'{metric} (log scale)')
        plt.savefig(f'{test_results_path}/{metric}')
        plt.close()



def plot_metric(txt_file):
    #receive the txt file from topaz which computes the different metrics, plot them and save the figure
    def split_line(line):
        return line[:-1].split('\t')

    with open(txt_file) as f:
        lines = f.readlines()


    label_line = split_line(lines[0])
    lines = [split_line(line) for line in lines[1:]]
    nb_epochs = [lines[k][2] for k in range(len(lines))].count('test')
    for i, metric in enumerate(label_line[3:]):

        i += 3
        print(metric)
        if metric != 'auprc':
            train_values = [float(lines[k][i]) for k in range(len(lines)) if lines[k][2] == 'train']
            plt.plot(np.linspace(0, nb_epochs, len(train_values)), train_values,
                 label=f'train {metric}')
        if metric != 'ge_penalty':
            valid_values = [float(lines[k][i]) for k in range(len(lines)) if lines[k][2] == 'test']
            plt.plot(range(1, nb_epochs + 1), np.array(valid_values), label=f'valid {metric}')
        #plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylabel(metric)
        plt.xlabel('epochs')
        plt.legend()
        plt.savefig(f'{metric}.png')
        plt.show()

def intenity_graph(image, image_name, save_figure=False):
    """receive a 3D image (3D array) and plot the intensity graph with respect to the depth dimension if save_figure
    returns local minimums of this graph"""
    depth = image.shape[0]

    intensities = [np.sum(image[i]) for i in range(depth)]
    ma = np.max(intensities)
    intensities = [I/ma for I in intensities]
    if save_figure:
        plt.plot(range(1, depth + 1), intensities)
        plt.title(f'intensity of {image_name} with respect to depth dimension')
        plt.xticks(range(1, depth + 1))
        plt.ylabel('intensity (sum of the pixels)')
        plt.xlabel('slices')
        plt.grid()
        loc = f'{pth.myHome}/intensity_graph_{image_name}.png'
        plt.savefig(loc)
        plt.close()

    max_locs = []
    min_locs = []
    ma = -10**10
    mi = 10 ** 10
    argma = 0
    argmi = 0
    for i in range(1,depth-1):
        diff1 = intensities[i] - intensities[i-1]
        diff2 = intensities[i+1] - intensities[i]
        if diff1 < 0 and diff2 > 0:
            min_locs.append(i)
        if diff1 > 0 and diff2 < 0:
            max_locs.append(i)
        if intensities[i] > ma:
            argma = i
            ma = intensities[i]
        if intensities[i] < mi:
            argmi = i
            mi = intensities[i]

    return min_locs, max_locs, argmi, argma















if __name__ == "__main__":
    #model_name = 'wnresnet_sig6_l1_loss_312'
    #plot_loss(model_name)
    txt_file = 'C:/Users/Thibaut/Documents/_Stage_Image_Super_Resolution/data/particle_detection/model_training.txt'
    plot_metric(txt_file)












