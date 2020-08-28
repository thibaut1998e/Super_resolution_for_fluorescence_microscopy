
import yaml
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.vision.models.unet import DynamicUnet
from fastai.vision.models import resnet18, resnet34, resnet50
from skimage.util import random_noise
from skimage import filters
from utils import *
from utils.resnet import *
import skimage
import PIL
from utils.Images2 import *
import paths_definitions as pth
import save_in_json as js
from utils.losses import L1
from utils.losses import Content_loss
from utils.losses import L_alpha
from utils.losses import Mix_content_l_loss


torch.backends.cudnn.benchmark = True

def get_src(x_data, y_data, mode='L', p=1):
    #p = proportion of training set used
    def map_to_hr(x):
        return y_data/x.relative_to(x_data)


    src = (ImageImageList
               .from_folder(x_data, convert_mode=mode).filter_by_rand(p=p)
               .split_by_folder()
                .label_from_func(map_to_hr, convert_mode=mode))

    #ImageImageList.filter_by_rand()
    print(src)


    return src


def get_data(bs, size, x_data, y_data,
             max_rotate=10.,
             min_zoom=1., max_zoom=1.1,
             use_cutout=False,
             use_noise=True,
             scale=2,
             xtra_tfms=None,
             gauss_sigma=(0.4,0.7),
             pscale=(5,30),
             mode='L',
             p=1,
             **kwargs):
    src = get_src(x_data, y_data,  mode=mode, p=p)

    print('cutout', use_cutout)
    x_tfms, y_tfms = get_xy_transforms(
                          max_rotate=max_rotate,
                          min_zoom=min_zoom, max_zoom=max_zoom,
                          use_cutout=use_cutout,
                          use_noise=use_noise,
                          gauss_sigma=gauss_sigma,
                          pscale=pscale,
                          xtra_tfms = xtra_tfms)
    x_size = size // scale


    data = (src
            .transform(x_tfms, size=x_size)
            .transform_y(y_tfms, size=size)
            .databunch(bs=bs, **kwargs))
    print("data", data)
    data.c = 3
    return data



def get_val_losses(learn):
    rec = learn.recorder
    return np.array(rec._split_list_val(rec.val_losses, 0, 0))

def get_losses(learn):
    rec = learn.recorder
    return np.array(rec._split_list(rec.losses, 0, 0))

def get_ssim_psnr(learn):
    rec = learn.recorder
    met = rec.metrics
    ssim = np.array(np.float32([met[i][2] for i in range(len(met))]))
    psnr = np.array(np.float32([met[i][1] for i in range(len(met))]))
    return ssim, psnr




def find_lr(learn):
    learn.lr_find()
    rec = learn.recorder
    lrs = rec._split_list(rec.lrs, 0, 0)
    losses = rec._split_list(rec.losses, 0, 0)
    best_loss = 100
    best_lr = 10**-4
    for i, loss in enumerate(losses):
        if loss < best_loss:
            best_loss = loss
            best_lr = lrs[i]
    return best_lr




@call_parse
def main(
        model: Param("architecture used, among : rrdb,rrdb2,wnresnet,srfbn,rcan,rdn", str) = 'wnresnet',
        gpu: Param("GPU to run on", str)=None,
        bs: Param("batch size per gpu", int) = 8,
        lr: Param("learning rate", float) = 1e-4,
        lr_start: Param("learning rate start", float) = None,
        noise: Param("add dynamic crappifier", action='store_true') = False,
        size: Param("img size", int) = 312,
        epochs: Param("num cyles", int) = 5,
        save_name: Param("model save name", str) = 'learn',
        feat_loss: Param('feat_loss', action='store_true')=False,
        mode: Param('image mode like L or RGB', str)='L',
        debug: Param('debug mode', action='store_true')=False,
        scale:Param("scaling factor", int) = 2,
        hr_folder: Param('HR images', str) = '',
        lr_folder: Param('corresponding LR images', str) = '',
        max_rotate : Param('maximum rotation in data augmantation', float) = 10,
        raw: Param('train_on_raw', action='store_true')=False,
        alpha: Param('loss power', float) = 1,
        betha: Param('feat loss power', float) = 1,
        prop: Param('proportion of feat loss (if used)', float) = 0.2,
        max_zoom: Param('max zoom in data augmentation', float) = 4.,
        nb_layer_vgg:Param("nb layer loss feature extractor", int) = 34,
        p:Param("proportion of training set used", float) = 1

):
    #hyper-parameters of architectures
    wnres_args = {
        'blur': True,
        'blur_final': True,
        'bottle': True,
        'self_attention': True,
        'last_cross': True,
        'scale': scale,
        'arch': eval('wnresnet34')
    }
    rrdb_args = {
        'nf': 8,
        'nb': 8,
        'gcval': 8,
        'upscale': scale
    }
    rdn_args = {'scale_factor': scale,
                'num_channels': 1,
                'num_features': 64,
                'growth_rate': 64,
                'num_blocks': 16,
                'num_layers': 8}
    rcan_args = {}
    srfbn_args = {}
    learner_args = {'wnresnet': wnres_args, 'rrdb': rrdb_args, 'rdn': rdn_args, 'rrdb2': rrdb_args,
                    'rcan': rcan_args, 'srfbn': srfbn_args}
    args = learner_args[model]
    learner_funcs = {'wnresnet': wnres_unet_learner,
                     'rrdb': rrdb_learner,
                     'rdn': rdn_learner,
                     'rrdb2': rrdb_learner2,
                     'rcan': rcan_learner,
                     'srfbn': SRFBN_learner}
    learner_func = learner_funcs[model] #function with generate the model

    model_dir = 'models'

    if not debug:
        gpu = setup_distrib(gpu)
        print('on gpu: ', gpu)
        n_gpus = num_distrib()
    else:
        print('debug mode')
        gpu = 0
        n_gpus = 0


    if feat_loss:
        #loss which combines L_alpha loss and feat_betha loss, proportion of feature loss used = prop
        #nb_layer_vgg = number of layers used to extract feature
        loss = Mix_content_l_loss(alpha=alpha, betha=betha, prop=prop, nb_layer_vgg=nb_layer_vgg)

    else:

        loss = L_alpha(alpha)


    print('loss: ', loss)
    metrics = sr_metrics #metrics used to evaluate the performance : PSNR, SSIM, norm-PSNR, norm-SSIM

    bs = max(bs, bs * n_gpus)
    size = size

    print('bs:', bs, 'size: ', size, 'ngpu:', n_gpus)
    data = get_data(bs, size, lr_folder, hr_folder,  max_zoom=max_zoom,
                    use_noise=noise, mode=mode, scale = scale, max_rotate=max_rotate, p=p)

    callback_fns = []
    if gpu == 0 or gpu is None:
        callback_fns.append(partial(SaveModelCallback, name=f'{save_name}_best'))

    learn = learner_func(data, args, path=Path('.'),
                             loss_func=loss, metrics=metrics, model_dir=model_dir, callback_fns=callback_fns, wd=1e-3)


    gc.collect()

    if not debug:
        if gpu is None: learn.model = nn.DataParallel(learn.model)
        else: learn.to_distributed(gpu)


    learn = learn.to_fp16(loss_scale=1)
    #print(learn.model)
    #print(learn.summary())

    if not lr_start is None: lr = slice(lr_start, lr)
    else: lr = slice(None, lr, None)
    print('lr:', lr)
    learn.fit_one_cycle(epochs, lr)



    if (gpu == 0 or gpu is None) and save_name != pth.do_not_save:
        save_location = f'{pth.models}/{save_name}.pkl'
        learn.export(save_location)
        print(f'exported in location {save_location}')

    #save ssim, psnr losses and val losses in a json file
    ssim, psnr = get_ssim_psnr(learn)
    print('ssim', ssim)
    print('psnr', psnr)

    losses = get_losses(learn)
    val_losses = get_val_losses(learn)
    js.save_in_json(losses, f'{pth.metrics_folder}/losses_{save_name}')
    js.save_in_json(val_losses, f'{pth.metrics_folder}/val_losses_{save_name}')
    js.save_in_json(ssim, f'{pth.metrics_folder}/ssim_{save_name}')
    js.save_in_json(psnr, f'{pth.metrics_folder}/psnr_{save_name}')
    #os.remove(f'{model_dir}/{save_name}_best_{size}.pth')
