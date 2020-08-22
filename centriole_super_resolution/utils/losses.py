import torch
import warnings
from torch import nn
from torchvision import models
from functools import partial


def l_alpha(input, target, alpha):

    ret = torch.abs(input - target)**alpha
    ret = torch.mean(ret)
    return ret

class L_alpha(nn.Module):
    def __init__(self, alpha=1):
        super(L_alpha, self).__init__()
        print('BJR2')
        self.alpha = alpha

    def forward(self, input, target):
        return l_alpha(input, target, self.alpha)

    def __str__(self):
        return f'L{self.alpha}_loss'


L1 = partial(L_alpha, alpha=1)



class VGGFeatureExtractor(nn.Module):
    def __init__(self, device=0, feature_layer=34):
        super(VGGFeatureExtractor, self).__init__()
        model = models.vgg19(pretrained=True)
        model.to(device)
        self.features=(nn.Sequential(*list(model.features.children())[:(feature_layer + 1)]))
        # No need to BP to variable
        #for k, v in self.features.named_parameters():
            #v.requires_grad = False

    def forward(self, x):
        x = torch.cat((x, x, x), 1).cuda()
        return self.features(x)


class Content_loss(nn.Module):
    def __init__(self, alpha=1, nb_layer_vgg=34):
        super(Content_loss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=nb_layer_vgg)
        self.l_loss = L_alpha(alpha)

    def forward(self, input, target):
        ft_input = self.vgg(input)
        ft_target = self.vgg(target)
        return self.l_loss(ft_input, ft_target)

class Mix_content_l_loss(nn.Module):
    def __init__(self, prop=0.2, alpha=1, betha=1, nb_layer_vgg=34):
        super(Mix_content_l_loss, self).__init__()
        self.content_loss = Content_loss(betha, nb_layer_vgg)
        self.l_loss = L_alpha(alpha)
        self.alpha = alpha
        self.betha = betha
        self.prop = prop
        self.nb = nb_layer_vgg

    def forward(self, input, target):
        return self.prop * self.content_loss(input, target) + (1-self.prop) * self.l_loss(input, target)

    def __str__(self):
        return f'mix content L{self.betha} loss and L{self.alpha} with proportion of content loss {self.prop}. {self.nb} layer'





