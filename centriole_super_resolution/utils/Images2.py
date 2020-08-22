from fastai.vision import *
from skimage import io


class Image3DList(ImageList):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def open(self, fn):
        x = io.imread(str(fn))
        x = pil2tensor(x, np.float32)
        return Image(x)


class ImageImage3DList(Image3DList):
    _label_cls, _square_show, _square_show_res = Image3DList, False, False


def pil2tensor(image,dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    return torch.from_numpy(a.astype(dtype, copy=False))