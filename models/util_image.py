import sys
import cv2
import torch
import numpy as np
from einops import rearrange


def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def imread(path, chn='rgb', dtype='float32'):
    '''
    Read image.
    chn: 'rgb', 'bgr' or 'gray'
    out:
        im: h x w x c, numpy tensor
    '''
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # BGR, uint8
    try:
        if chn.lower() == 'rgb':
            if im.ndim == 3:
                im = bgr2rgb(im)
            else:
                im = np.stack((im, im, im), axis=2)
        elif chn.lower() == 'gray':
            assert im.ndim == 2
    except:
        print(str(path))

    if dtype == 'float32':
        im = im.astype(np.float32) / 255.
    elif dtype == 'float64':
        im = im.astype(np.float64) / 255.
    elif dtype == 'uint8':
        pass
    else:
        sys.exit('Please input corrected dtype: float32, float64 or uint8!')

    return im


def img2tensor(imgs, out_type=torch.float32):
    """Convert image numpy arrays into torch tensor.
    Args:
        imgs (Array or list[array]): Accept shapes:
            3) list of numpy arrays
            1) 3D numpy array of shape (H x W x 3/1);
            2) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.

    Returns:
        (array or list): 4D ndarray of shape (1 x C x H x W)
    """

    def _img2tensor(img):
        if img.ndim == 2:
            tensor = torch.from_numpy(img[None, None,]).type(out_type)
        elif img.ndim == 3:
            tensor = torch.from_numpy(rearrange(img, 'h w c -> c h w')).type(out_type).unsqueeze(0)
        else:
            raise TypeError(f'2D or 3D numpy array expected, got{img.ndim}D array')
        return tensor

    if not (isinstance(imgs, np.ndarray) or (isinstance(imgs, list) and all(isinstance(t, np.ndarray) for t in imgs))):
        raise TypeError(f'Numpy array or list of numpy array expected, got {type(imgs)}')

    flag_numpy = isinstance(imgs, np.ndarray)
    if flag_numpy:
        imgs = [imgs,]
    result = []
    for _img in imgs:
        result.append(_img2tensor(_img))

    if len(result) == 1 and flag_numpy:
        result = result[0]
    return result