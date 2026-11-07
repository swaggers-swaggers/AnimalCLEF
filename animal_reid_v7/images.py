"""Native-coordinate crops; transform masks along with images, not separately."""
import cv2
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T
from settings import DATA, V4, V3


def native(row):
    im = Image.open(DATA / row.path).convert('RGB')
    if row.dataset == 'LynxID2025':
        a = np.asarray(im); yy, xx = np.where(a.max(2) > 12)
        if len(xx):
            box = (max(0, xx.min()-4), max(0, yy.min()-4), min(im.width, xx.max()+5), min(im.height, yy.max()+5))
            if (box[2]-box[0])*(box[3]-box[1]) > im.width*im.height*.25: im = im.crop(box)
    path = V4 / 'masks' / f'{row.image_id}.png'
    if not path.exists(): path = V3 / 'masks' / f'{row.image_id}.png'
    mask = np.array(Image.open(path).convert('L').resize(im.size, Image.Resampling.NEAREST)) > 0 if path.exists() else np.ones((im.height, im.width), bool)
    return im, mask


def body(row):
    im, mask = native(row); yy, xx = np.where(mask)
    if len(xx) and .02 < mask.mean() < .97:
        pad = max(4, round(.04 * max(xx.max()-xx.min(), yy.max()-yy.min())))
        box = (max(0, xx.min()-pad), max(0, yy.min()-pad), min(im.width, xx.max()+pad+1), min(im.height, yy.max()+pad+1))
        im = im.crop(box); mask = mask[box[1]:box[3], box[0]:box[2]]
    return im, mask


def square(im, size=384):
    return ImageOps.pad(im, (size, size), method=Image.Resampling.BICUBIC, color=(124, 116, 104))


NORMALIZE = T.Normalize([.485, .456, .406], [.229, .224, .225])
EVAL = T.Compose([T.ToTensor(), NORMALIZE])
AUGMENT = T.Compose([T.RandomResizedCrop(384, scale=(.75, 1.), ratio=(.85, 1.18)),
                     T.RandomHorizontalFlip(), T.RandomRotation(10, fill=(124, 116, 104)),
                     T.RandomApply([T.ColorJitter(.2, .2, .15, .025)], p=.7),
                     T.RandomGrayscale(p=.2), T.ToTensor(), NORMALIZE,
                     T.RandomErasing(p=.15, scale=(.02, .08))])


class EvaluationImages(Dataset):
    def __init__(self, frame): self.rows = list(frame.itertuples())
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]; im, _ = body(row)
        return int(row.image_id), EVAL(square(im))
