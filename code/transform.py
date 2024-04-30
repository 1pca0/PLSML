import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms


def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, base_size, ratio_range):
    w, h = img.size
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def cutout(img, label, p=0.5, n_holes=2, length = 48):
    if random.random() < p:
        img = np.array(img)
        label = np.array(label)

        img_d, img_h, img_w = img.shape
        mask = np.ones((img_d, img_h, img_w))

        # size = np.random.uniform(size_min, size_max) * img_h * img_w
        # ratio = np.random.uniform(ratio_1, ratio_2)
        # erase_w = int(np.sqrt(size / ratio))
        # erase_h = int(np.sqrt(size * ratio))
        for n in range(n_holes):
            x = np.random.randint(0, img_h)
            y = np.random.randint(0, img_w)
            z = np.random.randint(0, img_d)

            x1 = np.clip(x - length // 2, 0, img_h)
            x2 = np.clip(x + length // 2, 0, img_h)
            y1 = np.clip(y - length // 2, 0, img_w)
            y2 = np.clip(y + length // 2, 0, img_w)
            z1 = np.clip(z - length // 2, 0, img_d)
            z2 = np.clip(z + length // 2, 0, img_d)

            mask [z1:z2, x1:x2, y1:y2] = 0.
        img = img * mask
        label = label * mask

        # if pixel_level:
        #     value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        # else:
        #     value = np.random.uniform(value_min, value_max)

        # img[y:y + erase_h, x:x + erase_w] = 0
        # mask[y:y + erase_h, x:x + erase_w] = 0
        # img = Image.fromarray(img.astype(np.uint8))
        # mask = Image.fromarray(mask.astype(np.uint8))

    return img, label