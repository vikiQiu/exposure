import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def read_img(img_path, size=64):
    img = Image.open(img_path).convert('RGB')
    trans = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop((size, size))
    ]
    )
    img = trans(img)
    # img.show()
    img = np.asarray(img)
    print(img.shape)
    return img


def read_img_torch(img_path, size=64):
    img = Image.open(img_path).convert('RGB')
    trans = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor()
    ]
    )
    img = trans(img)
    return img


def process_exposure(img, p):
    param = np.array([1, 1, 1]) * p
    # img * tf.exp(param[:, None, None, :] * np.log(2))
    # Image.fromarray(img, 'RGB').show()
    img = img * np.exp(param[np.newaxis, np.newaxis, :] * np.log(2))
    img[img > 255] = 255
    img[img < 0] = 0
    img2 = Image.fromarray(np.uint8(img), 'RGB')
    img2.show()

    return img


def process_exposure_torch(img, p):
    param = torch.Tensor([1, 1, 1]) * p
    img = img * torch.exp(param.view(-1, 1, 1) * torch.log(torch.Tensor([2])))
    img[img > 1] = 1
    img[img < 0] = 0
    show_from_torch(img)

    return img


def rgb2lum(image):
    image = 0.27 * image[:, :, 0] + 0.67 * image[:, :, 1] + 0.06 * image[:, :, 2]
    return image[:, :, np.newaxis]


def rgb2lum_torch(image):
    image = 0.27 * image[0, :, :] + 0.67 * image[1, :, :] + 0.06 * image[2, :, :]
    return image.unsqueeze(0)


def lerp(a, b, l):
    return (1 - l) * a + l * b


def process_wnb(img, p):
    luminance = rgb2lum(img)
    param = np.array([1, 1, 1]) * p
    # img * tf.exp(param[:, None, None, :] * np.log(2))
    # Image.fromarray(img, 'RGB').show()
    img = lerp(img, luminance, param[np.newaxis, np.newaxis, :])
    img[img > 255] = 255
    img[img < 0] = 0
    img2 = Image.fromarray(np.uint8(img), 'RGB')
    img2.show()

    return img


def process_wnb_torch(img, p):
    luminance = rgb2lum_torch(img)
    param = torch.Tensor([1, 1, 1]) * p
    img = lerp(img, luminance, param.view(-1, 1, 1))
    img[img > 1] = 1
    img[img < 0] = 0
    show_from_torch(img)

    return img


def show_from_numpy(img):
    Image.fromarray(np.uint8(img), 'RGB').show()


def show_from_torch(img):
    transforms.ToPILImage()(img).show()


def process_contrast(img, p):
    luminance = rgb2lum(img)/255
    contrast_lum = -np.cos(np.pi * luminance) * 0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    param = np.array([1, 1, 1]) * p
    img = lerp(img, contrast_image, param[np.newaxis, np.newaxis, :])
    img[img > 255] = 255
    img[img < 0] = 0
    img2 = Image.fromarray(np.uint8(img), 'RGB')
    img2.show()

    return img


if __name__ == '__main__':
    # img = read_img('models\sample_inputs\jpg\F.jpg', 256)
    # print(rgb2lum(img))
    # process_exposure(img)
    # process_wnb(img, 0.5)
    img = read_img_torch('models\sample_inputs\jpg\F.jpg', 256)
    process_wnb_torch(img, 0.5)
    process_exposure_torch(img, 2)
