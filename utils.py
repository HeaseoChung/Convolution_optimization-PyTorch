import torch
import numpy as np


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def check_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])

# 전처리 과정 함수
def preprocess(img):
    x = np.array(img).astype(np.float32)
    x = x.transpose([2,0,1])
    x /= 255.
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    return x

# 후처리 과정 함수
def postprocess(tensor):
    x = tensor.mul(255.0).cpu().numpy().squeeze(0)
    x = np.array(x).transpose([1,2,0])
    x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return x

# PSNR 값 계산 함수
def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))