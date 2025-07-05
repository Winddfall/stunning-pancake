from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from PIL import Image
from io import BytesIO
import base64


def batch_psnr(preds, targets):
    psnr = []
    for p, t in zip(preds, targets):
        p_img = p.cpu().numpy().squeeze()
        t_img = t.cpu().numpy().squeeze()
        psnr.append(compare_psnr(t_img, p_img, data_range=1.0))
    return np.mean(psnr)


def batch_ssim(preds, targets):
    ssim = []
    for p, t in zip(preds, targets):
        p_img = p.cpu().numpy().squeeze()
        t_img = t.cpu().numpy().squeeze()
        ssim.append(compare_ssim(t_img, p_img, data_range=1.0, channel_axis=None))
    return np.mean(ssim)


def tensor_to_base64(tensor):
    tensor = tensor.squeeze(0)
    tensor = (tensor * 255).clamp(0, 255).byte()
    img = Image.fromarray(tensor.cpu().numpy(), mode="L")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64


def calculate_score(psnr, ssim_val):
    psnr_score = (min(psnr, 30) - 15) / (30 - 15)
    ssim_score = (ssim_val - 0.6) / (1.0 - 0.6)
    return 4 * psnr_score + 6 * ssim_score
