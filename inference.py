import torch
from tools import tensor_to_base64
from tqdm import tqdm
import torchvision.transforms.functional as TF


def inference(model, device, test_loader, batch_size):
    submission_data = []
    example_noisy, example_output = None, None
    with torch.no_grad():
        for batch_idx, noisy in enumerate(
            tqdm(test_loader, desc="Generating submission.csv")
        ):
            noisy = noisy.to(device)
            denoised_original = model(noisy)
            noisy_flipped = TF.hflip(noisy)
            denoised_flipped = model(noisy_flipped)
            denoised_flipped_restored = TF.hflip(denoised_flipped)
            denoised = (denoised_original + denoised_flipped_restored) / 2.0
            if batch_idx == 0:
                example_noisy = noisy.cpu()
                example_output = denoised.cpu()
            for i in range(denoised.shape[0]):
                sample_id = batch_idx * batch_size + i
                denoised_base64 = tensor_to_base64(denoised[i])
                submission_data.append(
                    {"id": sample_id, "denoised_base64": denoised_base64}
                )
    return example_noisy, example_output, submission_data
