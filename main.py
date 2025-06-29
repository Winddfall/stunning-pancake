import os
import torch
import pandas as pd
from train import train
from inference import inference
from dataset import dataloader
from model import UNet
from visual import visual_dataset, visual_results

batch_size = 128
learning_rate = 0.001
num_epochs = 100
cuda = 1
device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_loader, val_loader, test_loader = dataloader(batch_size)
    print("Saving dataset examples visualization to 'dataset_examples.png'...")
    visual_dataset(train_loader)
    print("Starting model training...")
    model = UNet().to(device)
    # 训练模型
    train(model, device, num_epochs, learning_rate, train_loader, val_loader)
    # 验证模型
    model.eval()
    example_noisy, example_output, submission_data = inference(
        model,
        device,
        test_loader,
        batch_size,
    )
    os.makedirs("result", exist_ok=True)
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv("./result/submission.csv", index=False)
    print("submission.csv saved successfully.")
    print("Saving denoising results visualization to 'denoising_results.png'...")
    visual_results(example_noisy, example_output)
    print("Script execution complete.")
