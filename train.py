import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model import CombinedLoss
from tools import batch_psnr, batch_ssim, calculate_score


def train(model, device, num_epochs, learning_rate, train_loader, val_loader):
    criterion = CombinedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    # 修改scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    best_score = -1.0

    print("Starting model training with Attention ResU-Net...")
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for noise, origin in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            noise, origin = noise.to(device), origin.to(device)

            outputs = model(noise)
            loss = criterion(outputs, origin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() # 累加训练损失
        scheduler.step()

        # 验证
        model.eval()
        val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
        with torch.no_grad():
            for noise, origin in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  "
            ):
                noise, origin = noise.to(device), origin.to(device)
                outputs = model(noise)
                # val_loss += criterion(outputs, origin).item()
                val_psnr += batch_psnr(outputs.detach(), origin)
                val_ssim += batch_ssim(outputs.detach(), origin)
        avg_train_loss = train_loss / len(train_loader)
        # avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        # 得分
        val_score = calculate_score(avg_val_psnr, avg_val_ssim)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | "
            f"Val PSNR: {avg_val_psnr:.2f} | Val SSIM: {avg_val_ssim:.4f} | "
            f"Val Score: {val_score:.4f} | LR: {current_lr:.6f}"
        )
        # 存储最好的模型参数
        if val_score > best_score:
            best_score = val_score
            os.makedirs("./checkpoint", exist_ok=True)
            torch.save(model.state_dict(), "./checkpoint/model.pth")
            print(f"🎉 New best model saved with score: {best_score:.4f}")
    print("Training finished. Loading best model for inference...")
    model.load_state_dict(torch.load("./checkpoint/model.pth"))
