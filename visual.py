import matplotlib.pyplot as plt
import os


def visual_dataset(train_loader):
    os.makedirs("./picture", exist_ok=True)
    visialization_num = 8
    example_noisy, example_origin = next(iter(train_loader))
    plt.figure(figsize=(14, 4))
    for i in range(visialization_num):
        plt.subplot(2, visialization_num, i + 1)
        plt.imshow(example_origin[i][0], cmap="gray")
        plt.axis("off")
        plt.subplot(2, visialization_num, visialization_num + i + 1)
        plt.imshow(example_noisy[i][0], cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Dataset Examples (Top: Original, Bottom: Noisy)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("./picture/dataset_examples.png")
    plt.close()


def visual_results(example_noisy, example_output):
    os.makedirs("./picture", exist_ok=True)
    num_examples = min(8, len(example_noisy))
    plt.figure(figsize=(16, 6))
    for i in range(num_examples):
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(example_noisy[i][0], cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Noisy Input", fontsize=14, rotation=90, labelpad=20)
    for i in range(num_examples):
        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.imshow(example_output[i][0], cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Denoised Output", fontsize=14, rotation=90, labelpad=20)
    plt.suptitle(
        f"Denoising Results Comparison (First {num_examples} Test Examples)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./picture/denoising_results.png")
    plt.close()
