import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from model import UNet
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, jaccard_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
from datetime import datetime

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

        self.aug_transform = A.Compose([
            A.Resize(150, 150),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        self.basic_transform = A.Compose([
            A.Resize(150, 150),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L")) / 255.0
        mask = np.expand_dims(mask, axis=-1)  # Shape: (H, W, 1)
    
        transform = self.aug_transform if self.augment else self.basic_transform
        augmented = transform(image=image, mask=mask)
    
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].permute(2, 0, 1).float()  # CHW
    
        return image_tensor, mask_tensor


class Metrics:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def update(self, preds, masks):
        preds = preds.detach().cpu().numpy().round().astype(int).flatten()
        masks = masks.detach().cpu().numpy().round().astype(int).flatten()
        self.y_true.extend(masks)
        self.y_pred.extend(preds)

    def compute(self):
        precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        iou = jaccard_score(self.y_true, self.y_pred, zero_division=0)
        return precision, recall, iou

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - ((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return self.bce(inputs, targets) + self.dice(inputs, targets)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load images and masks
    image_paths = sorted(glob(os.path.join(args.train_image_path, "*.png")))
    mask_paths = sorted(glob(os.path.join(args.train_mask_path, "*.png")))

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_dataset = SegmentationDataset(train_imgs, train_masks, augment=True)
    val_dataset = SegmentationDataset(val_imgs, val_masks, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = UNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    num_epochs = args.epoch

    train_losses, val_losses = [], []
    val_precisions, val_recalls, val_ious = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        metrics = Metrics()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                metrics.update(preds, masks)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        precision, recall, iou = metrics.compute()
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_ious.append(iou)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - IoU: {iou:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Plot and save metrics
    #epochs = list(range(1, len(train_losses) + 1))
	#
    #plt.figure(figsize=(12, 6))
    #plt.subplot(1, 2, 1)
    #plt.plot(epochs, train_losses, label='Train Loss')
    #plt.plot(epochs, val_losses, label='Val Loss')
    #plt.xlabel("Epochs")
    #plt.ylabel("Loss")
    #plt.legend()
    #plt.title("Loss Curve")
	#
    #plt.subplot(1, 2, 2)
    #plt.plot(epochs, val_ious, label='IoU')
    #plt.plot(epochs, val_precisions, label='Precision')
    #plt.plot(epochs, val_recalls, label='Recall')
    #plt.xlabel("Epochs")
    #plt.ylabel("Score")
    #plt.legend()
    #plt.title("Validation Metrics")
	#
    #plt.tight_layout()
    #plt.savefig("training_metrics.png")
    #plt.close()

    # Save final model
    save_path = os.path.join(args.save_path, "UNetModel.pth")
    torch.save(model.state_dict(), save_path)
    print("Training complete. Final model saved as UNetModel.pth")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("UNet")

    parser.add_argument("--train_image_path", type=str, required=True, 
                        help="path to the image that used to train the model")
    parser.add_argument("--train_mask_path", type=str, required=True,
                        help="path to the mask file for training")
    parser.add_argument('--save_path', type=str, required=True,
                        help="path to store the checkpoint")
    parser.add_argument("--epoch", type=int, default=20, 
                        help="training epochs")
    args = parser.parse_args()
    
    start = datetime.now()
    train(args)
    print("Total training time: ", datetime.now() - start)
