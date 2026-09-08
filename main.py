"""
Projeto: Segmentação semântica (pet vs fundo) com U-Net
Dataset: Oxford-IIIT Pet (via torchvision, com trimaps oficiais)

Exercício 02 - Redes Neurais Profundas (Deep Learning)
Especialização em IA Aplicada - UNISINOS

Estrutura:
1. Carregamento e análise do dataset
2. Pré-processamento
3. Construção do modelo (U-Net)
4. Treinamento e validação
5. Avaliação e interpretação
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {device}")

# ============================================================
# 1. CARREGAMENTO E ANÁLISE DO DATASET
# ============================================================

IMG_SIZE = 128  # múltiplo de 32, leve o suficiente pra treinar rápido

img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # já normaliza para [0,1]
])

# O trimap original tem valores {1: pet, 2: fundo, 3: borda}.
# Vamos binarizar: pet+borda = 1 (classe de interesse), fundo = 0.
def mask_transform(mask):
    mask = mask.resize((IMG_SIZE, IMG_SIZE), resample=0)  # NEAREST, não interpolar labels
    mask = torch.from_numpy(np.array(mask)).long()
    binary_mask = torch.where((mask == 1) | (mask == 3), 1, 0).float()
    return binary_mask.unsqueeze(0)  # shape (1, H, W)

full_dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="segmentation",
    transform=img_transform,
    target_transform=mask_transform,
    download=True,
)

test_dataset = OxfordIIITPet(
    root="./data",
    split="test",
    target_types="segmentation",
    transform=img_transform,
    target_transform=mask_transform,
    download=True,
)

# Split treino/validação (80/20) a partir do trainval
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Treino: {len(train_dataset)} | Validação: {len(val_dataset)} | Teste: {len(test_dataset)}")

# Visualização inicial: imagem + máscara sobreposta
def show_sample(dataset, idx=0):
    img, mask = dataset[idx]
    img_np = img.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Imagem original")
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Máscara binária")
    axes[2].imshow(img_np)
    axes[2].imshow(mask_np, cmap="jet", alpha=0.4)
    axes[2].set_title("Sobreposição")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_visualization.png", dpi=100)
    plt.close()
    print("Visualização salva em sample_visualization.png")

show_sample(train_dataset, idx=0)


# ============================================================
# 2. PRÉ-PROCESSAMENTO (augmentation leve, opcional)
# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ============================================================
# 3. CONSTRUÇÃO DO MODELO (U-Net simplificada)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)  # logits (sem sigmoid) — aplicamos na loss/inferência


model = UNet(in_channels=3, out_channels=1).to(device)


# ============================================================
# LOSS: BCE + Dice combinadas
# ============================================================

def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def combined_loss(pred_logits, target):
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    return bce + dice


def iou_score(pred_logits, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


# ============================================================
# 4. TREINAMENTO E VALIDAÇÃO
# ============================================================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

EPOCHS = 20
history = {"train_loss": [], "val_loss": [], "val_iou": []}
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = combined_loss(preds, masks)
            val_loss += loss.item() * imgs.size(0)
            val_iou += iou_score(preds, masks) * imgs.size(0)
    val_loss /= len(val_dataset)
    val_iou /= len(val_dataset)

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | "
          f"val_loss={val_loss:.4f} | val_IoU={val_iou:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

# Curvas de treino/validação (monitorar overfitting)
plt.figure(figsize=(8, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.title("Curva de perda - treino vs validação")
plt.savefig("loss_curve.png", dpi=100)
plt.close()


# ============================================================
# 5. AVALIAÇÃO E INTERPRETAÇÃO
# ============================================================

model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()

test_iou = 0.0
test_dice = 0.0
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        test_iou += iou_score(preds, masks) * imgs.size(0)
        test_dice += (1 - dice_loss(preds, masks).item()) * imgs.size(0)

test_iou /= len(test_dataset)
test_dice /= len(test_dataset)
print(f"\nResultado final no conjunto de TESTE:")
print(f"IoU médio: {test_iou:.4f}")
print(f"Dice médio: {test_dice:.4f}")

# Visualização qualitativa: imagem + máscara real + máscara predita
def show_predictions(n=5):
    model.eval()
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    with torch.no_grad():
        for i in range(n):
            img, mask = test_dataset[i]
            pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))
            pred = (pred > 0.5).float().cpu().squeeze().numpy()

            axes[i, 0].imshow(img.permute(1, 2, 0).numpy())
            axes[i, 0].set_title("Imagem")
            axes[i, 1].imshow(mask.squeeze().numpy(), cmap="gray")
            axes[i, 1].set_title("Máscara real")
            axes[i, 2].imshow(pred, cmap="gray")
            axes[i, 2].set_title("Máscara predita")
            axes[i, 3].imshow(img.permute(1, 2, 0).numpy())
            axes[i, 3].imshow(pred, cmap="jet", alpha=0.4)
            axes[i, 3].set_title("Sobreposição")
            for j in range(4):
                axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig("predictions_sample.png", dpi=100)
    plt.close()
    print("Predições salvas em predictions_sample.png")

show_predictions(n=5)