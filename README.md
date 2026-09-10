# Pet-Segmentation-Unet

Binary semantic segmentation project (pet vs. background) using a U-Net trained from scratch on the **Oxford-IIIT Pet** dataset.

> Developed for the Deep Learning course (Redes Neurais Profundas) — Specialization in Applied Artificial Intelligence, UNISINOS.

## About the project

The goal is to train a segmentation model capable of separating, pixel by pixel, the animal (dog or cat) from the image background. The dataset already provides official masks (trimaps), converted here into binary masks.

- **Architecture:** U-Net (encoder-decoder with skip connections), implemented from scratch in PyTorch
- **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 dog and cat breeds, ~7,400 images with trimap annotations
- **Loss:** combination of Binary Cross-Entropy + Dice Loss
- **Metrics:** IoU (Jaccard) and Dice coefficient

## Repository structure

```
.
├── main.py                    # main script (dataset → training → evaluation)
├── README.md
├── requirements.txt
└── outputs/                   # generated after running the script
    ├── sample_visualization.png
    ├── loss_curve.png
    ├── predictions_sample.png
    └── best_model.pth
```

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/guustavomc/Pet-Segmentation-Unet.git
cd pet-segmentation-unet
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python main.py
```

The dataset is downloaded automatically on the first run (via `torchvision.datasets.OxfordIIITPet`), including official images and trimaps — no manual download is required.

> Running in a GPU environment (e.g., Google Colab) is recommended. Training is significantly slower on CPU.

## Pipeline steps

1. **Dataset loading and analysis** — download via torchvision, train/validation/test split, initial visualization of image + mask.
2. **Preprocessing** — resizing to 128×128, normalization, and trimap binarization (pet + border = class 1, background = class 0).
3. **Model construction** — U-Net with 4 depth levels, batch normalization, and skip connections.
4. **Training and validation** — Adam + ReduceLROnPlateau, combined loss (BCE + Dice), overfitting monitoring via train/validation loss curve.
5. **Evaluation and interpretation** — IoU and Dice calculation on the test set, visualization comparing image, ground-truth mask, and predicted mask.

## Results

| Metric | Value |
|---|---|
| IoU (test) | 0.8520 |
| Dice (test) | 0.8985 |

*(Add loss curves and prediction examples generated after running the script here.)*

## Possible improvements

- Replace the from-scratch U-Net with a pretrained encoder (e.g., `segmentation-models-pytorch` with a ResNet34 backbone)
- Data augmentation (flip, rotation, brightness) via `albumentations`
- Automatic early stopping
- Resolution adjustment (256×256) for higher edge precision
- Final visualization with the predicted mask overlaid on the original image (overlay), as already done in step 1

## References

- Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). *Cats and Dogs*. IEEE CVPR.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
