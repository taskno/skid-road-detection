# Skid Road detection using UNet variations
## Introduction
The repository is a collection of UNet and two variations (SAM2-UNet, SAM2-UNeXT) to perform training and predictions for skid road detection.

- Built with PyTorch
- Uses U-Net based architectures from `networks/`
- Evaluation metrics in `evaluation/`
- Tested on Aargau,Switzerland ([skidroad_finder](https://github.com/RaffiBienz/skidroad_finder)) and Lower Austria Datasets

## How to's

1. Install:
   ```bash
   git clone https://github.com/taskno/skid-road-detection.git
   cd skid-road-detection
   pip install -r requirements.txt

2. Train:
- UNet
   ```bash
   cd networks/unet
   python train.py --train_image_path "<TRAIN-SET-DIR>/images/" \
                   --train_mask_path ""<TRAIN-SET-DIR>/masks/" \
                   --save_path "<MODEL-SAVE-DIR>" --epoch 20
- SAM2-UNet
   ```bash
   cd networks/sam2unet
   python train.py --hiera-path "<HIERA-DIR>/sam2-hiera_large.pt" \
                   --train_image_path "<TRAIN-SET-DIR>/images/" \
                   --train_mask_path ""<TRAIN-SET-DIR>/masks/" \
                   --save_path "<MODEL-SAVE-DIR>" --epoch 20 --lr 0.001 --batch-size 20
- SAM2-UNeXT
   ```bash
   cd networks/sam2unext
   python train.py --hiera-path "<HIERA-DIR>/sam2-hiera_large.pt" \
                   --dinov2-path "<DINO-DIR>/model.safetensors" \
                   --train_image_path "<TRAIN-SET-DIR>/images/" \
                   --train_mask_path ""<TRAIN-SET-DIR>/masks/" \
                   --save_path "<MODEL-SAVE-DIR>" --epoch 20 --lr 0.0002 --batch-size 1
   
3. Test:
- UNet
   ```bash
   cd networks/unet
   python python predict.py --checkpoint "<MODEL-SAVE-DIR>/UNetModel.pth" \
                            --test_image_path "<TEST-SET-DIR>/images/" \
                            --save_path "<PRED-DIR>/pred_unet/"

   
- SAM2-UNet
   ```bash
   cd networks/sam2unet
   python test.py --checkpoint "<MODEL-SAVE-DIR>/SAM2-UNet-20.pth" \
                  --test_image_path "<TEST-SET-DIR>/images/" \
                  --test_gt_path "<TEST-SET-DIR>/masks/" \
                  --save_path "<PRED-DIR>/pred_sam2unet/"
- SAM2-UNeXT
   ```bash
   cd networks/sam2unext
   python test.py --checkpoint "<MODEL-SAVE-DIR>/SAM2-UNeXT-20.pth" \
                  --test_image_path "<TEST-SET-DIR>/images/" \
                  --test_gt_path "<TEST-SET-DIR>/masks/" \
                  --save_path "<PRED-DIR>/pred_sam2unext/"
4. Evaluate:

*PRED_DIR and GT_DIR (lines 27-28 in eval.py) paths should be set properly depending on the prediction and mask directories of corresponding CNNs.
```python evaluation/eval.py```

## Model Zoo

## Citation
*Özkan, T., Gasica, T.A., Bienz, R., Zeiner, R., Hofstätter, M., Pfeifer, N., Hollaus, M. **Skid Road Detection from ALS point clouds...**, 2026

## Contact
```bash
taskin.oezkan@geo.tuwien.ac.at
