# SSD-DIS
## Overview

This repository contains the core code for generating the SSD-DIS (Semi-Synthetic Dataset for Document Image Shadow Removal) dataset described in our paper. The dataset itself (pre-generated) can be downloaded via the links below. This code allows you to reproduce or extend the dataset using your own collection of shadow-free document images.

## Dataset Download (Pre-generated)

The complete SSD-DIS dataset (12,224 image triples) is available at:

Figshare (Recommended for international users): https://figshare.com/account/articles/29401811

Baidu Disk (百度网盘, for users in China): https://pan.baidu.com/s/5ZmiPhoGZz1kqI0KxFl93aQ

## Project Structure
```
shadow_synthesis/
├── batch_synthesize.py          # Main script
├── shadows/                     # Shadow mask library folder
├── shadow_free_images/          # Shadow-free document images (train/val/test split)
└── outputs/                     # Generated datasets (auto-created)
```
## Dependency List

1. Python 3.6 or higher: The programming language used for the script.
2. PyTorch 1.9.0 or higher: A deep learning framework used for tensor operations and image reading.
3. torchvision 0.10.0 or higher: A library for computer vision that provides image transformations and utilities.
4. Pillow (PIL) 8.0.0 or higher: A library for image processing, used for opening, manipulating, and saving images.

   
## Quick Start

1. **Prepare Data**:
   - Place shadow mask images in the `shadows/` folder
   - Place shadow-free document images in `shadow_free_images/train/` (or val/test)

2. **Configure Parameters**:
   Edit the configuration section in `batch_synthesize.py`:
   ```python
   # Shadow mask library folder
   SHADOW_MASK_LIBRARY_DIR = 'path/to/your/shadows'
   
   # Shadow-free document images folder (choose train/val/test as needed)
   SHADOW_FREE_IMAGES_DIR = 'path/to/your/shadow_free_images/train'
   
   # Output folder settings
   OUTPUT_SHADOW_MASKS_DIR = 'path/to/output/shadow_masks/train'
   OUTPUT_SHADOW_IMAGES_DIR = 'path/to/output/shadow_images/train'
   OUTPUT_ADJUSTED_MASKS_DIR = 'path/to/output/adjusted_masks/train'
   
   # Shadow intensity range (recommended: 0.15-0.8)
   SHADOW_INTENSITY_RANGE = (0.15, 0.8)
   ```

3. **Run the Script**:
   ```bash
   python batch_synthesize.py
   ```

## Output Description

The program generates three directories:

1. **vers_shadow_masks/** - Resized original binary shadow masks
2. **vers_shadow_images/** - Synthetic shadow document images
3. **vers_shadow_masks_intensity_adjusted/** - Intensity-adjusted shadow masks (inverted for visualization)

## Parameter Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SHADOW_MASK_LIBRARY_DIR` | Path to shadow mask library | User-defined |
| `SHADOW_FREE_IMAGES_DIR` | Path to shadow-free images | User-defined |
| `OUTPUT_SHADOW_MASKS_DIR` | Output path for shadow masks | User-defined |
| `OUTPUT_SHADOW_IMAGES_DIR` | Output path for shadow images | User-defined |
| `OUTPUT_ADJUSTED_MASKS_DIR` | Output path for adjusted masks | User-defined |
| `SHADOW_INTENSITY_RANGE` | Shadow intensity range | (0.15, 0.8) |

## Algorithm Workflow

1. **Random Mask Selection**: Randomly select a shadow mask for each shadow-free document image
2. **Mask Resizing**: Resize the mask to match the document image dimensions
3. **Random Intensity Generation**: Generate random shadow intensity within the specified range for each image
4. **Shadow Synthesis**: Multiply the document image with the shadow mask (with applied intensity)
5. **Save Results**: Save the synthetic image, original mask, and adjusted mask simultaneously

## Important Notes

1. Shadow masks should be binary images (0=shadow area, 255=non-shadow area)
2. Shadow-free document images should be RGB three-channel images
3. It's recommended to create output folders before running the script
4. The program will automatically create non-existent output directories
5. Shadow intensity range can be adjusted: (0.15, 0.8) for natural shadows, (0.8, 1.0) for stronger shadows


## Troubleshooting

1. **Directory Not Found Error**: Ensure all input paths are correct
2. **Insufficient Memory**: Reduce the number of images processed at once
3. **Image Format Error**: Ensure all images can be read by PIL library
4. **Output Permission Issues**: Ensure write permissions for output directories
