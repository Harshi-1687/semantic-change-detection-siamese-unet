# Multi-Scale Channel-Spatial Attention Siamese U-Net for Semantic Change Detection

## Project Overview

This project implements a Siamese U-Net architecture integrated with Convolutional Block Attention Module (CBAM) for semantic change detection using the SECOND remote sensing dataset.

The model analyzes bi-temporal satellite images captured at two different time periods (T1 and T2), identifies changed regions, and classifies them into semantic categories.

## Objectives

- Detect changes between bi-temporal satellite images.
- Extract multi-scale features using a shared Siamese encoder.
- Apply CBAM attention to emphasize important channel and spatial features.
- Generate seven-class semantic change predictions.
- Evaluate model performance using accuracy, Intersection over Union (IoU), and F1-score.

## Dataset

**SECOND (Semantic Change Detection) Dataset**

The dataset contains satellite image pairs and corresponding ground-truth masks.

Dataset structure:

```text
SECOND/
├── train/
│   ├── T1/
│   ├── T2/
│   ├── GT_T1/
│   ├── GT_T2/
│   └── GT_CD/
│
└── test/
    ├── T1/
    ├── T2/
    ├── GT_T1/
    ├── GT_T2/
    └── GT_CD/
```

The project uses a custom seven-class target constructed using GT_T2 and GT_CD.

### Semantic Classes

| Class ID | Class Name |
|---|---|
| 0 | No Change |
| 1 | Non-vegetated Ground |
| 2 | Tree |
| 3 | Low Vegetation |
| 4 | Water |
| 5 | Building |
| 6 | Playground |

The GT_CD mask represents binary change information, where 0 indicates no change and 255 indicates change. The semantic category for changed pixels is obtained from GT_T2.

## Model Architecture

The proposed architecture consists of:

1. **Shared Siamese Encoder:** Extracts features from T1 and T2 using shared weights.
2. **Multi-scale Feature Difference:** Computes absolute differences between corresponding feature maps.
3. **CBAM Attention:** Applies channel and spatial attention at multiple feature scales.
4. **U-Net Decoder:** Reconstructs spatial information using decoder blocks and skip connections.
5. **Output Layer:** Produces seven-class semantic predictions.

## Technology Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow

## Implementation Details

| Parameter | Configuration |
|---|---|
| Input Image Size | 256 × 256 |
| Number of Classes | 7 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Loss Function | Cross Entropy Loss |
| Training Epochs | 50 (training interrupted at epoch 48) |
| Data Split | 90% Training, 10% Validation |
| Random Seed | 42 |

## Experimental Results

The original Siamese U-Net with CBAM achieved the following test performance:

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 85.06% |
| Test mIoU | 27.59% |
| Test Loss | 0.4666 |

### Class-wise Performance

The model was evaluated using IoU and F1-score for each semantic class. The results indicate stronger performance for the no-change and building classes, while smaller semantic categories remain challenging.

## Model Evaluation and Visualizations

The implementation includes:

- Class-wise IoU and F1-score analysis.
- Confusion matrix.
- Training and validation loss curves.
- Validation mIoU curve.
- Comparison of original and class-weighted experiments.
- Qualitative comparison of T1, T2, ground truth, and predicted masks.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/Harshi-1687/semantic-change-detection-siamese-unet.git
cd semantic-change-detection-siamese-unet
```

### 2. Install Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow
```

### 3. Run the Project

The source code is designed for Google Colab with GPU acceleration.

1. Open Google Colab.
2. Upload `dl_project.py` or copy its contents into a notebook cell.
3. Enable GPU runtime.
4. Execute the cells sequentially.
5. The SECOND dataset is downloaded automatically from Zenodo.
6. Training, evaluation, and visualization outputs are generated.

**Note:** The current implementation uses CUDA explicitly in the model section, so a CUDA-enabled GPU runtime is required.

## Project Structure

```text
semantic-change-detection-siamese-unet/
│
├── dl_project.py
├── README.md
└── requirements.txt
```

## Future Enhancements

- Improve detection of minority semantic classes.
- Experiment with advanced loss functions.
- Explore more effective feature fusion strategies.
- Evaluate additional attention mechanisms.
- Improve generalization across remote sensing datasets.

## Conclusion

This project demonstrates a deep learning approach to semantic change detection using a Siamese U-Net architecture with multi-scale CBAM attention. It combines shared feature extraction, attention mechanisms, and semantic segmentation to identify and classify changes in bi-temporal satellite imagery.

## Dataset Reference

SECOND Dataset: https://zenodo.org/records/15479555

