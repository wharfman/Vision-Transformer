# Vision Transformer (ViT) Implementation

A PyTorch implementation of Vision Transformer for image classification on CIFAR-10, achieving ~77.5% validation accuracy after 300 epochs of training.

### Key Parameters

| Parameter | Value | Description |
|-----------|--------|-------------|
| Image Size | 32x32 | Input image dimensions |
| Patch Size | 4x4 | Size of image patches |
| Embedding Dim | 32 | Hidden dimension size |
| Attention Heads | 8 | Number of attention heads |
| Transformer Layers | 3 | Number of transformer blocks |
| Dropout | 0.2 | Dropout probability |
| Batch Size | 128 | Training batch size |

### Dataset Split
- Training: 90% of CIFAR-10 (45,000 images)
- Validation: 10% of CIFAR-10 (5,000 images)

## Training Results

The model was trained for 300 epochs with the following results:

<img width="4470" height="1848" alt="training_metrics" src="https://github.com/user-attachments/assets/3d216aa5-672f-4121-b307-05d412ef689d" />

- **Final Training Accuracy**: 82.66%
- **Final Validation Accuracy**: 77.46%
- **Training Loss**: Converged from 1.96 to 0.49
- **Validation Loss**: Stabilized around 0.67

## References

- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
- Vaswani, A., et al. "Attention is All You Need." NIPS 2017.

