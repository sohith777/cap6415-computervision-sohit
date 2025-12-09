# cap6415-computervision-sohit
CIFAR-10 class-imbalance project for CAP6415 Topic 8: ResNet-18 baseline, class-weighted fine-tuning on worst class (cat), plus MobileNetV2 and transfer-learning comparisons.

## Abstract

In this project I train and analyze image classifiers on the Kaggle CIFAR-10 dataset (ayush1220/cifar10), which contains 10 classes of 32×32 color images. My main goal is to identify which class is hardest for a small convolutional network to recognize and to improve the model’s performance on that class without hurting overall accuracy.

I first train a ResNet-18 model from scratch as a baseline and compute per-class precision, recall, F1, and the confusion matrix. This analysis shows that the cat class has the lowest precision and recall. To address this, I fine-tune the same ResNet-18 using a class-weighted cross-entropy loss that doubles the loss weight of the cat class while keeping the rest of the training pipeline the same. After fine-tuning, the overall validation accuracy remains around 83–84%, but cat precision and recall increase, and many test images that were previously misclassified are now predicted correctly. I also run shorter experiments with MobileNetV2 and a transfer-learning ResNet-18 with ImageNet weights to compare architectures, and I use plots of accuracy, per-class metrics, and example images to show that targeted class-weighted fine-tuning is an effective way to improve the most challenging class on CIFAR-10.

## Dataset

I use the Kaggle CIFAR-10 dataset:

- Source: `ayush1220/cifar10` (Kaggle)
- Contents: 50,000 training images and 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Format: 32×32 RGB images organized by class in `train/` and `test/` folders

In this repository I do **not** store the dataset. The notebook expects the Kaggle zip file to be downloaded separately and placed in the repository root as:

```text
archive (9).zip
