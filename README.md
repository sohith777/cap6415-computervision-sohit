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

Project structure

Key files and folders:

sohit_final_cpmputervision19_revised_final.ipynb
Main notebook with all parts (1–8) of the project.

README.md
Project description, running instructions, and expected results.

requirements.txt
Python package dependencies for running the notebook.

logs/
Text files describing week-by-week progress (development logs).

baseline_results/, finetune_results/, alt_results/, transfer_results/
Created by the notebook at runtime to store model checkpoints and metric JSONs (not tracked in Git).


how to run

Clone the repository and enter the project directory:

how to run

Clone the repository and enter the project directory:


Install dependencies (use a virtual environment if desired):

pip install -r requirements.txt


Download the Kaggle CIFAR-10 dataset:

Go to Kaggle and download the dataset ayush1220/cifar10.

Save the downloaded zip file in the project root and name it:

archive (9).zip


Open the notebook sohit_final_cpmputervision19_revised_final.ipynb in Jupyter, VS Code, or Google Colab, with the working directory set to the project root.

Run all cells in order from Part 1 to Part 8. The notebook will:

Extract the CIFAR-10 data from archive (9).zip.

Create train and test dataloaders with standard CIFAR-10 transforms.

Train a baseline ResNet-18 model from scratch.

Compute per-class metrics and identify the worst class (cat).

Fine-tune ResNet-18 using class-weighted loss focused on cat.

Train a small MobileNetV2 model for a few epochs.

Train a transfer-learning ResNet-18 model using ImageNet weights.

Generate plots, confusion matrices, and example images used in the report and demo video.



Part 1 – Data loading 

In Part 1 I load the CIFAR-10 dataset from Kaggle's zip file archive (9).zip and extract it into the Colab file system. I set up standard image transforms such as random cropping, horizontal flipping, and normalization using CIFAR-10 mean and standard deviation. Then I build PyTorch ImageFolder datasets pointing to the train and test folders and wrap them in dataloaders for training and evaluation. I also show a small batch of images with labels to confirm that data is read correctly.

Part 2 – Baseline ResNet-18 training 

In Part 2 I define the baseline model as a ResNet-18 with its final fully connected layer adjusted to output 10 classes. I keep all model state in the sakt_p dictionary and set up the loss function, Adam optimizer, and learning rate scheduler. Then I train the model on the CIFAR-10 training set for 25 epochs and evaluate it on the test set after each epoch to track validation loss and accuracy. I save the best-performing checkpoint and a JSON history of train and validation curves for later analysis.

Part 3 – Baseline per-class metrics and confusion matrix

In Part 3 I reload the best baseline ResNet-18 checkpoint and run it over the full test set in evaluation mode. I collect all true labels and predictions and compute the confusion matrix and per-class precision, recall, F1, and support using scikit-learn. I store these metrics in sakt_p and save them to a JSON file for reproducibility. I sort the classes by precision to identify the worst class, which turns out to be the cat class, and I plot the confusion matrix to visualize common confusions.

Part 4 – Class-weighted fine-tuning for the worst class

In Part 4 I fine-tune the baseline ResNet-18 starting from the saved weights and modify the training objective to focus on the worst class. I build a class-weight vector that sets weight 2.0 for the cat class and 1.0 for all other classes and pass this into a weighted cross-entropy loss. Then I run an additional 10 epochs of training with a smaller learning rate while tracking train and validation performance. I save the best fine-tuned model and its training history so I can compare it against the baseline.

Part 5 – Baseline vs fine-tuned comparison 

In Part 5 I load both the baseline and fine-tuned models and compute per-class metrics for the fine-tuned model in the same way as for the baseline. I then compare precision and recall for every class and report how the cat class changes after fine-tuning. For the cat class, precision, recall, and F1 all increase while overall validation accuracy stays around the same level. I also create bar plots that show per-class precision for both models side by side and highlight the improvement on the cat class.

Part 6 – Cat examples fixed and harmed by fine-tuning

In Part 6 I investigate individual cat images to understand the effect of fine-tuning more concretely. I run both the baseline and fine-tuned models on the test set and look specifically at test images whose true label is cat. I collect images where the baseline model was wrong and the fine-tuned model is correct, and also cases where the baseline was correct and the fine-tuned model becomes wrong. I unnormalize these images and display them in grids so I can visually inspect what types of cat images are helped or harmed by the class-weighted fine-tuning.

Part 7 – Extra experiment with MobileNetV2 

In Part 7 I add an extra small model, MobileNetV2, to compare architecture changes against fine-tuning. I define MobileNetV2 with its classifier adjusted to output 10 classes and train it on CIFAR-10 for 5 epochs to keep the run time short. I then compute per-class precision and recall for this model and save its metrics to disk. Finally, I compare its overall validation accuracy and cat performance against the baseline and fine-tuned ResNet-18 and summarize whether MobileNetV2 alone can match the targeted improvement on the worst class.

Part 8 – Transfer-learning ResNet-18 and final comparison across models

In Part 8 I build a more advanced model by using ResNet-18 with ImageNet pretrained weights and applying transfer learning. I freeze all convolutional layers and replace the final fully connected layer with a new classifier for the 10 CIFAR-10 classes, then train only this layer for 5 epochs. After training I compute per-class metrics for the transfer-learning model and save them alongside the other models. I then create final comparison plots showing overall validation accuracy and cat precision and recall for four models: baseline ResNet-18, class-weighted fine-tuned ResNet-18, MobileNetV2, and transfer-learning ResNet-18, and use these plots to show which approach gives the best performance on both the dataset as a whole and on the cat class specifically.






