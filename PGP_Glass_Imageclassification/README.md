## Pytorch-Image-Classification

A simple demo of **image classification** using pytorch. Here, we use a **custom dataset** containing **10000 images** belonging to **7 classes** for training(and validation). Also, for training viz. **finetuning the convnet and convnet as a feature extractor**, with the help of **pretrained** pytorch models. The models used include: **Resnet18**.

The Trained Model is **model.pth**
https://drive.google.com/uc?id=1-FoQ3mv_4kl9b45j8LBa6WIebqm1hxSn

### Dependencies
create a conda environment

* Python3.7, Scikit-learn
* Pytorch, PIL
* Torchsummary, Tensorboard

```python
pip install torchsummary # keras-summary
pip install tensorboard  # tensoflow-logging
```


### How to run

**Download** and extract training dataset: [imds_small](https://drive.google.com/file/d/1fPDnom5uGTpCb0abkzCvKbLadtNx8FlW/view?usp=sharing)

Run the following **scripts** for training and/or testing

```python
# For training the model (image_classify_train.ipynb)
python test.py  # For testing the model on sample images(pass the testing folder containing test images)
python eval.py  # For evaluating the model on new dataset
```
**test.csv** result file

### Training results


| **Resnet18**  | 99.85  | 44.8 MB |  42 mins |  finetune |


**Batch size**: 64, **GPU**: Google colab GPU

 **Resnet18(transfer leraning) was trained for **10 epochs**


### Evaluation

Here we **evaluate** the performance of our **best model - resnet18** on a **new data-set** containing 50 images per class.

Accuracy of the network on the 350 test images: 100.00%
Confusion Matrix
----------------
[[50  0  0  0  0  0  0]
 [ 0 50  0  0  0  0  0]
 [ 0  0 50  0  0  0  0]
 [ 0  0  0 50  0  0  0]
 [ 0  0  0  0 50  0  0]
 [ 0  0  0  0  0 50  0]
 [ 0  0  0  0  0  0 50]] 

Per class accuracy
------------------
Accuracy of class Badminton : 100.00 %
Accuracy of class  Cricket : 100.00 %
Accuracy of class   Karate : 100.00 %
Accuracy of class   Soccer : 100.00 %
Accuracy of class Swimming : 100.00 %
Accuracy of class   Tennis : 100.00 %
Accuracy of class Wrestling : 100.00 %
```

```
### Observations

1. In **transfer learning**, if your custom dataset is **similar** to the pretrained model's training dataset, then you can easily acheive very **high accuracy**(>90) with very **few training epochs**(<10).

 
### Acknowledgments
* "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"
* "https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html"
* "https://www.learnopencv.com/image-classification-using-transfer-learning-in-pytorch/"
* "https://towardsdatascience.com/https-medium-com-dinber19-take-a-deeper-look-at-your-pytorch-model-with-the-new-tensorboard-built-in-513969cf6a72"
* "https://www.aiworkbox.com/lessons/how-to-define-a-convolutional-layer-in-pytorch#lesson-transcript-section"
* "https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20"
* "https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict"
* "https://www.kaggle.com/c/understanding_cloud_organization/discussion/112582"
* "https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial"
