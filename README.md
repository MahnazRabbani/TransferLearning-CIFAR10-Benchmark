# Transfer Learning Benchmarks: A Comparative Study of Image Classification on CIFAR10 


The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images. The classes include objects like cars, birds, dogs, and ships, among others.              

In this project, we will benchmark three distinct deep learning architectures—ResNet-50, Vision Transformer (ViT), and EfficientNet—on the CIFAR-10 dataset. ResNet-50, originally trained on the ImageNet dataset, consists of 50 layers and introduces the concept of residual connections. Vision Transformer (ViT) is a recent innovation that applies transformers to image classification and is also pretrained on ImageNet. EfficientNet provides a systematic way to scale up networks in a balanced manner and has been trained on several large-scale datasets.             

### Transfer learning         

Transfer learning is a machine learning technique where a model developed for a particular task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pretrained models are adapted for a specific but related problem. By leveraging knowledge gained from original training on extensive and diverse datasets, transfer learning enables the model to achieve high performance on new tasks without needing to train from scratch. In the context of various projects, including the one focused on CIFAR-10, transfer learning can be implemented using two main methods:

- **Finetuning the pre-trained model**: Instead of random initialization, the network is initialized with a pretrained network. The rest of the training proceeds as usual, allowing the model to fine-tune the pretrained features to the specific new task.              

- **Pre-trained model as Fixed Feature Extractor**: In this method, the weights for all of the network except the final fully connected layer are frozen. This last fully connected layer is replaced with a new one with random weights, and only this layer is trained, treating the pretrained network as a fixed feature extractor.              


## Project Overview

This project aims to evaluate and compare the performance of different deep learning models on the CIFAR10 dataset by employing various transfer learning techniques. The models explored include ResNet50, Vision Transformer (ViT), and EfficientNet-B0. The key objectives are:                


**1. Fine-tuning ResNet50**        
**2. ResNet50 as a Fixed Feature Extractor**                
**3. Fine-tuning Vision Transformer (ViT)**             
**4. Benchmarking EfficientNet-B0**                


## Technologies Used

**Libraries**: HuggingFace Transformers, PyTorch             
**Techniques**: Transfer Learning, Fine-Tuning               
**Tools**: CUDA for GPU optimization                   

## Dataset       

The CIFAR10 dataset was used, processed using the HuggingFace Datasets library, and further processed for model compatibility using respective model checkpoints.

## Models and Fine-Tuning           

### ResNet50       

**Full Fine-Tuning**: Initializing with pretrained weights and training the entire network.            
**Feature Extractor**: Freezing all weights except the final fully connected layer.

### Vision Transformer (ViT) 

Fine-Tuning: Training on CIFAR10.

### EfficientNet-B0

Fine-Tuning on CIFAR10. The efficientnet_V2 model could be too large to be fine-tuned on a typical desktop with one GPU, similar to what was used for this project. 


## Results

The models were evaluated using accuracy as the key metric.

**Accuracy:**            

**ViT Fine-Tuning**: 93.43%                
**ResNet Fine-Tuning**: 79.32%     
**EfficientNet-B0 Fine-Tuning**: 73.64%          
**ResNet Feature Extraction**: 46.20%              


## Insights

- ViT achieved 90% accuracy with only 3 epochs and 5k data, whereas ResNet50 only reached 30% with similar settings.          
- ViT reached  93.43% accuracy with 6 epochs and the entire dataset. Showing an increased of approximately 3.12% reflecting the improvement obtained through using more data and more training epochs.

- ResNet50's fine-tuning reached 79.32% accuracy, but feature extraction only yielded 46.20%.       


## Instructions


Please refer to the notebooks for each model:
         
- Resnet50.ipynb            
- vit.ipynb      
- Efficientnet.ipynb         

**Note**: Fine-tuned models are not included in the GitHub repo due to github size limits.


## Potential Next Step

**Implementing Quantized Transfer Learning**

A strategic approach to further enhance the transfer learning process could involve a two-phase training method. In the initial phase, the feature extractor can be frozen, allowing only the head (the final fully connected layer) to be trained. This will enable the model to adapt to the specific task without altering the pre-learned features.           

Once this first phase is complete, the subsequent phase can commence by unfreezing either the entire feature extractor or selected parts of it. By setting a reduced learning rate, the model can continue training, allowing for a more refined fine-tuning of the previously frozen layers. This method can potentially lead to a more nuanced adaptation of the pre-trained model to the target task, aligning with the principles of [Quantized Transfer Learning Tutorial](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html).

## Acknowledgments

The data used is from CIFAR10. No collaborations were involved in this project.