# CIFAR10 Transfer Learning Benchmarks: A Comparative Study

Benchmarking finetuned architectures ResNet-50, ViT, and EfficientNet on the CIFAR-10.

## Project Overview

This project aims to evaluate and compare the performance of different deep learning models on the CIFAR10 dataset by employing various transfer learning techniques. The models explored include ResNet50, Vision Transformer (ViT), and EfficientNet-B0. The key objectives are:                


**1. Fine-tuning ResNet50**:            

Two methods were explored:             

**1.1** Initializing the network with a pretrained ResNet50.            

**1.2** Using ResNet50 as a fixed feature extractor and only training the final fully connected layer.         

**2. Fine-tuning Vision Transformer (ViT)**            

**3. Benchmarking EfficientNet-B0**


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

Benchmarking: Evaluation without specific fine-tuning. At first the goal was to finetune efficientnet_V2 but the model was too large and we encounter GPU out of memory error. 


## Results

The models were evaluated using accuracy as the key metric.

**ViT Fine-Tuning**: Accuracy: 93.43%                
**ResNet Fine-Tuning**: Accuracy: 79.32%                
**ResNet Feature Extraction**: Accuracy: 46.20%              


## Insights

- ViT achieved 90% accuracy with only 3 epochs and 5k data, whereas ResNet50 only reached 30% with similar settings.          
- ViT reached  93.43% accuracy with 6 epochs and the entire dataset. Showing an increased of approximately 3.12% reflecting the improvement obtained through using more data and more training epochs.

- ResNet50's fine-tuning reached 79.32% accuracy, but feature extraction only yielded 46.20%.       


## Instructions


Please refer to the notebooks for each model:
         
- Resnet50.ipynb            
- vit.ipynb      
- Efficientnet.ipynb         

**Note**: Fine-tuned models are not included in the GitHub repo due to size limits.


## Potential Next Step

**Implementing Quantized Transfer Learning**

A strategic approach to further enhance the transfer learning process could involve a two-phase training method. In the initial phase, the feature extractor can be frozen, allowing only the head (the final fully connected layer) to be trained. This will enable the model to adapt to the specific task without altering the pre-learned features.           

Once this first phase is complete, the subsequent phase can commence by unfreezing either the entire feature extractor or selected parts of it. By setting a reduced learning rate, the model can continue training, allowing for a more refined fine-tuning of the previously frozen layers. This method can potentially lead to a more nuanced adaptation of the pre-trained model to the target task, aligning with the principles of [Quantized Transfer Learning Tutorial](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html).

## Acknowledgments

The data used is from CIFAR10. No collaborations were involved in this project.