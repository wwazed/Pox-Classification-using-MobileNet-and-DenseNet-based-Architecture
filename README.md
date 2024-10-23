# Pox-Classification-using-MobileNet-and-DenseNet-based-Architecture
The script is structured to perform the classification of pox-related skin lesions using a deep learning model. 
## Key Features
In this script, two sequential models are built using DenseNet 121 and MobileNet V2. Both models follow the same architecture described below:
### 1. Sequential Model Architecture:
The model follows a Sequential structure, which means layers are stacked one after the other. This is a simple but effective architecture for image classification tasks.
### 2. Convolutional Neural Networks (CNN):
Several layers of 2D convolution (**Conv2D**) are used. These layers help in extracting spatial features from the input images by learning filters for detecting patterns like edges, textures, etc. The **ReLU (Rectified Linear Unit)** activation function is used in the convolutional layers to introduce non-linearity, which helps in learning complex patterns. **MaxPooling Layers** are used to reduce the spatial dimensions of the feature maps, making the computation more efficient while preserving the important features.
### 3. Dropout Layers:
The architecture includes Dropout layers to prevent overfitting by randomly setting a fraction of input units to zero during training. This helps the model generalize better.
### 4. Fully Connected (Dense) Layers:
After the convolutional layers, the model includes Dense layers. These layers are fully connected, meaning every neuron in a layer is connected to every neuron in the previous layer. The last Dense layer uses **Softmax activation** to output probabilities for the multiple classes (Chickenpox, Cowpox, HFMD, Healthy, Measles, Monkeypox).
## Data Preprocessing
The dataset was downloaded from Kaggle. This comprises images from six distinct classes, namely **Mpox, Chickenpox, Measles, Cowpox, Hand-foot-mouth disease or HFMD, and Healthy**. The dataset includes 755 original skin lesion images sourced from 541 distinct patients, ensuring a representative sample.
### Image Data Augmentation:
The **ImageDataGenerator** is used to preprocess and augment the image data. This tool performs several preprocessing transformations on the images before feeding them into the model:

**Rescaling**: The pixel values are rescaled (normalized) to fall between 0 and 1 by dividing by 255. This helps the model converge faster.

**Rotation**: Random rotations are applied to the images within a specified range, which helps the model handle images taken from different angles.

**Width and Height Shifts**: Random shifts are applied to the images to help the model become robust to positional variations.

**Zooming**: Random zooms are performed, allowing the model to handle variations in the scale of objects within the images.

**Horizontal Flips**: Images are randomly flipped horizontally to introduce variation and increase the diversity of the training data.

Note: These augmentations are applied in real-time during training, meaning each image can be presented to the model in slightly different variations each time.
###  Resizing and Batching:
All images are resized to 224x224 to maintain uniformity across the dataset. This ensures the images match the input shape required by the Convolutional Neural Network (CNN).
### Data Shuffling:
The data is shuffled during training, ensuring that the model does not learn any unwanted patterns from the order in which the data is presented. Shuffling is especially important in preventing the model from learning temporal or positional correlations that could lead to overfitting.

Since the task is a multi-class classification problem, the class labels are one-hot encoded (**categorical encoding**).

## Model Construction and Compilation
### DenseNet 121
DenseNet121, a pre-trained convolutional neural network that is part of the DenseNet family, known for its dense connections between layers. DenseNet121 is loaded with pre-trained weights from ImageNet and used as a feature extractor. The model's top layers (classification head) are modified to suit the specific multi-class classification task of pox disease detection. After the DenseNet121 backbone, a **Global Average Pooling** layer is added to reduce the spatial dimensions before connecting to fully connected Dense layers. The final layer uses **softmax activation** to output probabilities for each class (e.g., Chickenpox, Cowpox, Monkeypox, etc.). The model is compiled using the **Adam** optimizer with a learning rate of 0.00001 set for fine-tuning, and categorical cross-entropy as the loss function since it’s a multi-class classification problem. Accuracy is used as the evaluation metric during training. This construction leverages DenseNet121’s transfer learning capabilities for robust feature extraction while allowing customization for the pox classification task.
### MobileNetV2
MobileNetV2, a lightweight, efficient deep learning model, particularly suited for mobile and edge device applications due to its lower computational requirements. Similar to DenseNet, pre-trained weights are loaded from ImageNet, top layers are modified, **Global Average Pooling** layer is added and **softmax activation** is applied on last layer. The model is compiled using the **Adam optimizer**, with a learning rate of 0.001, and **categorical cross-entropy** as the loss function to handle multi-class classification.

Here, input size is set to 224x224x3, and batch size is 16.
## Model Evaluation
Two models, DenseNet121 and MobileNetV2, are compared based on their performance. The dataset we used consists of 7532 images for training, 153 images for validation, and 64 images for testing.

For DenseNet121, the model achieved an accuracy of **80%** on the test set. This higher accuracy reflects the model's capability to extract robust features, owing to its densely connected layers and deep architecture, which are well-suited for complex image classification tasks. In contrast, MobileNetV2 achieved a slightly lower accuracy of **73%** on the test set. While MobileNetV2 is optimized for efficiency and is highly suitable for mobile and edge devices, its performance is slightly compromised compared to the deeper DenseNet121 model.

Both models perform reasonably well, but DenseNet121 shows an edge in accuracy, making it a better choice when higher classification precision is required.

## Requirements
1. Python 3.12
2. TensorFlow 2.x & Keras
3. PIL (Pillow)
4. Seaborn
5. NumPy
6. Pandas
7. Glob
8. SciPy
10. CUDA (Nvidia) or ANE Support (Apple)
## Contributors
Contributions are welcome! Feel free to open issues or submit pull requests.
## Licence
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
