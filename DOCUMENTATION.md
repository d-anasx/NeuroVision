## shuffle() 
— mixes the images and labels while keeping each pair together.
## prefetch() 
— prepares upcoming data while the model processes the current data.
## split_dataset() 
— separates the data into 80% training, 10% validation, and 10% testing.
## batch(32)
— groups 32 images together for each training step.


# CNN Architecture — NeuroVision AI

A Convolutional Neural Network (CNN) is a deep learning model designed to learn spatial patterns from images. In NeuroVision AI, the CNN processes brain MRI images of size 224 × 224 × 3 and classifies them into four categories: glioma, meningioma, notumor, and pituitary.

## Feature extraction

- **Convolution operation:** applies filters to local regions of the input, producing feature maps.
- **Kernel/filter:** a set of trainable weights that learns patterns such as edges, textures and more complex visual features.
- **Stride:** determines how many pixels a filter moves at each step.
- **Padding:** determines whether extra border pixels are added before convolution. Valid padding shrinks the spatial dimensions; same padding preserves them when stride is 1.
- **Conv2D:** performs the convolution using a specified number of filters, kernel size, stride and padding.
- **Activation function:** introduces nonlinearity. ReLU replaces negative values with zero and keeps positive values unchanged.
- **MaxPooling2D:** reduces spatial dimensions by retaining the maximum value within each pooling window.

## Classification

- **Flatten:** converts the final three-dimensional feature tensor into a one-dimensional vector.
- **Dense layer:** combines input features using trainable weights and biases. It is also called a fully connected layer because each neuron connects to every input value.
- **Dropout:** randomly disables a proportion of activations during training to help reduce overfitting.
- **Output layer:** uses four neurons and softmax to produce probabilities for the four target classes.
- **Classification:** selects the class associated with the highest predicted probability.

## Learning process

During training, the model performs forward propagation, calculates the loss, uses backpropagation to calculate gradients, and applies an optimizer such as Adam to update trainable weights and biases. This process repeats across batches and epochs.

The purpose of the CNN is to learn useful image features and combine them to distinguish the four classes. Its actual performance must be assessed using independent test data and appropriate classification metrics.