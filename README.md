# SoundNet: Music Source Separation

## Overview
SoundNet is a deep learning-based project focused on music source separation, allowing the isolation of vocals and instrumentals from audio tracks. This tool can be used for applications such as karaoke creation, music remixing, and audio analysis.

## Features
- **State-of-the-art Model**: Powered by U-Net architecture.
- **Custom Dataset**: Trained on [MUSDB18](https://sigsep.github.io/datasets/musdb.html) for optimal performance.
- **Ease of Use**: Simplified input-output process for end-users.
- **Scalable**: Designed to handle diverse audio genres and formats.

## About the project - Problem Statement and IDEA
After some research, I found out that most of the present solutions that exist for the source separation problem were based on taking the Fourier Transform of the audio, studying the characteristic frequency spectrum of a particular source type, and then masking that particular band out. I was aware from my experience in computer vision already that U-Nets have been used for a few years for Image Segmentation problems, but could not find much work upon U-Nets being used over audio datasets. Since in a way thinking about source separation problem, I was segmenting the different frequency bands of the audio sample, hence I felt that U-Nets could be potentially useful for source separation problem too. Hence I decided to give my own novel and naive innovation an attempt.

## How it Works
1. **Data Preparation**:
   - Preprocessed audio tracks split into vocal and instrumental components.
2. **Model Training**:
   - Leveraging the U-Net architecture, the model learns to predict the individual components of a mixed audio track.
3. **Separation**:
   - Users input an audio file, and the model outputs separate vocal and instrumental tracks.

## Initial Challenges Faced

Initially, I attempted to train the ML model on the entire MUSDB dataset (a detailed explanation of the dataset is provided later). However, this effort encountered RAM memory limitations in the Kaggle notebooks. As a result, I had to restrict training and testing the U-Net model to just two songs, which, despite being a small subset, still contained millions of samples. These constraints also prevented me from expanding the model’s width significantly, limiting it to just 64 timesteps. This narrow width restricted the model’s ability to capture spatial information across time effectively. To put this in perspective, each second of a song contains around 10,000 samples, but my model could only process 64 samples at a time. It’s almost unimaginable for humans to identify separate sources with such limited temporal variation in sound.

The dataset comprised stereo audio, meaning the audio differed for the left and right ears. To reduce computational complexity while maintaining training stability, I chose to use mono audio instead of stereo. Specifically, I utilized the left channel of the audio files for training and the right channel for testing.

In the future, I aim to make the U-Net wider to better capture the time variations in the signal, thereby improving the model’s ability to separate audio sources more accurately.

## Input

The musdb18 consists of 150 songs of different styles along with the images of their constitutive objects.

It contains two folders, a folder with a training set: "train", composed of 100 songs, and a folder with a test set: "test", composed of 50 songs. Supervised approaches were used to train on the training set and test on both sets.

All files from the musdb18 dataset are encoded in the Native Instruments stems format (.mp4). It is a multitrack format composed of 5 stereo streams, each one encoded in AAC @256kbps. These signals correspond to:

0 - The mixture,

1 - The drums,

2 - The bass,

3 - The rest of the accompaniment,

4 - The vocals.

For each file, the mixture correspond to the sum of all the signals. All signals are stereophonic and encoded at 44.1kHz.

Dependency/ Library Used
numpy as np

pandas as pd

tensorflow as tf

keras

matplotlib

Python

Preprocessing the Input Dataset
Overall, the code processes the audio data from the mus_train dataset by dividing it into smaller chunks of 64 time stamps, pairs each chunk with the corresponding drums data, and stores them in a pandas DataFrame for further analysis or training in a machine learning model.

As the audio tracks are stereophonic, therefore the left channel of the audio tracks in mus_train has been used as train dataset and the right channel of the audio tracks has been used as validation set. To make the process less time consuming, the model has been trained for 2 tracks only.

## Constructing the Model

What is U-Net?
The U-Net model is a convolutional neural network (CNN) architecture commonly used for image segmentation tasks. The U-Net architecture derives its name from its U-shaped design, consisting of an encoding path (contracting path) and a decoding path (expansive path). The encoding path captures high-level features by progressively reducing the spatial resolution through convolutional and pooling layers. The decoding path, which is symmetric to the encoding path, upsamples the feature maps using deconvolutional layers to restore the spatial resolution.

The U-Net architecture effectively combines the advantages of a CNN's feature extraction capability with the ability to preserve spatial information. It has been widely adopted in various image segmentation tasks, such as medical image analysis (e.g., tumor segmentation, cell segmentation), semantic segmentation, and more recently, even in audio-based tasks like instrument separation.

### Model Constructed in this Project
* The input layer takes in audio data with variable-length sequences (shape: (None, 64)).

* The model starts with a series of convolutional layers (Conv1D) with increasing filters (16, 32, 64, 128, 256, 512), each followed by batch normalization and leaky ReLU activation.

* Then, the model performs upsampling (transposed convolution) using Conv1DTranspose layers to restore the spatial resolution.

* At each upsampling stage, skip connections are implemented by concatenating the upsampled features with the corresponding features from the encoding path using Concatenate layers.

* ReLU activation is applied after each concatenation.

* Dropout layers with a dropout rate of 0.5 are used after the first and third upsampling stages.

* Finally, a Conv1DTranspose layer with 1 filter is used to generate a mask, and the output layer applies element-wise multiplication (Multiply) between the input audio and the mask.

### Training the Model
* The loss function is specified as Mean Absolute Error ('mae'), which measures the average absolute difference between the predicted and target values.

* The optimizer chosen is Adam, with a learning rate of 1e-3.

* The training data (x_train and y_train) and validation data (x_val and y_val) are prepared by converting them into TensorFlow tensors. The df_train and df_test DataFrames are used to extract the respective input (x) and target (y) data.

* The epochs parameter is set to 40, indicating the number of times the model will iterate over the entire training dataset.

* The verbose parameter is set to 2 to display a detailed progress bar, providing information about the training and validation steps.

**The model is tested on the 11th track of mus_test.**
