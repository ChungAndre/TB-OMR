# TB-OMR: An End-to-End Optical Music Recognition System
### Bachelor's Thesis Project | March - October 2024

This repository contains the source code and research developed for my Bachelor's Thesis in Optical Music Recognition (OMR). The project implements an end-to-end deep learning model capable of transcribing printed sheet music images directly into a machine-readable symbolic format.

---

### Project Goal

The primary goal of this research was to develop and train a deep learning model that can learn to read a musical score image and output its symbolic representation, effectively automating the process of digitizing sheet music. The model is trained on the **PrIMuS dataset** and learns to predict sequences in both agnostic and semantic encodings.

---

### Key Features

*   **End-to-End Recognition:** Transcribes a full musical staff from an image into a symbolic sequence in one step.
*   **Deep Learning Architecture:** Implements a robust model (likely a CRNN: CNN + RNN) in TensorFlow/Keras for image feature extraction and sequential data processing.
*   **PrIMuS Dataset Integration:** Includes scripts for parsing, preprocessing, and generating batches from the PrIMuS (Printed Music Score) dataset.
*   **Modular Scripts:** Provides separate, clear scripts for `training`, `prediction`, and a `one-step_predict` demo.
*   **Agnostic & Semantic Support:** The model can be trained using different symbolic vocabularies as defined in the PrIMuS methodology.

---

### Technical Approach

The system employs an end-to-end, sequence-to-sequence learning approach. 

1.  **Input:** A standardized image of a single staff of printed music.
2.  **Feature Extraction:** A Convolutional Neural Network (CNN) backbone processes the image, learning to identify relevant visual features and transforming the image into a lower-dimensional feature map.
3.  **Sequence Processing:** The feature map is then fed into a Recurrent Neural Network (RNN), likely an LSTM or GRU layer, which reads the features sequentially (from left to right) and learns the temporal dependencies between musical symbols.
4.  **Output:** A final dense layer with a softmax activation function predicts the probability distribution over the vocabulary for each step in the sequence, which is then decoded to produce the final symbolic representation (e.g., in MEI or a custom agnostic format).

---

### Getting Started

#### 1. Prerequisites
*   Python 3.9 or higher
*   Git
*   A GPU is highly recommended for training.

#### 2. Installation
```bash
# Clone the repository
git clone https://github.com/ChungAndre/TB-OMR.git
cd TB-OMR

# Install the required Python packages
pip install -r requirements.txt## TB-OMR
```
#### 3. Data
This model is designed to work with the PrIMuS dataset. You will need to download it and place the relevant files (images and transcriptions) into the Data/ directory. The train.txt and test.txt files should contain the paths to the training and testing samples, respectively.

### Usage
The repository includes three main scripts for interacting with the model.
#### 1. Training a New Model
To train the model from scratch, use the training.py script. You will need to configure the paths to your dataset and vocabularies within the script or pass them as arguments.

#### Example command 
```bash
python training.py --data_path ./Data/ --epochs 50 --batch_size 16
```

#### 2. Making Predictions on a Test Set
To evaluate a trained model, use the predict.py script, which typically runs predictions on a predefined test set.

#### Example command
```bash
python predict.py --model_path ./Models/my_trained_model.h5 --test_file ./Data/test.txt
```

#### 3. One-Step Prediction (Demo)
The one_step_predict.py script is the easiest way to see the model in action on a single image.
```bash
python one_step_predict.py \
    -image path/to/your/music_score_image.png \
    -model ./Models/your_trained_model.h5 \
    -vocabulary ./Data/vocabulary_agnostic.txt
```
# Project Structure
```sh

TB-OMR/
├── Data/                   # Holds the dataset, vocabularies, and train/test splits.
├── Models/                 # Directory to save and load trained model checkpoints (.h5).
│
├── model.py                # Defines the Keras/TensorFlow model architecture (CNN+RNN).
├── training.py             # Script for training the model on the PrIMuus dataset.
├── predict.py              # Script for running batch predictions on a test set.
├── one_step_predict.py     # End-to-end script for transcribing a single image.
│
├── primus.py               # Core data loading and preprocessing logic specific to the PrIMuS dataset format.
├── utils.py                # General utility functions used across the project.
│
├── requirements.txt        # Python dependencies.
└── README.md               # This file.
```


# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
Special thanks to Prof. Dr. Rolf Ingold for supervising this project.
