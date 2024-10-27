## TB-OMR

Repository for the code developed for the Bachelor's Thesis from March to October 2024 on Optical Music Recognition (OMR).

# Project Structure
```sh
.DS_Store
.gitignore
Data/
    Example/
        000051652-1_2_1.agnostic
        000051652-1_2_1.mei
        000051652-1_2_1.semantic
    test.txt
    train.txt
    vocabulary_agnostic.txt
    vocabulary_semantic.txt
LICENSE
model.py
Models/
one_step_predict.py
predict.py
primus.py
README.md
training.py
utils.py
```

## Getting Started

# Installation
1. Clone the repository:
```sh
git clone https://github.com/yourusername/TB-OMR.git
cd TB-OMR
```
3. Install the required packages:
```sh
pip install -r requirements.txt
```

# Usage
To run the one-step prediction script:
```sh
python one_step_predict.py -image path/to/image.png -model path/to/model.h5 -vocabulary path/to/vocabulary.txt
```

# File Descriptions
one_step_predict.py: Script for end-to-end music symbol recognition from an image.

model.py: Contains the model architecture.

predict.py: Script for making predictions using the trained model.

primus.py: Data processing and preparation script.

training.py: Script for training the model.

utils.py: Utility functions used across the project.


# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
Special thanks to Prof. Dr. Rolf Ingold for supervising this project.
