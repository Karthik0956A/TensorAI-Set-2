# Adversarial Attack Project

## Overview
This project aims to demonstrate the implementation of adversarial attacks on a pre-trained fashion classification model using the Fast Gradient Sign Method (FGSM). The objective is to extract hidden information from a secret image by fooling the model into misclassifying it.

## Project Structure
```
adversarial-attack-project
├── src
│   ├── attack.py            # Main script to orchestrate the adversarial attack
│   ├── model_loader.py      # Functions to load the pre-trained model
│   ├── image_processor.py    # Functions to preprocess images
│   └── fgsm_attack.py       # Implementation of the FGSM attack
├── data
│   ├── fashion_classifier.h5 # Pre-trained model file
│   ├── Flag_image.png       # Secret image to be attacked
│   ├── class_names.txt      # Class names for the model
│   └── flag_data.json       # Additional metadata related to the flag
├── test_images              # Directory for storing test images
├── output
│   └── adversarial_examples  # Directory for saving generated adversarial examples
├── requirements.txt         # Python dependencies for the project
└── README.md                # Project documentation
```

## Setup Instructions
1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines
1. Ensure that the `fashion_classifier.h5` model and `Flag_image.png` are present in the `data` directory.
2. Run the main attack script:
   ```
   python src/attack.py
   ```
3. The generated adversarial examples will be saved in the `output/adversarial_examples` directory.

## Adversarial Attack Challenge
The goal of this project is to successfully implement an adversarial attack that reveals the hidden information in the `Flag_image.png` by fooling the model. The attack is executed using gradient-based methods, specifically FGSM.

## Acknowledgments
This project is inspired by research in adversarial machine learning and aims to provide insights into the vulnerabilities of AI models.