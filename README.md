# Urdu News Classification with Transformers

This repository contains a project focused on classifying Urdu news articles into various categories using transformers such as DistilBERT and CamemBERT. The project includes preprocessing of the data, training of models, and evaluation using various metrics like accuracy, confusion matrix, and ROC AUC curve.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models Used](#models-used)
- [Training](#training)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The goal of this project is to classify Urdu news articles into multiple categories such as National News, Sports, Entertainment, Politics, etc., using state-of-the-art transformer models. The project demonstrates the process of data preprocessing, model training, and evaluation using transformer-based architectures.

## Dataset

The dataset consists of Urdu news articles categorized into 16 different classes:
- Terrorist attack
- National News
- Sports
- Entertainment
- Politics
- Fraud and Corruption
- Sexual Assault
- Weather
- Accidents
- Forces
- Inflation
- Murder and Death
- Education
- Law and Order
- Social Media
- Earthquakes

The data is provided in an Excel file, where each sheet represents a different category.

## Preprocessing

Preprocessing steps include:
- Tokenization and stop words removal specific to Urdu.
- Cleaning the text by removing special characters and punctuation marks.
- Converting text into sequences for model input.
- Vectorizing the text data using TF-IDF for traditional models.
- Tokenization using DistilBERT and CamemBERT tokenizers.

## Models Used

1. **DistilBERT**: A distilled version of BERT, optimized for efficiency and speed while retaining most of the accuracy.
2. **CamemBERT**: A transformer model specifically fine-tuned for the French language, used here due to its multilingual capabilities.

## Training

The training process involves:
- Splitting the data into training and testing sets.
- Vectorizing text data using tokenizers from the transformer models.
- Training the models using the provided data.
- Fine-tuning the models with learning rate schedulers and optimizers.

## Evaluation

The models are evaluated using:
- **Accuracy**: The overall accuracy of the model.
- **Confusion Matrix**: To visualize the performance of the model in predicting different classes.
- **ROC AUC Curve**: To evaluate the model's ability to distinguish between different classes.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/urdu-news-classification.git
   cd urdu-news-classification
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the appropriate directory.

## Usage

After setting up the environment, you can run the provided Python script to train and evaluate the models. Make sure to adjust the file paths accordingly.

To run the script, use the following command:

```bash
python urdu_news_classification.py
```

This will initiate the training process, followed by the evaluation of the model on the test data. The results, including accuracy, confusion matrix, and ROC AUC curves, will be displayed after the execution.

## Results

The results of the models will be displayed in terms of accuracy, confusion matrix, and ROC AUC curves. These results will help in understanding the performance of the models on the Urdu news classification task.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have any ideas for improvement.

## License

This project is licensed under the MIT License. 

## Contact

To access dataset used in this project! email us

- **Email**: [saad.naveed.dev@gmail.com]

