# Sentiment Analysis of Movie Reviews
<img src="https://github.com/mjmaher987/Sentiment-Analysis-Project/assets/77095635/1660c25a-fc39-4e59-8130-50eb9f9975eb" width="500" />

This project performs sentiment analysis on the Large Movie Review Dataset using various machine learning techniques.

## About
This project is related to the machine learning course project.
Here we are going to clean data, train, test and evaluate some models for sentiment analysis.
We use a lot of traditional and new models for training and testing.
Share any ideas or brainstorming if you have.

## Contributors
- Mohammad Javad Maheronnaghsh

## Data
The dataset contains 50,000 reviews from IMDB. There are 25,000 training reviews and 25,000 testing reviews. The sentiment labels are balanced between positive and negative.

The data is loaded and preprocessed by:

- Removing stopwords
- Converting to lowercase
- Removing punctuation
- Tokenizing

## Models
The following models are implemented and evaluated:

- Traditional ML Models
- Logistic Regression
- Decision Tree
- AdaBoost with Decision Tree base estimator
- Neural Network Models
- Simple feedforward network
- Deep network with dropout and regularization

## Word Embedding Models
- TF-IDF vectors
- FastText word embeddings trained in unsupervised mode

## Training
Models are trained on 80% of the dataset and validated on 10% for hyperparameter tuning.

The TensorFlow models use the Adam optimizer and sparse categorical crossentropy loss. They are trained for 30 epochs with a batch size of 32.

The FastText model is trained with supervised labeling on the training set sentences.

## Evaluation
All models are evaluated on the remaining 10% test set using accuracy and F1 score.

Classification reports are printed showing precision, recall and F1 for each sentiment class.

## Results
The deep neural network achieves the best performance with 63% test accuracy. Logistic regression, decision tree, and FastText also perform reasonably well.

In general, the neural network models outperform the traditional ML models. The word embedding models achieve better performance compared to pure TF-IDF vectors.

There is still room for improvement by using more advanced architectures, pretrained embeddings, and regularization techniques.

## Usage
The main scripts are:
```
train.py - Train a model
evaluate.py - Evaluate on test set
predict.py - Make predictions on new data
```
Example:
```
# Train FastText model
python train.py --model fasttext

# Evaluate LSTM model
python evaluate.py --model lstm

# Make predictions with ensemble
python predict.py --model ensemble
```

## References
- [Link 1](https://miro.medium.com/max/3260/1*8XIjunF2z6dmsVlkEuOUaw.png)
