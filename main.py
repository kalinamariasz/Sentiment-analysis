import re
import string
import nltk
import numpy as np
import pandas as pd
import keras_tuner
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from numpy import array
from sklearn.model_selection import KFold
from tensorflow.python.keras.saving.save import load_model
from unidecode import unidecode
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nlp = WordNetLemmatizer()
positive_lines = list()
negative_lines = list()
acc_per_fold = []
loss_per_fold = []

# NOTE: some function calls are commented as we only had to build the vocabulary, tune the hyperparameters and
# cross-validate only once.
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# Plot inspired from https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb
def plotgraph(history):
    history_dict = history.history
    history_dict.keys()
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def lemmatization(file):
    words = []
    for t in file:
        words.append(nlp.lemmatize(t))
    return " ".join(words)


# code taken from https://medium.com/analytics-vidhya/data-preparation-and-text-preprocessing-on-amazon-fine-food
# -reviews-7b7a2665c3f4


# this function is meant to get rid of any abbreviations
def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"<br />", " ", phrase)  # added to get rid of the line breaks

    return phrase

# Code inspired from https://www.kaggle.com/code/harshalgadhe/imdb-sentiment-classifier-97-accuracy-model/notebook
def word_cloud(reviews):
    plt.figure(figsize=(10, 10))
    sentiment_text = ' '.join(reviews)
    WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
    sentiment_words = WC.generate(sentiment_text)
    plt.imshow(sentiment_words, interpolation='bilinear')
    plt.show()


# source: https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/
def clean_doc(doc):
    # remove abbreviations and line breaks
    doc = unidecode(doc.lower())
    doc = decontracted(doc)
    # remove punctuation from each token
    for char in string.punctuation:
        doc = doc.replace(char, ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# we construct our dictionary using this function
def dictionary(file, dic):
    for word in file:
        if word not in dic:
            dic[word] = 1
        else:
            dic[word] = dic[word] + 1


# we use this function to store the dictionary as our vocabulary
def write_to_file(dictionary, name):
    dic = dict(sorted(dictionary.items(), reverse=True, key=lambda x: x[1]))
    for key in list(dic.keys()):
        with open(name + ".txt", "a") as f:
            print(key, '\t', file=f)
    f.close()


# this function is used to build the vocabulary
def build_vocabulary(text):
    dic = dict()
    # for our training set we chose our data deterministically;
    # we take the first 3000 positive reviews and the first 3000 negative reviews in order to build our set
    for i in range(0, 3000):
        file = text["review"][i]
        tokens = clean_doc(file)
        lemmas = []
        for w in tokens:
            w = nlp.lemmatize(w, get_wordnet_pos(w))
            lemmas.append(w)
        dictionary(lemmas, dic)
    for i in range(25002, 28002):
        file = text["review"][i]
        tokens = clean_doc(file)
        lemmas = []
        for w in tokens:
            w = nlp.lemmatize(w, get_wordnet_pos(w))
            lemmas.append(w)
        dictionary(lemmas, dic)
    # remove word with low occurrence to restrain our vocabulary
    min_frequency = 2
    final_dic = dict()
    for word in dic:
        if dic[word] >= min_frequency:
            final_dic[word] = dic[word]
    # we then store our vocabulary to use it later for training the network
    write_to_file(final_dic, "vocabulary")


# this function is used to deterministically build our input vector for training
# similar to the way we build our vocabulary
def process_train_data(text, vocab):
    for i in range(0, 3000):
        file = text["review"][i]
        # each review is pre-processed
        tokens = clean_doc(file)
        # then we only keep the words from our vocabulary
        tokens = [w for w in tokens if w in vocab]
        # and join them as a review again
        line = ' '.join(tokens)
        negative_lines.append(line)
    # the same process is done for the positive reviews
    for i in range(25002, 28002):
        file = text["review"][i]
        tokens = clean_doc(file)
        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        positive_lines.append(line)


# this function is similar to the function used to process the training data
# the process is identical, now we just take the next 1000 positive reviews + the next 1000 negative reviews
def process_test_data(text, vocab):
    for i in range(3001, 4001):
        textfile = text["review"][i]
        tokens = clean_doc(textfile)
        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        negative_lines.append(line)
    for i in range(28003, 29003):
        textfile = text["review"][i]
        tokens = clean_doc(textfile)
        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        positive_lines.append(line)


# function used to do a five-fold cross validation
def k_fold_cross_validation(inputs, targets):
    fold_no = 1
    k_fold = KFold(n_splits=5, shuffle=True)
    for train, test in k_fold.split(inputs, targets):
        # Define the model architecture
        network = models.Sequential()
        network.add(layers.Dense(units=160, activation='relu', input_shape=(features,)))
        network.add(layers.Dense(units=192, activation='relu'))
        network.add(layers.Dense(units=1, activation='sigmoid'))
        # Compile the model
        network.compile(loss='binary_crossentropy', optimizer=optimizers.Nadam(learning_rate=0.00016116), metrics=['accuracy'])
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        # Fit data to model
        history = network.fit(inputs[train], targets[train],
                              epochs=50,
                              verbose=2)
        # Generate generalization metrics
        scores = network.evaluate(inputs[test], targets[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {network.metrics_names[0]} of {scores[0]}; {network.metrics_names[1]} of '
            f'{scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        filepath = './saved_model/' + str(fold_no)
        network.save(filepath=filepath)
        # Increase fold number
        fold_no = fold_no + 1
    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


# Code inspired from https://keras.io/guides/keras_tuner/getting_started/
def build_model(hp):
    model = models.Sequential()
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation="relu"
            )
        )
    model.add(layers.Dense(units=1, activation="sigmoid"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=optimizers.Nadam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def tune_hyperparameters(x_train, y_train, x_test, y_test):
    build_model(keras_tuner.HyperParameters())
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=2
    )
    tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


if __name__ == '__main__':
    # load the document
    col_list = ["review", "sentiment"]
    text = pd.read_csv("IMDB Dataset.csv", usecols=col_list)
    # build_vocabulary(text)
    file = open("vocabulary.txt", 'r')
    # load the results
    vocabulary = file.read()
    file.close()
    vocabulary = vocabulary.split()
    vocabulary = set(vocabulary)
    process_train_data(text, vocabulary)
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    docs = negative_lines + positive_lines
    tokenizer.fit_on_texts(docs)
    # create the input vectors and output vectors for training
    x_train = tokenizer.texts_to_matrix(docs, mode='freq')
    y_train = array([0 for _ in range(3000)] + [1 for _ in range(3000)])
    # word_cloud(positive_lines)
    # word_cloud(negative_lines)
    # we clear the lists of positive/negative reviews in order to now load the test data
    positive_lines.clear()
    negative_lines.clear()
    docs.clear()
    # load the test data
    process_test_data(text, vocabulary)
    docs = negative_lines + positive_lines
    # create the input vectors and output vectors for training
    x_test = tokenizer.texts_to_matrix(docs, mode='freq')
    y_test = array([0 for _ in range(1000)] + [1 for _ in range(1000)])
    features = x_test.shape[1]
    # tune hyperparameters:
    # tune_hyperparameters(x_train, y_train, x_test, y_test)
    # perform the k-fold cross-validation
    # k_fold_cross_validation(x_train, y_train)
    network = models.Sequential()
    network.add(layers.Dense(units=160, activation='relu', input_shape=(features,)))
    network.add(layers.Dense(units=192, activation='relu'))
    network.add(layers.Dense(units=1, activation='sigmoid'))
    # Compile the model
    network.compile(loss='binary_crossentropy', optimizer=optimizers.Nadam(learning_rate=0.00016116),
                    metrics=['accuracy'])
    history = network.fit(x_train, y_train, validation_data=(x_test, y_test),
                             epochs=50)
    # plot the accuracy and loss over each iteration
    plotgraph(history)
    # check the accuracy with the test data
    loss, acc = network.evaluate(x_test,y_test, verbose = 0)
    print('Test Accuracy: %f' % (acc * 100))
