import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are
from features import load_data

emotions_to_observe = ['Neutral', 'Happy','Angry']

#Load the data and extract features for each sound file from the RAVDESS dataset
x,y=load_data("C:/Users/afiya/Desktop/website/SpeechRecog/RAVDESSActor_*/*.wav",
emotions_to_observe)

#Initialize multilayer perceptron model (alpha: L2 regularization parameter,
# bath_size: Size of minibatches for stochastic optimizers deafault --> 200)
# max iter: Maximum number of iterations. The solver iterates until convergence or this number of iterations.
model=MLPClassifier(batch_size=256, max_iter=500)

#split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

#fit the model
model.fit(x_train,y_train)

#predict the emotion using testing features
y_pred=model.predict(x_test)

#calculate accuracy of predictions
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Accuracy of Model {accuracy*100}")
print(f"Accuracy of Random Guessing {1/len(emotions_to_observe)*100}")

pickle.dump(model, open("mlp_classifier-main.model", "wb"))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
print (matrix)