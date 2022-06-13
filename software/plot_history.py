import os
import pickle

import matplotlib.pyplot as plt

fileDirectory = os.path.dirname(os.path.abspath(__file__))
parentDirectory = os.path.dirname(fileDirectory)

DATA_DIR = os.path.join(parentDirectory, 'data/')  

# The Below code is loading the history of the model and plotting the accuracy and loss of the model.
with open(DATA_DIR + 'history.pkl', 'rb') as f:
    history = pickle.load(f)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
