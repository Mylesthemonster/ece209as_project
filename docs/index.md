# Finger Writing With IMU Data

> Project for UCLA ECEM202A/CSM213A (Spring 2022)

## 🚩 Table of Contents

- [Team](#-team)
- [Packages](#-packages)
- [Why Why This Project?](#-why-this-project)
- [Features](#-features)
- [Overview](#-overview)
- [Usage](#-usage)
- [Submissions](#-submissions)


## 👥 Team

- **Myles Johnson**
- **Anchal Sinha**

## 📦 Packages

### Python Libraries To Install

| Package name (Linked to Installation Guide) | Description |
| --- | --- |
| [Tensorflow](https://www.tensorflow.org/install) | End-to-end open source platform for machine learning |
| [Numpy](https://numpy.org/install/) | Library used for working with arrays |
| [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) | Library for data manipulation and analysis |
| [Scikit-Learn](https://scikit-learn.org/stable/install.html) |  A robust library for machine learning  |
| [Levenshtein](https://github.com/maxbachmann/Levenshtein) | Library containing functions for fast computation of Levenshtein (edit) distance |
| [Matplotlib](https://matplotlib.org/stable/users/installing/index.html) | A plotting library |

## ❓ Why This Project?

- To expand upon the brilliant idea, brought about in the ViFin paper, of using a device like a smartwatch to give users more ways to interact with their other devices
- This application can play into many real world categories such as accessibility and even a new input format for typing/writing in mobile/vr applications 
- Majority of people carry around smart devices with many sensors and that data we can be utilized to implement a lot of cool features

## 🎨 Features

- [Viewer](https://github.com/nhn/tui.editor/tree/master/docs/en/viewer.md) : Supports a mode to display only markdown data without an editing area.
  
- [Internationalization (i18n)](https://github.com/nhn/tui.editor/tree/master/docs/en/i18n.md) : Supports English, Dutch, Korean, Japanese, Chinese, Spanish, German, Russian, French, Ukrainian, Turkish, Finnish, Czech, Arabic, Polish, Galician, Swedish, Italian, Norwegian, Croatian + language and you can extend.
  
- [Widget](https://github.com/nhn/tui.editor/tree/master/docs/en/widget.md) : This feature allows you to configure the rules that replaces the string matching to a specific `RegExp` with the widget node.

- [Custom Block](https://github.com/nhn/tui.editor/tree/master/docs/en/custom-block.md) : Nodes not supported by Markdown can be defined through custom block. You can display the node what you want through writing the parsing logic with custom block.

## 🐾 Overview

A generalization of main.py and the steps it takes:

### Data Collection State (Ciomment out by default)

- A TCP socket is opened to the watch, then the senor data is logged to ***raw_sensor_data.csv***, via [Sensorlog](http://sensorlog.berndthomas.net), and then the soocket is closed

```py
def collector():
watch_data_exists = os.path.exists(DATA_DIR + 'raw_sensor_data.csv')
if watch_data_exists:
    os.remove(DATA_DIR + 'raw_sensor_data.csv')

start_time = datetime.datetime.now()
#end time is 10 sec after the current time
end_time = start_time + datetime.timedelta(seconds=10)

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('192.168.68.124', 59554)
sock.connect(server_address)
try:
    print('Opening socket')
    while end_time > datetime.datetime.now():
        data = sock.recv(4096)
        with open(DATA_DIR + 'raw_sensor_data.csv', 'ab') as f:
            f.write(data)
finally:
    print('Closing socket')
    sock.close()
```

### Loading the Dataset

- This loads the data from the csv file, and then loads the data from the indexed numpy files. Splits data into Test Train and Validation datasets.
  
```py
def load_data(name):
    print('\nLoading Data....')
    data_record = pd.read_csv(DATA_RECORD_NAME, converters={'labels': pd.eval})
    data_record['data_path'] = data_record['data_path'].str.replace('\\', '/', regex=False) # update path syntax
    data_record['data_path'] = data_record['data_path'].str.replace('../train_data/numbers/', DATA_DIR, regex=False) # rename directory

    if name is not None: # filter dataset by name
        data_record = data_record[data_record['name'] == name]
    data = []
    labels = []
    for index, record in data_record.iterrows():
        raw_data = np.load(record['data_path']).astype(np.float32)
        raw_labels = np.array(record['labels']).astype(np.float32)
        data.append(raw_data)
        labels.append(raw_labels)

    data = np.array(data, dtype=object)
    labels = np.array(labels, dtype=object)

    data_train, data_rem, labels_train, labels_rem = train_test_split(data, labels, train_size=TRAIN_SIZE)
    data_val, data_test, labels_val, labels_test = train_test_split(data_rem, labels_rem, test_size=TEST_SIZE)

    data_train_length = np.array([len(e) for e in data_train]).astype(np.int64)
    labels_train_length = np.array([len(l) for l in labels_train]).astype(np.int64)
    data_train_chunked, labels_train_chunked, data_train_chunked_length, labels_train_chunked_length = \
        chunk_data(data_train, labels_train, data_train_length, labels_train_length)
    data_train_chunked, labels_train_chunked = pad_data(data_train_chunked, labels_train_chunked, data_train_chunked_length, labels_train_chunked_length)
    data_train, labels_train = pad_data(data_train, labels_train, data_train_length, labels_train_length)

    data_val_length = np.array([len(e) for e in data_val]).astype(np.int64)
    labels_val_length = np.array([len(l) for l in labels_val]).astype(np.int64)
    data_val_chunked, labels_val_chunked, data_val_chunked_length, labels_val_chunked_length = \
        chunk_data(data_val, labels_val, data_val_length, labels_val_length)
    data_val_chunked, labels_val_chunked = pad_data(data_val_chunked, labels_val_chunked, data_val_chunked_length, labels_val_chunked_length)
    data_val, labels_val = pad_data(data_val, labels_val, data_val_length, labels_val_length)

    data_test_length = np.array([len(e) for e in data_test]).astype(np.int64)
    labels_test_length = np.array([len(l) for l in labels_test]).astype(np.int64)
    data_test_chunked, labels_test_chunked, data_test_chunked_length, labels_test_chunked_length = \
        chunk_data(data_test, labels_test, data_test_length, labels_test_length)
    data_test_chunked, labels_test_chunked = pad_data(data_test_chunked, labels_test_chunked, data_test_chunked_length, labels_test_chunked_length)
    data_test, labels_test = pad_data(data_test, labels_test, data_test_length, labels_test_length)
    
    print(f'\nData Loaded & Partitioned, it took: {timedelta(seconds = (time.time() - start_time))}')

    return (data_train, labels_train, data_train_length, labels_train_length), \
        (data_val, labels_val, data_val_length, labels_val_length), \
        (data_test, labels_test, data_test_length, labels_test_length), \
        (data_train_chunked, labels_train_chunked, data_train_chunked_length, labels_train_chunked_length), \
        (data_val_chunked, labels_val_chunked, data_val_chunked_length, labels_val_chunked_length), \
        (data_test_chunked, labels_test_chunked, data_test_chunked_length, labels_test_chunked_length)
```

### Creating the Model

- Model Architecture contains an input layer, a GRU layer, and a Softmax layer
  
```py
def create_model(feature_size, num_classes, gru_size=64):
    print('\nCreating Model....')
    input_data = Input(name='input', shape=(None, feature_size), dtype='float32')
    labels = Input(name='labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    gru = GRU(gru_size, return_sequences=True, name='gru1')(input_data)
    dense1 = Dense(100, name='dense1')(gru)
    outputs = Dense(num_classes + 1, activation='softmax', name='output')(dense1)

    loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
    
    test_model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=[loss, outputs]
                  )

    optimizer = Adam()

    train_model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=[loss]
                  )

    train_model.compile(
        loss={'ctc': lambda y_true, y_pred: y_pred},

        optimizer=optimizer
    )

    train_model.summary()
    return train_model, test_model
```

### Model Training

- Takes in the model, training data, training labels, input length, label length validation data, validation labels, validation input length, validation label length, and callbacks. This then trains the model and returns the trained model. Also provides the user with plots of loss.

```py
def train(model, data, labels, input_length, label_length, data_val, labels_val, val_data_length, val_label_length, plot=False):
    inputs = {
        'input': data,
        'labels': labels,
        'input_length': input_length,
        'label_length': label_length
    }
    outputs = {'ctc': np.zeros([data.shape[0]])}

    validate_inputs = {
        'input': data_val,
        'labels': labels_val,
        'input_length': val_data_length,
        'label_length': val_label_length,
    }
    validate_outputs = {'ctc': np.zeros([data_val.shape[0]])}

    # Training
    print('\nTraining Model\n')
    history = model.fit(
        inputs, outputs,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=(validate_inputs, validate_outputs)
    )

    print(f'\nModel trained, it took: {timedelta(seconds = (time.time() - start_time))}')
    
    if plot:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        
    
    return model
```

### Saving the Model

- Saves the mode to a single HDF5 filer that contains:
  - the architecture of the model, allowing to re-create the model
  - the weights of the model
  - the training configuration (loss, optimizer)
  - the state of the optimizer, allowing to resume training exactly where you left off
  
```py
def save_model(model, path):
    model.save_weights(path)
    print("\nModel has been saved")
```

### Loading Model Weights

- Loads a single HDF5 file which will contain the weights of the model
  
```py
def load_model(model, path):
    model.load_weights(path, by_name=True)
```

### Model Testing

- A dictionary of the test inputs, the model is then evaluated using the
    test inputs. The results are then printed

```py
def test(model, data_test, labels_test, test_data_length, test_label_length):
    test_inputs = {
        'input': data_test,
        'labels': labels_test,
        'input_length': test_data_length,
        'label_length': test_label_length
    }
    
    return model_accuracy(model, test_inputs, labels_test, test_data_length, test_label_length)
```

### Converting Model to be Deployed***

- (CURRENTLY NOT FUNCTIONAL)
- `convert_model_coreml` takes in a path to a saved Keras model and converts it to a CoreML model
- The CoreML models can be deployed onto the apple watch to get live predictions off of the sensor data on-watch

```py
def convert_model_coreml(model_dir):
    model = keras.models.load_model(model_dir)
    coreml_model = ct.convert(model)
    
    fileDirectory = os.path.dirname(os.path.abspath(__file__))
    parentDirectory = os.path.dirname(fileDirectory)
    SAVEDMODEL_DIR = os.path.join(parentDirectory, 'software/saved_model/model.mlmodel') 
    
    coreml_model.save(SAVEDMODEL_DIR)
    
    print("TF Model has been converted top CoreML Model")
```

`***Limitation of the Apples CoreML machine learning platform is that it currently does not support CTC layers, explained in detail in presentation slides under 'Future Work / Improvements`

## Usage

Following the Overview above the entire code can be ran with:

```sh
python main.py
```

`**IMPORTANT** Dataset number folder containing the sample numpy arrays not included in this repo, the user will have to obtain that dataset or provide there own`

## Submissions

- [Final Presentation Slides](https://docs.google.com/presentation/d/1_H9WaXdZecpDAY4uhZGEy5Dg0js1neWY28Mcll32O2U/edit?usp=sharing)
- [Final Report](https://github.com/Mylesthemonster/ece209as_project/blob/main/docs/report.md)