import tensorflow.keras as keras
import coremltools as ct
import os 

def convert_model_coreml(model_dir):
    """
    `convert_model_coreml` takes in a path to a saved Keras model and converts it to a CoreML model
    
    :param model_dir: The directory where the model is saved
    """
    model = keras.models.load_model(model_dir)
    coreml_model = ct.convert(model)
    
    fileDirectory = os.path.dirname(os.path.abspath(__file__))
    parentDirectory = os.path.dirname(fileDirectory)
    SAVEDMODEL_DIR = os.path.join(parentDirectory, 'software/saved_model/model.mlmodel') 
    
    coreml_model.save(SAVEDMODEL_DIR)
    
    print("TF Model has been converted top CoreML Model")
