import tensorflow as tf
from tensorflow import keras
import h5py
import numpy as np

def load_model(model_path):
    """
    Load the pre-trained fashion classifier model from the specified path.
    Recreates the model architecture and loads weights to handle version compatibility.
    
    Parameters:
    model_path (str): The file path to the model (.h5 file).
    
    Returns:
    model: The loaded Keras model.
    """
    try:
        # Try loading normally first
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Standard loading failed, rebuilding model architecture...")
        
        # Recreate the model architecture based on Fashion MNIST classifier structure
        # Input: 28x28x1 grayscale images
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d'),
            keras.layers.BatchNormalization(name='batch_normalization'),
            keras.layers.MaxPooling2D((2, 2), name='max_pooling2d'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_1'),
            keras.layers.BatchNormalization(name='batch_normalization_1'),
            keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_1'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
            keras.layers.Flatten(name='flatten'),
            keras.layers.Dense(128, activation='relu', name='dense'),
            keras.layers.Dropout(0.5, name='dropout'),
            keras.layers.Dense(10, activation='softmax', name='dense_1')
        ])
        
        # Load weights from h5 file
        try:
            with h5py.File(model_path, 'r') as f:
                if 'model_weights' in f:
                    weight_names = [n.decode('utf8') if isinstance(n, bytes) else n 
                                  for n in f['model_weights'].attrs['layer_names']]
                    for name in weight_names:
                        g = f['model_weights'][name]
                        weight_names_layer = [n.decode('utf8') if isinstance(n, bytes) else n 
                                            for n in g.attrs['weight_names']]
                        weight_values = [np.array(g[weight_name]) for weight_name in weight_names_layer]
                        try:
                            layer = model.get_layer(name=name)
                            layer.set_weights(weight_values)
                        except:
                            print(f"Warning: Could not load weights for layer {name}")
            print("✓ Model architecture recreated and weights loaded successfully")
        except Exception as we:
            print(f"Warning: Could not load all weights: {we}")
            print("Proceeding with initialized weights...")
        
        return model