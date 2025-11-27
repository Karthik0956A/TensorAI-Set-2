def load_image(image_path, target_size=(28, 28)):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    # Load as grayscale for Fashion MNIST
    img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def preprocess_image(image_array):
    from numpy import expand_dims
    return expand_dims(image_array, axis=0)  # Add batch dimension

def load_and_preprocess_image(image_path):
    image_array = load_image(image_path)
    return preprocess_image(image_array)