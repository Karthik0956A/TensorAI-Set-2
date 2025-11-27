import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def generate_adversarial_example(model, image, epsilon=0.01, target_class=None):
    """
    Generate an adversarial example using FGSM (Fast Gradient Sign Method).
    
    Parameters:
    model: The pre-trained Keras model
    image: The input image (preprocessed, shape: (1, H, W, C))
    epsilon: The perturbation magnitude
    target_class: If provided, performs targeted attack to this class
    
    Returns:
    adversarial_image: The perturbed adversarial image
    """
    # Get original prediction
    original_pred = model.predict(image, verbose=0)
    original_class = np.argmax(original_pred)
    
    # Convert to tensor
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Perform FGSM attack
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        
        if target_class is not None:
            # Targeted attack: minimize loss for target class
            target_label = tf.one_hot(target_class, depth=prediction.shape[-1])
            target_label = tf.reshape(target_label, prediction.shape)
            loss = -tf.keras.losses.categorical_crossentropy(target_label, prediction)
        else:
            # Untargeted attack: maximize loss for original class
            true_label = tf.one_hot(original_class, depth=prediction.shape[-1])
            true_label = tf.reshape(true_label, prediction.shape)
            loss = tf.keras.losses.categorical_crossentropy(true_label, prediction)
    
    # Get gradient
    gradient = tape.gradient(loss, image_tensor)
    
    # Create adversarial example
    signed_grad = tf.sign(gradient)
    adversarial_image = image_tensor + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image.numpy()


def iterative_fgsm_attack(model, image, epsilon=0.01, alpha=0.001, iterations=40, target_class=None):
    """
    Generate an adversarial example using iterative FGSM (I-FGSM).
    
    Parameters:
    model: The pre-trained Keras model
    image: The input image (preprocessed)
    epsilon: Maximum perturbation magnitude
    alpha: Step size for each iteration
    iterations: Number of iterations
    target_class: If provided, performs targeted attack
    
    Returns:
    adversarial_image: The perturbed adversarial image
    """
    # Get original prediction
    original_pred = model.predict(image, verbose=0)
    original_class = np.argmax(original_pred)
    
    adversarial_image = tf.identity(image)
    
    for i in range(iterations):
        adversarial_image = tf.Variable(adversarial_image)
        
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            prediction = model(adversarial_image)
            
            if target_class is not None:
                # Targeted attack
                target_label = tf.one_hot(target_class, depth=prediction.shape[-1])
                target_label = tf.reshape(target_label, prediction.shape)
                loss = -tf.keras.losses.categorical_crossentropy(target_label, prediction)
            else:
                # Untargeted attack
                true_label = tf.one_hot(original_class, depth=prediction.shape[-1])
                true_label = tf.reshape(true_label, prediction.shape)
                loss = tf.keras.losses.categorical_crossentropy(true_label, prediction)
        
        # Get gradient
        gradient = tape.gradient(loss, adversarial_image)
        
        # Update adversarial image
        signed_grad = tf.sign(gradient)
        adversarial_image = adversarial_image + alpha * signed_grad
        adversarial_image = tf.clip_by_value(adversarial_image, image - epsilon, image + epsilon)
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image.numpy()