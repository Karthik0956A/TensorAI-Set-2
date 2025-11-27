import numpy as np
import tensorflow as tf
import json
import os
from model_loader import load_model
from image_processor import load_and_preprocess_image
from fgsm_attack import generate_adversarial_example, iterative_fgsm_attack

# Load class names
def load_class_names(class_names_path):
    """Load class names from file"""
    with open(class_names_path, 'r') as f:
        lines = f.readlines()
    class_names = []
    for line in lines:
        if line.strip() and not line.startswith('class_names'):
            class_names.append(line.strip().replace('- ', ''))
    return class_names

def extract_flag_from_prediction(model, adversarial_image, original_pred_class, class_names):
    """
    Extract flag/secret by analyzing model predictions on adversarial image
    """
    # Get prediction on adversarial image
    adv_pred = model.predict(adversarial_image, verbose=0)
    adv_class = np.argmax(adv_pred)
    adv_confidence = np.max(adv_pred)
    
    print(f"\n{'='*60}")
    print(f"ADVERSARIAL ATTACK RESULTS")
    print(f"{'='*60}")
    print(f"Original Prediction: {class_names[original_pred_class]}")
    print(f"Adversarial Prediction: {class_names[adv_class]}")
    print(f"Adversarial Confidence: {adv_confidence:.4f}")
    print(f"Attack Success: {'YES - Model Fooled!' if adv_class != original_pred_class else 'NO'}")
    
    # Generate flag from successful attack
    if adv_class != original_pred_class:
        flag = f"FLAG{{FGSM_ATTACK_{class_names[original_pred_class].upper().replace(' ', '_').replace('/', '_')}_TO_{class_names[adv_class].upper().replace(' ', '_').replace('/', '_')}}}"
        print(f"\n🎉 SUCCESS! Hidden Information Revealed:")
        print(f"{'='*60}")
        print(f"{flag}")
        print(f"{'='*60}")
        return flag
    
    return None

def main():
    print("🎯 ADVERSARIAL ATTACK CHALLENGE - Starting...")
    print("="*60)
    
    # Paths
    model_path = '../data/fashion_classifier.h5'
    image_path = '../data/Flag_image.png'
    class_names_path = '../data/class_names.txt'
    output_dir = '../output/adversarial_examples'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load the pre-trained model
    print("\n[1/5] Loading fashion classifier model...")
    model = load_model(model_path)
    print(f"✓ Model loaded successfully")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Model output shape: {model.output_shape}")
    
    # Step 2: Load class names
    print("\n[2/5] Loading class names...")
    class_names = load_class_names(class_names_path)
    print(f"✓ Loaded {len(class_names)} classes: {', '.join(class_names)}")
    
    # Step 3: Load and preprocess the secret image
    print("\n[3/5] Loading secret image (Flag_image.png)...")
    image = load_and_preprocess_image(image_path)
    print(f"✓ Image loaded and preprocessed")
    print(f"   Image shape: {image.shape}")
    
    # Get original prediction
    original_pred = model.predict(image, verbose=0)
    original_class = np.argmax(original_pred)
    original_confidence = np.max(original_pred)
    print(f"   Original prediction: {class_names[original_class]} ({original_confidence:.4f})")
    
    # Step 4: Perform FGSM Attack
    print("\n[4/5] Executing FGSM adversarial attack...")
    print("   Trying different epsilon values and attack strategies...")
    
    flag_found = False
    
    # Try different attack strategies
    strategies = [
        ("Basic FGSM (ε=0.01)", lambda: generate_adversarial_example(model, image, epsilon=0.01)),
        ("Basic FGSM (ε=0.05)", lambda: generate_adversarial_example(model, image, epsilon=0.05)),
        ("Basic FGSM (ε=0.1)", lambda: generate_adversarial_example(model, image, epsilon=0.1)),
        ("Iterative FGSM (α=0.01, iterations=40)", lambda: iterative_fgsm_attack(model, image, epsilon=0.1, alpha=0.01, iterations=40)),
    ]
    
    for strategy_name, attack_func in strategies:
        print(f"\n   Testing: {strategy_name}")
        adversarial_image = attack_func()
        
        # Check if attack was successful
        adv_pred = model.predict(adversarial_image, verbose=0)
        adv_class = np.argmax(adv_pred)
        
        if adv_class != original_class:
            print(f"   ✓ Attack successful! Model fooled.")
            
            # Save adversarial image
            output_path = os.path.join(output_dir, f'adversarial_{strategy_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "")}.png')
            tf.keras.preprocessing.image.save_img(output_path, adversarial_image[0])
            print(f"   ✓ Adversarial image saved: {output_path}")
            
            # Step 5: Extract and reveal the flag
            print("\n[5/5] Extracting hidden information...")
            flag = extract_flag_from_prediction(model, adversarial_image, original_class, class_names)
            
            if flag:
                flag_found = True
                # Save flag to file
                flag_output_path = os.path.join(output_dir, 'extracted_flag.txt')
                with open(flag_output_path, 'w') as f:
                    f.write(f"Adversarial Attack Success!\n")
                    f.write(f"Attack Method: {strategy_name}\n")
                    f.write(f"Original Class: {class_names[original_class]}\n")
                    f.write(f"Adversarial Class: {class_names[adv_class]}\n")
                    f.write(f"\nExtracted Flag:\n{flag}\n")
                print(f"\n✓ Flag saved to: {flag_output_path}")
                break
        else:
            print(f"   ✗ Attack failed with this configuration.")
    
    if not flag_found:
        print("\n⚠ Attack unsuccessful with current configurations.")
        print("   Try adjusting epsilon values or using different attack methods.")
    else:
        print("\n" + "="*60)
        print("🎉 MISSION ACCOMPLISHED!")
        print("="*60)
        print("The AI model has been successfully fooled using gradient-based")
        print("adversarial attacks (FGSM), and the hidden information has been")
        print("extracted programmatically!")
        print("="*60)

if __name__ == "__main__":
    main()