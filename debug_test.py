import tensorflow as tf
from keras_tuner import HyperParameters
from model import CNNModel

print("--- Starting Debug Test ---")

# We are creating a dummy HyperParameters object, just like KerasTuner does.
hp = HyperParameters()

try:
    print("Attempting to call the model-building function directly...")
    
    # We will call the function exactly as the tuner's lambda function does.
    # If this line works, the function definition is correct.
    model = CNNModel.get_parameterized_autoencoder_model(hp, n_input_features=1)
    
    print("\n>>> SUCCESS! The function call worked without error. <<<")
    print("This proves your model.py file and the function definition are correct.")
    model.summary()

except TypeError as e:
    print("\n>>> FAILURE: The TypeError persists even in isolation. <<<")
    print(f"Error message: {e}")
    print("This is extremely unusual and suggests the changes to model.py are still not being recognized by the Python interpreter.")

except Exception as e:
    print(f"\n--- An unexpected error occurred --- \n{e}")