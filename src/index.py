import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import numpy

from tensorflow.keras import layers
from tensorflow.keras import losses


topic_model = tf.saved_model.load("./models/topic_model")

if __name__ == "__main__":
    print(list(topic_model.signatures.keys()))

    infer = topic_model.signatures["custom_inference"]

    # Example input data
    input_data = tf.constant(["gv"])  # Example input for a model with input_shape=(32,)
    output = infer(input_data)
    print("Inference output:", output)