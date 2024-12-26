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
topic_infer = topic_model.signatures["custom_inference"]

sentiment_model = tf.saved_model.load("./models/sentiment_model")
sentiment_infer = sentiment_model.signatures["custom_inference"]

TOPIC = ["LECTURER", "TRAINING_PROGRAM", "FACILITY", "OTHER"]

SENTIMENT = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def predict_topic(sentence):
    input_data = tf.constant([sentence])
    output = topic_infer(input_data)
    output_tensor = next(iter(output.values()))
    argmax = tf.math.argmax(output_tensor, axis=1).numpy()[0]
    return TOPIC[argmax]


def predict_sentiment(sentence):
    input_data = tf.constant([sentence])
    output = sentiment_infer(input_data)
    output_tensor = next(iter(output.values()))
    argmax = tf.math.argmax(output_tensor, axis=1).numpy()[0]
    return SENTIMENT[argmax]


if __name__ == "__main__":
    sentence = "tài liệu thiếu thốn"
    print("\n\nTopic:", predict_topic(sentence))
    print("Sentiment:", predict_sentiment(sentence))
