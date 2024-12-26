import tensorflow as tf

from fastapi import FastAPI


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


app = FastAPI()


@app.get("/predict/topic")
async def root(sentence: str):
    return {"topic": predict_topic(sentence)}


@app.get("/predict/sentiment")
async def root(sentence: str):
    return {"sentiment": predict_sentiment(sentence)}
