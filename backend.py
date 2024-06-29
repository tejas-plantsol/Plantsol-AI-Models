from flask import Flask
from flask_restful import Api, Resource, reqparse
import requests
import werkzeug

from transformers import AutoTokenizer, BartForConditionalGeneration
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()

MAX_WORDS = 100
MIN_WORDS = 50

summarizer = BartForConditionalGeneration.from_pretrained(
    "philschmid/distilbart-cnn-12-6-samsum"
)
tokenizer = AutoTokenizer.from_pretrained("philschmid/distilbart-cnn-12-6-samsum")

summarizer_kwargs = {
    "truncation": True,
    "max_length": MAX_WORDS,
    "min_length": MIN_WORDS,
}


new_model = tf.keras.models.load_model(
    "../ai/models/final/SAVED_MODELS/Efficient_Net/SavedModelFormat",
    custom_objects={"KerasLayer": hub.KerasLayer},
)


class News(Resource):
    def get(self):
        r = requests.get(
            'https://gnews.io/api/v4/search?q="plant disease"&lang=en&apikey=772385e2caa6152b36deef7f9d141409&expand=content&max=1'
        )
        articles = r.json()["articles"]
        summarized = []
        index = 0
        titles = set()
        for article in articles:
            try:
                if article["title"] not in titles:
                    curr = articles[index]
                    if article["content"] is not None:
                        inputs = tokenizer([article["content"]], return_tensors="pt")
                        summary_ids = summarizer.generate(inputs["input_ids"])
                        content = tokenizer.batch_decode(
                            summary_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]
                    else:
                        if article["description"] is not None:
                            content = article["description"]
                        else:
                            content = "N/A"
                    curr["content"] = content
                    summarized.append(curr)
                    titles.add(article["title"])
                index += 1
            except:
                pass

        return {"data": summarized}


class Predict(Resource):
    def post(self):
        parser.add_argument("file")
        args = parser.parse_args()

        file = args["file"]
        image = (
            np.array(
                [
                    tf.keras.utils.img_to_array(
                        tf.keras.utils.load_img(
                            file.replace("file://", "")
                            .replace("%2540", "%40")
                            .replace("%25", "%")
                        )
                    )
                ]
            )
            / 255
        )

        prediction = new_model.predict(image)
        pred_value = tf.argmax(prediction, axis=1).numpy()[0].item()
        return {"data": pred_value}


api.add_resource(News, "/news")
api.add_resource(Predict, "/predict")

if __name__ == "__main__":
    app.run(port=8000, debug=True)
