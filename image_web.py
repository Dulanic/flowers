import os
import json

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # load cached predictions from file
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)

    # prepare image paths and predictions for rendering in template
    image_paths = [img_path for img_path, _ in predictions]
    predicted_labels = [pred_label for _, pred_label in predictions]

    return render_template('index.html', image_paths=image_paths, predicted_labels=predicted_labels)

if __name__ == '__main__':
    app.run()
