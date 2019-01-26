import functools
import pickle

from flask import Flask

import data
import model


app = Flask(__name__)

@functools.lru_cache()
def load_model(model_path=model.DEFAULT_FPATH):
    
    return model.load_model(model_path)

@app.route('/')
def hello():
    return about()

@app.route('/about')
def about():
    return 'This microservice gives you the estimated gender of a given name'

@app.route('/infer/<name>')
def infer(name):
    clf = load_model()

    parsed_name = data.infer_input_fn(name)
    label = clf.predict(parsed_name)
    label = data.parse_output(label)
    return label


if __name__ == '__main__':
    app.run()