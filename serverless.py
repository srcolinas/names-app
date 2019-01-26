import functools
import pickle

import boto3
from flask import Flask

import data


BUCKET_NAME = 'srcolinas-names'
REGION_NAME = 'us-east-1'
models_PREFIX = 'models/'
S3 = boto3.resource('s3')


app = Flask(__name__)

@functools.lru_cache()
def load_model(s3_fpath="srcolinas-names/models/model.pkl"):
    
    bucket_name, key = s3_fpath.split('/', 1)
    response = S3.Object(bucket_name=bucket_name, key=key).get()
    return pickle.loads(response['Body'].read())

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