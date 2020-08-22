from flask import Flask
from flask_restplus import Api, Resource, fields, reqparse, marshal
import numpy
import tensorflow as tf
import json

tf.compat.v1.enable_eager_execution()

app = Flask(__name__)
app.config['RESTPLUS_MASK_SWAGGER'] = False
api = Api(app)

model = api.model('IrisClassification', {
    'classification': fields.String(description='The obtained classification for the Iris.'),
    'probability': fields.Float(description='The obtained probability for the Iris Classification.')
})

parser = reqparse.RequestParser()
parser.add_argument('sepal_length', required = True, type=float, help='Sepal length')
parser.add_argument('sepal_width', required = True, type=float, help='Sepal width')
parser.add_argument('petal_length', required = True, type=float, help='Petal length')
parser.add_argument('petal_width', required = True, type=float, help='Petal width')

ns = api.namespace('iris_classification', description='Classification of iris flower')

@ns.route('/')
class IrisClassification(Resource):
  @api.marshal_with(model, code=200, description='Represents an Iris Classification.')
  @api.expect(parser)
  def get(self):
    args = parser.parse_args()
    sepal_length = float(args['sepal_length'])
    sepal_width = float(args['sepal_width'])
    petal_length = float(args['petal_length'])
    petal_width = float(args['petal_width'])

    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    model = tf.keras.experimental.load_from_saved_model("./models")
    predict_data = tf.convert_to_tensor([[sepal_length, sepal_width, petal_length, petal_width]])

    predictions = model(predict_data)

    for i, logits in enumerate(predictions):
      class_idx = tf.argmax(logits).numpy()
      p = tf.nn.softmax(logits)[class_idx]
      name = class_names[class_idx]
      probability = float(p.numpy()*100)

    return {
      'classification': name,
      'probability': probability
    }