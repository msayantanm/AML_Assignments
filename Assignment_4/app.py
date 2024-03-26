from flask import Flask, request, jsonify
from score import *
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open(r'D:\code\AppliedML_assgn_03\model.pkl', 'rb'))

@app.route('/score', methods = ['POST'])
def main():
    prediction, propensity = score(request.json.get('text'), model, threshold = 0.5)
    return jsonify({'prediction': prediction, 'propensity': propensity})


if __name__ == '__main__':
    app.run(debug=True, port = 5000)