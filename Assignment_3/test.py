import pandas as pd
import pickle
import xgboost
import sklearn
import requests
import app
from app import *
import os
from score import *

test_df = pd.read_csv(r'D:\code\AppliedML_assgn_01\test.csv')
X_test = test_df['text']
y_test = test_df['spam']

model = pickle.load(open(r'D:\code\AppliedML_assgn_03\model.pkl', 'rb'))

def test_score():
    threshold = 0.5

    prediction, propensity = score(X_test[0], model, threshold)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)

    assert type(X_test[10]) == str
    assert type(model) == xgboost.sklearn.XGBClassifier
    assert type(threshold) == float
    assert type(prediction) == bool
    assert type(propensity) == float

    assert (prediction in [0, 1]) == True

    assert (0 <= propensity <= 1) == True

    threshold = 0
    prediction, propensity = score(X_test[1], model, threshold)
    assert prediction == True
    prediction, propensity = score(X_test[10], model, threshold)
    assert prediction == True

    threshold = 1
    prediction, propensity = score(X_test[1], model, threshold)
    assert prediction == False
    prediction, propensity = score(X_test[10], model, threshold)
    assert prediction == False

    print('Spam text:')
    spam_text = 'Click here to get instant 75% discount'
    prediction, propensity = score(spam_text, model, threshold = 0.5)
    assert prediction == True

    print('Non-Spam Text')
    non_spam_text = 'Kindly contact me after office hours on the telephone'
    prediction, propensity = score(non_spam_text, model, threshold = 0.5)
    assert prediction == False


import requests
import subprocess
import time 
import unittest

class TestFlaskIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Launch Flask app using command line
        cls.flask_process = subprocess.Popen(['python', 'app.py']) 
        time.sleep(3)  # Give some time for the server to start

    def test_flask(self):
        # Test the response from the localhost endpoint
        data = {'text': 'Kindly let me know about the details about the meeting ASAP.'}
        response = requests.post('http://127.0.0.1:5000/score', json = data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json())
        self.assertIn('propensity', response.json())
        
    @classmethod
    def tearDownClass(cls):
        # Close Flask app using command line
        cls.flask_process.terminate()