import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
import pickle


train_df = pd.read_csv(r'D:\code\AppliedML_assgn_01\train.csv')
X_train = train_df['text']
y_train = train_df['spam']

tfidf = TfidfVectorizer(stop_words='english')
train_tfidf = tfidf.fit_transform(X_train)

model = pickle.load(open(r'D:\code\AppliedML_assgn_03\model.pkl', 'rb'))

def score(text, model, threshold):
    propensity = model.predict_proba(tfidf.transform([text]))[0]
    desired_predictions = (model.predict_proba(tfidf.transform([text]))[:,1] >= threshold).astype(bool)

    return (bool(desired_predictions[0]), float(max(propensity)))