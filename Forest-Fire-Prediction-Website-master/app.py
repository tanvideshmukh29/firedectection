from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import re
import math
app = Flask(__name__)







@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    model = pickle.load(open('model.pkl', 'rb'))

    int_features=[int(x) for x in request.form.values()]

    final=[np.array(int_features)]

    print(int_features)

    print(final)

    prediction=model.predict_proba(final)

    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('result.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('result.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")





if __name__ == '__main__':
    app.run(debug=True)