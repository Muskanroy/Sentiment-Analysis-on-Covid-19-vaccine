2# -*- coding: utf-8 -*-
"""
Created on Thu May 20 18:39:55 2021

@author: muska
"""

from flask import Flask,request,render_template,jsonify
import requests
import pandas as pd
import numpy as np
import pickle 

app=Flask(__name__)
pickle_in=open('C:\\Users\\muska\\Downloads\\classifier_linear.pkl','rb')
classifier_linear =pickle.load(pickle_in)
pickle_vec=open('C:\\Users\\muska\\Downloads\\vectorizer.pkl','rb')
vectorizer=pickle.load(pickle_vec)

@app.route('/index',methods=['GET','POST'])
def index ():   
    results = {}
    if request.method == "POST":
        Tweet=request.form['Tweet']
        vectorize=vectorizer.transform([Tweet])
        pred=classifier_linear.predict(vectorize)
        percentage=classifier_linear.predict_proba(vectorize)
        answer=np.max(percentage)
        results={'result':pred[0],'score':answer}
        
        
    return render_template('index.html',results=results)    
  

    
    
    
if __name__ =='__main__':
    app.run(debug=True)
                           