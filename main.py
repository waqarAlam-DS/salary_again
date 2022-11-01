from logging import debug
from urllib import request
#from urllib import form
from flask import Flask,render_template,request
import joblib
import numpy as np
 
app=Flask(__name__ )
model=joblib.load('hiring_model.pkl')
@app.route('/')
def hello_world():
    return render_template('base.html')
@app.route('/contact')
def contact():
    return "welcome to my contact page"
@app.route('/feedback')
def feedback():
    return "welcome to my feedback page"
@app.route('/help')
def help():
    return "welcome to my help page"
@app.route('/predict',methods=['POST'])
def predict():
        exp=request.form.get('experience')
        score=request.form.get('test_score')
        interview_score=request.form.get('Interview_score')
        prediction=  model.predict(np.array([[int(exp),int(score),int(interview_score)]]))
        print(prediction)
        output=round(prediction[0],2)
        return render_template('base.html',prediction_test=f"Employee salary will be {output}")
        

if __name__== '__main__': #fixed syntax
      app.run(debug=True)