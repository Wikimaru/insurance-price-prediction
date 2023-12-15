import streamlit as st
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ClosedFormRegression:
    def __init__(self):
        self.weights = None  # model parameters θ

    def fit(self, x: np.ndarray, y: np.ndarray):
        bias = np.ones((x.shape[0], 1))  
        x = np.concatenate((bias, x), axis=1) 
        self.weights = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)  
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        bias = np.ones((x.shape[0], 1))  
        x = np.concatenate((bias, x), axis=1)  
        return np.dot(x, self.weights)  

    def score(self, x: np.ndarray, y: np.ndarray) -> np.float64:
        '''Calculate coefficient of determination (R^2)'''
        ss_mean = np.sum((y - y.mean()) ** 2)
        ss_fit = np.sum((y - self.predict(x)) ** 2)
        return (ss_mean - ss_fit) / ss_mean

Input = [18,"male",25,0,"no","southeast"]
age = Input[0]
sex = Input[1]
bmi = Input[2]
children = Input[3]
smoker = Input[4]
region = Input[5]
NiceTitle = st.title('โดนchargesราคาประกันเท่าไหร่ดี?')
AgeInput_column,BMIInput_column,ChildemInput_column,Nice_column=st.columns(4)
Input_column1,Input_column2,Input_column3,Nice_column2=st.columns(4)
NiceLoadBar = st.progress(0)
resultText = st.text('')
datafile= pd.read_csv("insurance.csv")
with AgeInput_column:
    AgeInput = st.number_input('Age',min_value=1,max_value=200,value=18)
    age = AgeInput
with BMIInput_column:
    BMIInput = st.number_input('BMI',min_value=1.0,max_value=200.0,value=20.0)
    bmi = BMIInput
with ChildemInput_column:
    ChildrenInput = st.number_input('Children',min_value=0,max_value=200)
    children = ChildrenInput
with Input_column1:
    Sex_input=st.radio("Sex",('male','female'))
    sex = Sex_input
with Input_column2:
    smoker_Input = st.radio("smoker?",('no','yes'))
    smoker = smoker_Input
with Input_column3:
    region_Input = st.radio("region",('northeast','northwest','southeast','southwest'))
    region = region_Input
with Nice_column2:
    NiceButton = st.button('Confirm')
if(NiceButton):
    NiceLoadBar.progress(0)
    newData = datafile.loc[datafile['region']==region]
    newData = newData.loc[datafile['sex']==sex]
    newData = newData.loc[datafile['smoker']==smoker]
    X=newData.iloc[:,[True,False,True,True,False,False,False]]
    Y=newData.iloc[:,[False,False,False,False,False,False,True]]
    seed = 42
    test_percent = 0.1
    split: list[np.ndarray] = train_test_split(X, Y, test_size=test_percent, random_state=seed)
    X_train, X_test, y_train, y_test = split
    def mse(y: np.ndarray, y_hat: np.ndarray):
        return ((y - y_hat).T.dot(y - y_hat) / len(y))[0, 0]
    model = ClosedFormRegression()
    model.fit(X_train, y_train)
    inp= np.array([[age,bmi,children]])
    NiceResult = model.predict(inp)
    time.sleep(1)
    for i in range(100):
        NiceLoadBar.progress(i+1)
        time.sleep(0.03)
    resultText.text("charges = "+"%.2f"%round(NiceResult[0][0],2)+" $")

