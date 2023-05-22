#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import pandas as pd
import streamlit as st 


# In[2]:


pickle_in = open("Customer_Personality_Analysis_Group5_V01.pkl","rb")
model=pickle.load(pickle_in)


# In[4]:


def Classifier_predict (Income,Customer_Retention,Age,Kids,Total_Spend):
    prediction=model.predict([[Income,Customer_Retention,Age,Kids,Total_Spend]])
    print(prediction)
    return prediction


# In[1]:


def main():
    st.title("Customer Personality Analysis Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Customer Personality Analysis Predictor </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Income = st.text_input("Income","")
    st.text("Please provide Income of the Customer")
    Customer_Retention = st.text_input("Customer_Retention","")
    st.text("Please provide from how many months customer is with store")   
    Age = st.text_input("Age","")
    st.text("Please provide Age of customer")    
    Kids = st.text_input("Kids","")
    Total_Spend = st.text_input("Total_Spend","")
    st.text("Please provide total spending of the customer ")
    result=""
    if st.button("Predict"):
        result = Classifier_predict (Income,Customer_Retention,Age,Kids,Total_Spend)
    st.success('The output is {}'.format(result))
    if st.button("For More Info regarding Prediction"):
        st.text("0 == Platinum Class")
        st.text("2 == Gold Class")
        st.text("1 == Sliver Class")
        st.text("3 == Bronze Class")        


# In[6]:


if __name__=='__main__':
    main()

