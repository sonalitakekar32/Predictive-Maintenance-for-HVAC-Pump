#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[35]:


model=pickle.load(open('C:/Users/Lenovo/PMR.pkl','rb'))


# In[37]:


st.title('Random Forest Model Deployment')


# In[39]:


def user_input_parameters():
    sensor_00=st.sidebar.number_input('sensor_00')
    sensor_01=st.sidebar.number_input('sensor_01')
    sensor_02=st.sidebar.number_input('sensor_02')
    sensor_03=st.sidebar.number_input('sensor_03')
    sensor_04=st.sidebar.number_input('sensor_04')
    sensor_05=st.sidebar.number_input('sensor_05')
    sensor_06=st.sidebar.number_input('sensor_06')
    sensor_07=st.sidebar.number_input('sensor_07')
    sensor_08=st.sidebar.number_input('sensor_08')
    sensor_09=st.sidebar.number_input('sensor_09')
    sensor_10=st.sidebar.number_input('sensor_10')
    sensor_11=st.sidebar.number_input('sensor_11')
    sensor_12=st.sidebar.number_input('sensor_12')
    sensor_13=st.sidebar.number_input('sensor_13')
    sensor_14=st.sidebar.number_input('sensor_14')
    sensor_16=st.sidebar.number_input('sensor_16')
    sensor_17=st.sidebar.number_input('sensor_17')
    sensor_18=st.sidebar.number_input('sensor_18')
    sensor_19=st.sidebar.number_input('sensor_19')
    sensor_20=st.sidebar.number_input('sensor_20')
    sensor_21=st.sidebar.number_input('sensor_21')
    sensor_22=st.sidebar.number_input('sensor_22')
    sensor_23=st.sidebar.number_input('sensor_23')
    sensor_24=st.sidebar.number_input('sensor_24')
    sensor_25=st.sidebar.number_input('sensor_25')
    sensor_26=st.sidebar.number_input('sensor_26')
    sensor_27=st.sidebar.number_input('sensor_27')
    sensor_28=st.sidebar.number_input('sensor_28')
    sensor_29=st.sidebar.number_input('sensor_29')
    sensor_30=st.sidebar.number_input('sensor_30')
    sensor_31=st.sidebar.number_input('sensor_31')
    sensor_32=st.sidebar.number_input('sensor_32')
    sensor_33=st.sidebar.number_input('sensor_33')
    sensor_34=st.sidebar.number_input('sensor_34')
    sensor_35=st.sidebar.number_input('sensor_35')
    sensor_36=st.sidebar.number_input('sensor_36')
    sensor_37=st.sidebar.number_input('sensor_37')
    sensor_38=st.sidebar.number_input('sensor_38')
    sensor_39=st.sidebar.number_input('sensor_39')
    sensor_40=st.sidebar.number_input('sensor_40')
    sensor_41=st.sidebar.number_input('sensor_41')
    sensor_42=st.sidebar.number_input('sensor_42')
    sensor_43=st.sidebar.number_input('sensor_43')
    sensor_44=st.sidebar.number_input('sensor_44')
    sensor_45=st.sidebar.number_input('sensor_45')
    sensor_46=st.sidebar.number_input('sensor_46')
    sensor_47=st.sidebar.number_input('sensor_47')
    sensor_48=st.sidebar.number_input('sensor_48')
    sensor_49=st.sidebar.number_input('sensor_49')
    sensor_50=st.sidebar.number_input('sensor_50')
    sensor_51=st.sidebar.number_input('sensor_51')
    data={'sensor_00':sensor_00, 'sensor_01':sensor_01, 'sensor_02':sensor_02, 'sensor_03':sensor_03,'sensor_04':sensor_04,'sensor_05':sensor_05,'sensor_06':sensor_06,'sensor_07':sensor_07,'sensor_08':sensor_08,
         'sensor_09':sensor_09,
         'sensor_10':sensor_10,
         'sensor_11':sensor_11,
         'sensor_12':sensor_12,
         'sensor_13':sensor_13,
         'sensor_14':sensor_14,
         'sensor_16':sensor_16,
         'sensor_17':sensor_17,
         'sensor_18':sensor_18,
         'sensor_19':sensor_19,
         'sensor_20':sensor_20,
         'sensor_21':sensor_21,
         'sensor_22':sensor_22,
         'sensor_23':sensor_23,
         'sensor_24':sensor_24,
         'sensor_25':sensor_25,
         'sensor_26':sensor_26,
         'sensor_27':sensor_27,
         'sensor_28':sensor_28,
         'sensor_29':sensor_29,
         'sensor_30':sensor_30,
         'sensor_31':sensor_31,
         'sensor_32':sensor_32,
        'sensor_33':sensor_33,
         'sensor_34':sensor_34,
          'sensor_35':sensor_35,
          'sensor_36':sensor_36,
         'sensor_37':sensor_37,
         'sensor_38':sensor_38,
         'sensor_39':sensor_39,
         'sensor_40':sensor_40,
         'sensor_41':sensor_41,
         'sensor_42':sensor_42,
         'sensor_43':sensor_43,
         'sensor_44':sensor_44,
         'sensor_45':sensor_45,
         'sensor_46':sensor_46,
         'sensor_47':sensor_47,
         'sensor_48':sensor_48,
         'sensor_49':sensor_49,
         'sensor_50':sensor_50,
         'sensor_51':sensor_51,}    
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_parameters()
if st.button("Predict Machine Status"):
     predicted=model.predict(df)
     st.subheader('Predict Machine Status')
     st.write(predicted)
import nbconvert

converter = nbconvert.ScriptExporter()
body, _ = converter.from_filename("Mod_Dep_PMRF.ipynb")

with open("Mod_Dep_PMRF.py", "w", encoding="utf-8") as f:
    f.write(body)


# In[41]:


df


# In[ ]:




