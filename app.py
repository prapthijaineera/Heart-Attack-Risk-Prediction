import pickle
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.title('Heart Attack Risk Classification')
#load model
model=pickle.load(open('gboost_model.pkl','rb'))

#scaling
scaler=StandardScaler()

#input features
Age=st.number_input('Age', min_value=20, max_value=100, value=25)
RestingBP=st.number_input('RestingBP', min_value=0, max_value=250, value=100)
Cholesterol=st.number_input('Cholesterol', min_value=0, max_value=650, value=100)
MaxHR=st.number_input('MaxHR', min_value=100, value=100)
Oldpeak=st.number_input('Oldpeak', min_value=-3, max_value=7, value=1)
FastingBS=st.selectbox('FastingBS',(1,0))
Gender=st.selectbox('Gender',('M','F'))
ChestPainType=st.selectbox('ChestPainType',('ATA','NAP','ASY','TA'))
ExerciseAngina=st.selectbox('ExerciseAngina',('Y','N'))
RestingECG=st.selectbox('RestingECG',('Normal','ST','LVH'))
ST_Slope=st.selectbox('ST_Slope',('Up','Flat','Down'))
gender=st.selectbox('gender',('M','F'))
#encoding
#sex, exercise angina
sex=1 if gender=='M' else 0
exerciseAngina=1 if ExerciseAngina=='Y' else 0

#Chestpain 
ChestPainType_ASY=1 if ChestPainType=='ASY' else 0
ChestPainType_ATA=1 if ChestPainType=='ATA' else 0
ChestPainType_NAP=1 if ChestPainType=='NAP' else 0
ChestPainType_TA=1 if ChestPainType=='TA' else 0

#RestingECG
RestingECG_LVH=1 if RestingECG=='LVH' else 0
RestingECG_Normal=1 if RestingECG=='Normal' else 0
RestingECG_ST=1 if RestingECG=='ST' else 0

#ST_Slope {'Up':0, 'Down':1, 'Flat':2}
st_Slop_dict={'Up':0, 'Down':1, 'Flat':2}
sT_Slope=st_Slop_dict[ST_Slope]

#define dataFrame
input_features=pd.DataFrame({'Age':[Age], 'RestingBP':[RestingBP], 'Cholesterol':[Cholesterol], 'FastingBS':[FastingBS], 'MaxHR':[MaxHR], 'Oldpeak':[Oldpeak],
        'sex':[sex], 'exerciseAngina':[exerciseAngina], 'RestingECG_LVH':[RestingECG_LVH],
       'RestingECG_Normal':[RestingECG_Normal], 'RestingECG_ST':[RestingECG_ST], 'ChestPainType_ASY':[ChestPainType_ASY],
       'ChestPainType_ATA':[ChestPainType_ATA], 'ChestPainType_NAP': [ChestPainType_NAP], 'ChestPainType_TA':[ChestPainType_TA],
       'st_Slope':[ST_Slope]})

#scaling
input_features[['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']]=scaler.fit_transform(input_features[['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']]) 

if st.button('Predict'):
  predictions=model.predict(input_features)
  if predictions==1:
    st.error('High Risk of Heart Attack')
  else:
    st.success('Low Risk of Heart Attack')