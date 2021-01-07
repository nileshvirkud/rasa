
import pickle
import pandas as pd
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from text_preprocess import *
# from text_preprocess import TemporalVariableEstimator ,ImputeEstimator ,ColumnsLableEncoder
from capstone_preprocess import *
# from actions import text_preprocess
import os
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import pickle
from sklearn.decomposition import PCA



df1 = pd.read_csv("IHMStefanini_industrial_safety_and_health_database_with_accidents_description_Dataset_SMOTE.csv")

df1.drop("Unnamed: 0", axis=1, inplace=True)

df1.rename(columns=
            {'Data':'Date','Genre':'Gender','Industry Sector':'Industry','Accident Level':'Accident','Potential Accident Level':'Potential_Accident','Employee or Third Party':'Emp_Type','Critical Risk':'Critical Risk'},inplace=True)

myvar = {
        # "Unnamed: 0":0,
        "Date" : ["2016-01-02 00:00:00"],
        "Countries": ["Country_01"],
        "Local":["Local_01"],
        "Industry": ["Mining"],
        "Potential_Accident":["I"],
        "Gender": ["Male"],
        "Emp_Type":  ["Employee"],
        "Critical Risk":["Pressed"],
        "Description":['While removing the drill rod of the Jumbo 08 for maintenance, the supervisor proceeds to loosen the support of the intermediate centralizer to facilitate the removal, seeing this the mechanic supports one end on the drill of the equipment to pull with both hands the bar and accelerate the removal from this, at this moment the bar slides from its point of support and tightens the fingers of the mechanic between the drilling bar and the beam of the jumbo.']
        }

df2 = pd.DataFrame(myvar)

data= df1.append(df2,ignore_index=True)

# print(df3.head())
this_folder = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(this_folder,'capstone_NLP_chatbot2.pkl')

# filename = 'capstone_NLP_chatbot.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
# print(loaded_model)


# ctext = remove_punctuation('I was ,hurt.')
# te = TemporalVariableEstimator()
result = loaded_model.predict(data)
print(result) 

# accident_level = {0:'I',1 :'II',2:'III',3:'IV',4:'V',5:'VI'}


