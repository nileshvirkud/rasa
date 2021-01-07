
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('C:\\Users\\niles\\OneDrive\\Learn\\rasa\\safebot\\predictions')
sys.path.append('C:\\Users\\niles\\OneDrive\\Learn\\rasa\\safebot\\actions')
# print(sys.path)

import pickle
import pandas as pd
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from capstone_preprocess import *
import os
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import pickle
from sklearn.decomposition import PCA


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

class Predictions:
        def __init__(self, data):
                self.data=data
        
        def predict(self):
                this_folder = os.path.dirname(os.path.abspath(__file__))
                # filename = os.path.join(this_folder,'IHMStefanini_industrial_safety_and_health_database_with_accidents_description_Dataset_SMOTE.csv')

                # df1 = pd.read_csv(filename)

                # df1.drop("Unnamed: 0", axis=1, inplace=True)

                # df1.rename(columns={'Data':'Date','Genre':'Gender','Industry Sector':'Industry','Accident Level':'Accident','Potential Accident Level':'Potential_Accident','Employee or Third Party':'Emp_Type','Critical Risk':'Critical Risk'},inplace=True)

                df2 = pd.DataFrame(self.data)

                # preddata= df1.append(df2,ignore_index=True)

                preddata = pd.concat([df2]*400).sort_index()

                filename = os.path.join(this_folder,'capstone_NLP_chatbot2.pkl')

                
                loaded_model = CustomUnpickler(open(filename, 'rb')).load()
                # loaded_model = pickle.load(open(filename, 'rb'))

                result = loaded_model.predict(preddata)

                accident_level = {0:'I',1 :'II',2:'III',3:'IV',4:'V',5:'VI'}


                return accident_level[result[0]]
# myvar = {
#         # "Unnamed: 0":0,
#         "Date" : ["2016-01-02 00:00:00"],
#         "Countries": ["Country_01"],
#         "Local":["Local_01"],
#         "Industry": ["Mining"],
#         "Potential_Accident":["I"],
#         "Gender": ["Male"],
#         "Emp_Type":  ["Employee"],
#         "Critical Risk":["Pressed"],
#         "Description":['While removing the drill rod of the Jumbo 08 for maintenance, the supervisor proceeds to loosen the support of the intermediate centralizer to facilitate the removal, seeing this the mechanic supports one end on the drill of the equipment to pull with both hands the bar and accelerate the removal from this, at this moment the bar slides from its point of support and tightens the fingers of the mechanic between the drilling bar and the beam of the jumbo.']
#         }

# print (__name__)

# p1 = Predictions(myvar)
# x= p1.predict()
# print(x)



# import pkgutil
# search_path = ['.'] # set to None to see all modules importable from sys.path
# all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
# print(all_modules)

