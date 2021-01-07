# -*- coding: utf-8 -*-
"""
Capstone Preprocess Procedures
1. Text
2. date
3. imputer
4. lable encoder
"""

# Import packages
import nltk; nltk.download('wordnet'); nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('averaged_perceptron_tagger')

# Reference: https://github.com/sharmapratik88/AppliedNLPWorkshop/blob/master/HelperCodes/text_prep_config.py
appos = {"ain't": "am not", "aren't": "are not", "can't": "cannot", 
         "can't've": "cannot have", "'cause": "because", 
         "could've": "could have", "couldn't": "could not", 
         "couldn't've": "could not have", "didn't": "did not", 
         "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
         "hadn't've": "had not have", "hasn't": "has not", 
         "haven't": "have not", "he'd": "he would", "he'd've": "he would have", 
         "he'll": "he will", "he'll've": "he will have", 
         "he's": "he is", "how'd": "how did", 
         "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
         "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
         "I'll've": "I will have", "I'm": "I am", "I've": "I have", 
         "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
         "it'll": "it will", "it'll've": "it will have", "it's": "it is", 
         "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
         "might've": "might have", "mightn't": "might not", 
         "mightn't've": "might not have", "must've": "must have", 
         "mustn't": "must not", "mustn't've": "must not have", 
         "needn't": "need not", "needn't've": "need not have",
         "o'clock": "of the clock", "oughtn't": "ought not", 
         "oughtn't've": "ought not have", "shan't": "shall not", 
         "sha'n't": "shall not", "shan't've": "shall not have", 
         "she'd": "she would", "she'd've": "she would have", 
         "she'll": "she will", "she'll've": "she will have",
         "she's": "she is", "should've": "should have", 
         "shouldn't": "should not", "shouldn't've": "should not have", 
         "so've": "so have", "so's": "so is", 
         "that'd": "that had", "that'd've": "that would have", 
         "that's": "that that is", "there'd": "there would", 
         "there'd've": "there would have", "there's": "there is", 
         "they'd": "they would", "they'd've": "they would have", 
         "they'll": "they will", "they'll've": "they will have", 
         "they're": "they are", "they've": "they have", 
         "to've": "to have", "wasn't": "was not", "we'd": "we would", 
         "we'd've": "we would have", "we'll": "we will", 
         "we'll've": "we will have", "we're": "we are", 
         "we've": "we have", "weren't": "were not", 
         "what'll": "what will", "what'll've": "what will have", 
         "what're": "what are", "what's": "what is", 
         "what've": "what have", "when's": "when is", 
         "when've": "when have", "where'd": "where did", 
         "where's": "where is", "where've": "where have", 
         "who'll": "who will", "who'll've": "who will have", 
         "who's": "who is", "who've": "who have", 
         "why's": "why is", "why've": "why have", "will've": "will have", 
         "won't": "will not", "won't've": "will not have",
         "would've": "would have", "wouldn't": "would not", 
         "wouldn't've": "would not have", "y'all": "you all", 
         "y'all'd": "you all would", "y'all'd've": "you all would have", 
         "y'all're": "you all are", "y'all've": "you all have", 
         "you'd": "you would", "you'd've": "you would have",
         "you'll": "you will", "you'll've": "you will have", 
         "you're": "you are", "you've": "you have"}


# Helper function to replace appos
def replace_words(headline):
    cleaned_headlines = []
    for word in str(headline).split():
        if word.lower() in appos.keys():
            cleaned_headlines.append(appos[word.lower()])
        else:
            cleaned_headlines.append(word)
    return ' '.join(cleaned_headlines)

# Helper function to remove punctuations
# Reference: https://www.programiz.com/python-programming/methods/string/translate
PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' #string.punctuation
def remove_punctuation(text):
    """function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

# Helper function to lemmatize
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
  return ''.join([lemmatizer.lemmatize(word) for word in text])

# Helper function to remove stopwords
stoplist = set(stopwords.words('english'))
stoplist.remove('not')
def remove_stopwords(text):
    """function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in stoplist])

# Helper function for wordcloud
# Reference: https://www.kaggle.com/aashita/word-clouds-of-various-shapes
def plot_wordcloud(text, mask = None, max_words = 500, max_font_size = 40, 
                   figure_size = (12, 6), title = None, title_size = 15):
    wordcloud = WordCloud(background_color = 'white', max_words = max_words,
                          random_state = 42, width = 350, height = 150, 
                          mask = mask, stopwords = stoplist, collocations = False)
    wordcloud.generate(str(text))
    
    plt.figure(figsize = figure_size)
    plt.imshow(wordcloud, interpolation = 'bilinear');
    plt.title(title, fontdict = {'size': title_size, 'color': 'black', 
                               'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()

# Second helper function for lemmatizing
lemmatizer = WordNetLemmatizer()
def lem(text):
    pos_dict = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}
    return(' '.join([lemmatizer.lemmatize(w,pos_dict.get(t, wn.NOUN)) for w,t in nltk.pos_tag(text.split())]))

# Complete data clean
def desc_clean(df):
   #print('--'*30)
   #print('Converting headlines to lower case')
    df['cleaned_Description'] = df['Description'].apply(lambda x : x.lower())

   #print('Replacing apostrophes to the standard lexicons')
    df['cleaned_Description'] = df['cleaned_Description'].apply(lambda x : replace_words(x))

   #print('Removing punctuations')
    df['cleaned_Description'] = df['cleaned_Description'].apply(lambda x: remove_punctuation(x))

   #print('Removing Numbers')
    df['cleaned_Description'] = df['cleaned_Description'].apply(lambda x : ''.join([i for i in x if not i.isdigit()]))

   #print('Applying Lemmatizer')
    df['cleaned_Description'] = df['cleaned_Description'].apply(lambda x: lem(x))

   #print('Removing multiple spaces between words')
    df['cleaned_Description'] = df['cleaned_Description'].apply(lambda x: re.sub(' +', ' ', x))

   #print('Removing stopwords')
    df['cleaned_Description'] = df['cleaned_Description'].apply(lambda x: remove_stopwords(x))

    # remove short words (length < 3)
   #print('Removing short words')
    df['cleaned_Description'] = df['cleaned_Description'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
   #print('--'*30)
   
def clean_text(df, col):
	#print('--'*30); print('Converting headlines to lower case')
	new_col_name = "cleaned_" + str(col)
#	print(new_col_name)
	df.new_col_name= df[col].apply(lambda x : x.lower())
	#print(df.new_col_name[0])

	#print('Replacing apostrophes to the standard lexicons')
	df.new_col_name = df.new_col_name.apply(lambda x : replace_words(x))

	#print('Removing punctuations')
	df.new_col_name = df.new_col_name.apply(lambda x: remove_punctuation(x))

	#print('Removing Numbers')
	df.new_col_name = df.new_col_name.apply(lambda x : ''.join([i for i in x if not i.isdigit()]))

	#print('Applying Lemmatizer')
	df.new_col_name = df.new_col_name.apply(lambda x: lem(x))

	#print('Removing multiple spaces between words')
	df.new_col_name = df.new_col_name.apply(lambda x: re.sub(' +', ' ', x))

	#print('Removing stopwords')
	df.new_col_name = df.new_col_name.apply(lambda x: remove_stopwords(x))

	# remove short words (length < 3)
	#print('Removing short words')
	df.new_col_name = df.new_col_name.apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
	#print('--'*30)
	return(df.new_col_name)   



from sklearn.base import BaseEstimator,TransformerMixin
import pickle
from sklearn.decomposition import PCA

class ImputeEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass
    
    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self
    

    def transform(self, X):
            X = X.copy()
            ##print(X.info())
            ## Lable encoder
            industry = {'Metals':1,'Mining':2,'Others':99}
            X['Industry'] = pd.Series([industry[x] for x in X['Industry']], index=X.index)

            #print('Industry Encoded')
            
            risk_map = {'Not applicable': 99,
            'Bees': 1,
            'Venomous Animals': 1,
            'Blocking and isolation of energies': 3,
            'Burn': 3,
            'Confined space': 3,
            'Cut': 3,
            'Machine Protection': 3,
            'Manual Tools': 3,
            'Poll': 3,
            'Projection': 3,
            'Projection of fragments': 3,
            'Projection/Burning': 3,
            'Projection/Choco': 3,
            'Projection/Manual Tools': 3,
            'remains of choco': 3,
            'Suspended Loads': 3,
            'Fall': 4,
            'Fall prevention': 4,
            'Fall prevention (same level)': 4,
            'Electrical installation': 5,
            'Electrical Shock': 5,
            'Plates': 5,
            'Power lock': 5,
            'Chemical substances': 6,
            'Liquid Metal': 7,
            'Pressed': 8,
            'Pressurized Systems': 8,
            'Pressurized Systems / Chemical Substances': 8,
            'Individual protection equipment': 9,
            'Traffic': 10,
            'Vehicles and Mobile Equipment': 11,
            'Others': 99}

            X['Critical Risk'] = pd.Series([risk_map[x] for x in X['Critical Risk']], index=X.index)

            #print('Critical Risk Encoded')

            #Y = X.copy()
           #print("X['Description']:",X['Description'])
            #print("pd.DatetimeIndex(X):",pd.DatetimeIndex(X[self.variables]['Data']))
            X['Cleaned_Description'] = clean_text(X,"Description")
            X_desc=X['Cleaned_Description']
           #print("Cleaned_Description",X_desc.head())
            tokenizer = Tokenizer (num_words = 100)
            tokenizer.fit_on_texts(list(X_desc))
            X_desc = tokenizer.texts_to_sequences(X_desc)
           # print('Cleaned text Tokenized.')

           #print("Tokenized",X_desc)
            #max_len=max( X['cleaned_Description'].apply(lambda x: len(x.split(' '))))
            max_len=100
           #print("max_len",max_len)
            X_pad = pad_sequences(X_desc, maxlen = max_len)
            X_final = pd.DataFrame(X_pad)
  #         #print("padded",X_pad[0,:])
            #print(text_encoded.head())
            #X = X[X['Critical Risk'] == 99]
            #print(X.shape)
            riskpred_model = 'predict_risk.pkl'
            riskpred_model = pickle.load(open(riskpred_model, 'rb'))
            pca=PCA(n_components=45)
            X_processed_pca=pca.fit_transform(X_pad)
            #print("predictions",X[0])
           #print("predictions",X_processed_pca.shape)
            X['predicted_risk'] = riskpred_model.predict(X_processed_pca)
            X['predicted_risk'] = X.apply(lambda x: x['predicted_risk'] if x['Critical Risk']==99 else x['Critical Risk'], axis=1)
            #print("new",Y.info())
           #print(X.head())
          # #print('predicted  risk shape1',X_pred_risk.shape)            
            X_pred_risk = X['predicted_risk'].values
           #print('predicted  risk values shape1',X_pred_risk.shape)             
            X_pred_risk = X_pred_risk.reshape(X_pred_risk.shape[0],1)

           # print('Risk Category imputation complete.')
           #print('predicted  risk shape2',X_pred_risk.shape)
 #           pca=PCA(n_components=50)
            X_processed_pca=pca.fit_transform(X_pad)
##
           
            indpred_model = 'predict_industry.pkl'
            indpred_model = pickle.load(open(indpred_model, 'rb'))
            X['predicted_ind'] = indpred_model.predict(X_processed_pca)
            X['predicted_ind'] = X.apply(lambda x: x['predicted_ind'] if x['Industry']==99 else x['Industry'], axis=1)
           #print('predicted ind',X['predicted_ind'])
            #print("new",Y.info())
           #print(X.tail(20))
            X_pred_ind = X['predicted_ind'].values
            X_pred_ind = X_pred_ind.reshape(X_pred_ind.shape[0],1)

           # print('Industry imputation complete.')

           #print('predicted shape',X_pred_ind.shape)
           ##print('padded X_pad',X_pad.type())
            X_final['pred_risk'] = X_pred_risk
            X_final['pred_ind'] = X_pred_ind
            X = X_final
           #print(X_final.shape)
            return X


from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
class TemporalVariableEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
       #print("X.shape:",X.shape)
       #print("X.head:",X.head())
       #print("[self.variables :",self.variables)
       #print("X[self.variables]['Date']",X[self.variables]['Date'])
       #print("X['Date']:",X['Date'])
       #print("pd.DatetimeIndex(X):",pd.DatetimeIndex(X[self.variables]['Date']))
        X['month'] = pd.DatetimeIndex(X[self.variables]['Date']).month
        X['year'] = pd.DatetimeIndex(X[self.variables]['Date']).year
        X['day'] = pd.DatetimeIndex(X[self.variables]['Date']).day
#        X['dayname'] = pd.DatetimeIndex(X[self.variables]['Date']).day_name()
        X['weekofyear'] = pd.Int64Index(pd.DatetimeIndex(X[self.variables]['Date']).isocalendar().week)  
       #print("###############################X.shape:",X.shape)
#        enc_attribs=["month", "year", "day","dayname","weekofyear"]
        enc_attribs=["month", "year", "day","weekofyear"]
        X=X.loc[:,enc_attribs]
       #print("enc Dateframe chk",X.head())
        ##X=X.drop('Date',inplace=True)
        #print("df columns",X.columns())
#        one_hot_pipeline= Pipeline([('one_hot_encoder',OneHotEncoder(drop='first'))])
#        X=one_hot_pipeline.fit_transform(X)
       #print("###############################X.shape:",X.shape)
 #      #print("###############################X.head:",X[0,:])
        #print('Date transformed.')
        return X
        

from sklearn.base import BaseEstimator,TransformerMixin
import pickle
from sklearn.decomposition import PCA

class ImputeEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass
    
    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self
    

    def transform(self, X):
            X = X.copy()
            ##print(X.info())
            ## Lable encoder
            industry = {'Metals':1,'Mining':2,'Others':99}
            X['Industry'] = pd.Series([industry[x] for x in X['Industry']], index=X.index)

            #print('Industry Encoded')
            
            risk_map = {'Not applicable': 99,
            'Bees': 1,
            'Venomous Animals': 1,
            'Blocking and isolation of energies': 3,
            'Burn': 3,
            'Confined space': 3,
            'Cut': 3,
            'Machine Protection': 3,
            'Manual Tools': 3,
            'Poll': 3,
            'Projection': 3,
            'Projection of fragments': 3,
            'Projection/Burning': 3,
            'Projection/Choco': 3,
            'Projection/Manual Tools': 3,
            'remains of choco': 3,
            'Suspended Loads': 3,
            'Fall': 4,
            'Fall prevention': 4,
            'Fall prevention (same level)': 4,
            'Electrical installation': 5,
            'Electrical Shock': 5,
            'Plates': 5,
            'Power lock': 5,
            'Chemical substances': 6,
            'Liquid Metal': 7,
            'Pressed': 8,
            'Pressurized Systems': 8,
            'Pressurized Systems / Chemical Substances': 8,
            'Individual protection equipment': 9,
            'Traffic': 10,
            'Vehicles and Mobile Equipment': 11,
            'Others': 99}

            X['Critical Risk'] = pd.Series([risk_map[x] for x in X['Critical Risk']], index=X.index)

            #print('Critical Risk Encoded')

            #Y = X.copy()
           #print("X['Description']:",X['Description'])
            #print("pd.DatetimeIndex(X):",pd.DatetimeIndex(X[self.variables]['Data']))
            X['Cleaned_Description'] = clean_text(X,"Description")
            X_desc=X['Cleaned_Description']
           #print("Cleaned_Description",X_desc.head())
            tokenizer = Tokenizer (num_words = 100)
            tokenizer.fit_on_texts(list(X_desc))
            X_desc = tokenizer.texts_to_sequences(X_desc)
           # print('Cleaned text Tokenized.')

           #print("Tokenized",X_desc)
            #max_len=max( X['cleaned_Description'].apply(lambda x: len(x.split(' '))))
            max_len=100
           #print("max_len",max_len)
            X_pad = pad_sequences(X_desc, maxlen = max_len)
            X_final = pd.DataFrame(X_pad)
  #         #print("padded",X_pad[0,:])
            #print(text_encoded.head())
            #X = X[X['Critical Risk'] == 99]
            #print(X.shape)
            riskpred_model = 'predict_risk.pkl'
            riskpred_model = pickle.load(open(riskpred_model, 'rb'))
            pca=PCA(n_components=45)
            X_processed_pca=pca.fit_transform(X_pad)
            #print("predictions",X[0])
           #print("predictions",X_processed_pca.shape)
            X['predicted_risk'] = riskpred_model.predict(X_processed_pca)
            X['predicted_risk'] = X.apply(lambda x: x['predicted_risk'] if x['Critical Risk']==99 else x['Critical Risk'], axis=1)
            #print("new",Y.info())
           #print(X.head())
          # #print('predicted  risk shape1',X_pred_risk.shape)            
            X_pred_risk = X['predicted_risk'].values
           #print('predicted  risk values shape1',X_pred_risk.shape)             
            X_pred_risk = X_pred_risk.reshape(X_pred_risk.shape[0],1)

           # print('Risk Category imputation complete.')
           #print('predicted  risk shape2',X_pred_risk.shape)
 #           pca=PCA(n_components=50)
            X_processed_pca=pca.fit_transform(X_pad)
##
           
            indpred_model = 'predict_industry.pkl'
            indpred_model = pickle.load(open(indpred_model, 'rb'))
            X['predicted_ind'] = indpred_model.predict(X_processed_pca)
            X['predicted_ind'] = X.apply(lambda x: x['predicted_ind'] if x['Industry']==99 else x['Industry'], axis=1)
           #print('predicted ind',X['predicted_ind'])
            #print("new",Y.info())
           #print(X.tail(20))
            X_pred_ind = X['predicted_ind'].values
            X_pred_ind = X_pred_ind.reshape(X_pred_ind.shape[0],1)

           # print('Industry imputation complete.')

           #print('predicted shape',X_pred_ind.shape)
           ##print('padded X_pad',X_pad.type())
            X_final['pred_risk'] = X_pred_risk
            X_final['pred_ind'] = X_pred_ind
            X = X_final
           #print(X_final.shape)
            return X


from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd


class ColumnsLableEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
       
        country = {'Country_01':1,'Country_02':2,'Country_03':3}
      
        local = { 'Local_01': 1,
                  'Local_02': 2,
                  'Local_03': 3,
                  'Local_04': 4,
                  'Local_05': 5,
                  'Local_06': 6,
                  'Local_07': 7,
                  'Local_08': 8,
                  'Local_09': 9,
                  'Local_10': 10,
                  'Local_11': 11,
                  'Local_12': 12 }

        emp = {
                  'Third Party': 1,
                  'Employee': 2,
                  'Third Party (Remote)': 3
        }
        
        risk_grade = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6}        
        

        X['Countries'] = pd.Series([country[x] for x in X['Countries']], index=X.index)
        #print('Countries Encoded')

        X['Local'] = pd.Series([local[x] for x in X['Local']], index=X.index)
        #print('Local Encoded')

        X['Emp_Type'] = pd.Series([emp[x] for x in X['Emp_Type']], index=X.index)
        #print('Emp_type Encoded')

        X['Potential_Accident'] = pd.Series([risk_grade[x] for x in X['Potential_Accident']], index=X.index)
        #print('Potential Accident Encoded')

       # X['Gender'] = pd.Series([country[x] for x in X['Gender']], index=X.index)
      #  X=X.loc[:,enc_attribs]
       #print("enc Dateframe chk",X.head())
       #print("###############################X.shape:",X.shape)
       ##print("###############################X.head:",X[0,:])
        return X



                                    ###########################################
                                    ##### LSTM preprocess procedures###########
                                    ###########################################
from sklearn.base import BaseEstimator,TransformerMixin
import pickle
from sklearn.decomposition import PCA

class TextEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass
    
    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self
    

    def transform(self, X):
        X = X.copy()
        #print("X['Description']:",X['Description'])
        X['Cleaned_Description'] = clean_text(X,"Description")
        X_desc=X['Cleaned_Description']
        #print("Cleaned_Description",X_desc.head())
        tokenizer = Tokenizer (num_words = 2000)
        tokenizer.fit_on_texts(list(X_desc))
        #print ("#############tokenizer.word_index",tokenizer.word_index)
        X_desc = tokenizer.texts_to_sequences(X_desc)
        #print("Tokenized",X_desc)
        max_len=100
        #print("max_len",max_len)
        X_pad = pad_sequences(X_desc, maxlen = max_len)
        X = pd.DataFrame(X_pad)
        #print("X.shape **********",X.shape)
        #print(X.head(2))
        return X
            


from sklearn.base import BaseEstimator,TransformerMixin
import pickle
from sklearn.decomposition import PCA

class OnlyImputeEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass
    
    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self
    

    def transform(self, X):
            X = X.copy()
            ##print(X.info())
            ## Lable encoder
            industry = {'Metals':1,'Mining':2,'Others':99}
            X['Industry'] = pd.Series([industry[x] for x in X['Industry']], index=X.index)

            #print('Industry Encoded')
            
            risk_map = {'Not applicable': 99,
            'Bees': 1,
            'Venomous Animals': 1,
            'Blocking and isolation of energies': 3,
            'Burn': 3,
            'Confined space': 3,
            'Cut': 3,
            'Machine Protection': 3,
            'Manual Tools': 3,
            'Poll': 3,
            'Projection': 3,
            'Projection of fragments': 3,
            'Projection/Burning': 3,
            'Projection/Choco': 3,
            'Projection/Manual Tools': 3,
            'remains of choco': 3,
            'Suspended Loads': 3,
            'Fall': 4,
            'Fall prevention': 4,
            'Fall prevention (same level)': 4,
            'Electrical installation': 5,
            'Electrical Shock': 5,
            'Plates': 5,
            'Power lock': 5,
            'Chemical substances': 6,
            'Liquid Metal': 7,
            'Pressed': 8,
            'Pressurized Systems': 8,
            'Pressurized Systems / Chemical Substances': 8,
            'Individual protection equipment': 9,
            'Traffic': 10,
            'Vehicles and Mobile Equipment': 11,
            'Others': 99}

            X['Critical Risk'] = pd.Series([risk_map[x] for x in X['Critical Risk']], index=X.index)

            #print('Critical Risk Encoded')

            #Y = X.copy()
           #print("X['Description']:",X['Description'])
            #print("pd.DatetimeIndex(X):",pd.DatetimeIndex(X[self.variables]['Data']))
            X['Cleaned_Description'] = clean_text(X,"Description")
            X_desc=X['Cleaned_Description']
           #print("Cleaned_Description",X_desc.head())
            tokenizer = Tokenizer (num_words = 100)
            tokenizer.fit_on_texts(list(X_desc))
            X_desc = tokenizer.texts_to_sequences(X_desc)
            #print('Cleaned text Tokenized.')

           #print("Tokenized",X_desc)
            #max_len=max( X['cleaned_Description'].apply(lambda x: len(x.split(' '))))
            max_len=100
           #print("max_len",max_len)
            X_pad = pad_sequences(X_desc, maxlen = max_len)
            X_final = pd.DataFrame(X_pad)
  #         #print("padded",X_pad[0,:])
            #print(text_encoded.head())
            #X = X[X['Critical Risk'] == 99]
            #print(X.shape)
            riskpred_model = 'predict_risk.pkl'
            riskpred_model = pickle.load(open(riskpred_model, 'rb'))
            pca=PCA(n_components=45)
            X_processed_pca=pca.fit_transform(X_pad)
            #print("predictions",X[0])
           #print("predictions",X_processed_pca.shape)
            X['predicted_risk'] = riskpred_model.predict(X_processed_pca)
            X['predicted_risk'] = X.apply(lambda x: x['predicted_risk'] if x['Critical Risk']==99 else x['Critical Risk'], axis=1)
            #print("new",Y.info())
           #print(X.head())
          # #print('predicted  risk shape1',X_pred_risk.shape)            
            X_pred_risk = X['predicted_risk'].values
           #print('predicted  risk values shape1',X_pred_risk.shape)             
            #X_pred_risk = X_pred_risk.reshape(X_pred_risk.shape[0],1)

            print('Risk Category imputation complete.')
            #print('predicted  risk shape2',X_pred_risk.shape)
            #pca=PCA(n_components=50)
            X_processed_pca=pca.fit_transform(X_pad)
##
           
            indpred_model = 'predict_industry.pkl'
            indpred_model = pickle.load(open(indpred_model, 'rb'))
            X['predicted_ind'] = indpred_model.predict(X_processed_pca)
            X['predicted_ind'] = X.apply(lambda x: x['predicted_ind'] if x['Industry']==99 else x['Industry'], axis=1)
           #print('predicted ind',X['predicted_ind'])
            #print("new",Y.info())
           #print(X.tail(20))
            X_pred_ind = X['predicted_ind'].values
            #X_pred_ind = X_pred_ind.reshape(X_pred_ind.shape[0],1)

            #print('Industry imputation complete.')

           #print('predicted shape',X_pred_ind.shape)
           ##print('padded X_pad',X_pad.type())
            X_final=pd.DataFrame()
            #print('X_pred_risk shape',X_pred_risk.shape)
            #print('X_pred_ind shape',X_pred_ind.shape)
            X_final['pred_risk'] = X_pred_risk
            X_final['pred_ind'] = X_pred_ind
            X = X_final
           #print(X_final.shape)
            return X



from sklearn.base import BaseEstimator,TransformerMixin
import pickle
from sklearn.decomposition import PCA

class LstmModelPredictions(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass
    
    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self
    

    def transform(self, X):
        X = X.copy()
        #print("X['Description']:",X['Description'])
        X['Cleaned_Description'] = clean_text(X,"Description")
        X_desc=X['Cleaned_Description']
        #print("Cleaned_Description",X_desc.head())
        tokenizer = Tokenizer (num_words = 2000)
        tokenizer.fit_on_texts(list(X_desc))
        #print ("#############tokenizer.word_index",tokenizer.word_index)
        X_desc = tokenizer.texts_to_sequences(X_desc)
        #print("Tokenized",X_desc)
        max_len=100
        #print("max_len",max_len)
        X_pad = pad_sequences(X_desc, maxlen = max_len)
        #Import the lstm pred model
        from keras.models import model_from_json
        json_file = open('lstm_pred_model.json', 'r')
        loaded_lstm_pred_model_json = json_file.read()
        json_file.close()
        loaded_lstm_pred_model = model_from_json(loaded_lstm_pred_model_json)
        # load weights into new lstm_pred_model
        loaded_lstm_pred_model.load_weights("lstm_pred_model.h5")
        print("Loaded lstm_pred_model from disk")
        lstm_predictions = loaded_lstm_pred_model.predict(X_pad)
        X = pd.DataFrame(lstm_predictions)
        #print("X.shape **********",X.shape)
        #print(X.head(2))
        return X


from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
class ColumnsLabelEncoder(BaseEstimator, TransformerMixin):
      
    def __init__(self, variables=None):
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
       
        country = {'Country_01':1,'Country_02':2,'Country_03':3}
      
        local = { 'Local_01': 1,
                  'Local_02': 2,
                  'Local_03': 3,
                  'Local_04': 4,
                  'Local_05': 5,
                  'Local_06': 6,
                  'Local_07': 7,
                  'Local_08': 8,
                  'Local_09': 9,
                  'Local_10': 10,
                  'Local_11': 11,
                  'Local_12': 12 }

        emp = {
                  'Third Party': 1,
                  'Employee': 2,
                  'Third Party (Remote)': 3
        }
        
        risk_grade = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6}        
        

        X['Countries'] = pd.Series([country[x] for x in X['Countries']], index=X.index)
        #print('Countries Encoded')

        X['Local'] = pd.Series([local[x] for x in X['Local']], index=X.index)
        #print('Local Encoded')

        X['Emp_Type'] = pd.Series([emp[x] for x in X['Emp_Type']], index=X.index)
        #print('Emp_type Encoded')

        X['Potential_Accident'] = pd.Series([risk_grade[x] for x in X['Potential_Accident']], index=X.index)
        #print('Potential Accident Encoded')

       # X['Gender'] = pd.Series([country[x] for x in X['Gender']], index=X.index)
      #  X=X.loc[:,enc_attribs]
       #print("enc Dateframe chk",X.head())
       #print("###############################X.shape:",X.shape)
       ##print("###############################X.head:",X[0,:])
        return X

            
     
from sklearn.base import BaseEstimator,TransformerMixin
import pickle
from sklearn.decomposition import PCA

class FinalModelPredictions(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass
    
    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self
    

    def transform(self, X):
            X = X.copy()
            lstm_pred_model = pickle.load(open('lstm_xgb_lstm_full_model.pkl', 'rb'))
            lstm_pred = lstm_pred_model.predict(X)  
            X = pd.DataFrame(lstm_pred)
            print("lstm_pred.shape **********",X.shape)
            return X     