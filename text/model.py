import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import neattext.functions as nfx
import re

df=pd.read_csv('data/text_emotion.csv') #ISEAR

#print(df['sentiment'].value_counts())

sns.countplot(x='Emotion',data = df)
#plt.show()

#print(df.head())

df['Clean_Text']=df['Text'].apply(nfx.remove_userhandles)

def remove_punctuations(text):
    punctuation_pattern = r'[.,\']'
    return re.sub(punctuation_pattern, '', text)

df['Clean_Text'] = df['Clean_Text'].apply(remove_punctuations)

#print(dir(nfx))

df['Clean_Text']=df['Clean_Text'].apply(nfx.remove_stopwords)

x=df['Clean_Text']
y=df['Emotion']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


#Model Initalization

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#print(df)

pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_lr.fit(x_train,y_train)
print(pipe_lr.score(x_test,y_test))

pipe_svm = Pipeline(steps=[('cv',CountVectorizer()),('svc',SVC(kernel='rbf', C=10))])
pipe_svm.fit(x_train,y_train)
print(pipe_svm.score(x_test,y_test))

pipe_rf = Pipeline(steps=[('cv',CountVectorizer()),('rf', RandomForestClassifier(n_estimators=10))])
pipe_rf.fit(x_train,y_train)
print(pipe_rf.score(x_test,y_test))


import joblib
pipeline_file = open("text_emotion_model.pkl", "wb")
joblib.dump(pipe_lr , pipeline_file)
pipeline_file.close()








