import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer

wnl=WordNetLemmatizer()

from sklearn import svm,metrics, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer


train=pd.read_json('train.json')

train['ingredients_string'] = [' '.join([wnl.lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['ingredients']]


X=train.ingredients_string.values

y=train.cuisine.values


vec1=TfidfVectorizer(stop_words='english')

tfidfTrain=vec1.fit_transform(X).todense()


test=pd.read_json('test.json')

test['ingredients_string'] = [' '.join([wnl.lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in test['ingredients']]

Xtest=test.ingredients_string.values

tfidfTest=vec1.transform(Xtest)


svm1=svm.LinearSVC(C=1)

svm1.fit(tfidfTrain,y)

ypred1=svm1.predict(tfidfTest)

test['cuisine']=ypred1

test[['id','cuisine']].to_csv('pred1.csv',index=False)

