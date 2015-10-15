#For data exploration, examine most-used ingredients in each cuisine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()

train=pd.read_json('train.json')

def clean(x):
    cleanlist=[]
    cleanlist=[wnl.lemmatize(re.sub('[^a-zA-Z]',' ',item)) for item in x]
    return cleanlist

train['ingreC']=train.ingredients.apply(lambda x:clean(x))

all_ingredients=set()
train.ingreC.map(lambda x:[all_ingredients.add(i) for i in list(x)])

#Add a column for each ingredient in the set
for ingredient in all_ingredients:
    train[ingredient]=train.ingreC.apply(lambda x:ingredient in x)
   

#Use groupby.sum() to get the number of times each ingredient appeared in a particular cuisine
train_g1=train.drop(['ingredients','id','ingreC'],axis=1)
train_g2=train_g1.groupby('cuisine').sum()
train_g3=train_g2.tranpose()

#Now the dataframe is ready to be examined and plotted, by each cuisine
train_g3.italian.order(ascending=False)[:10]
train_g3.italian.order(ascending=False)[:10].plot(kind=’bar’)
