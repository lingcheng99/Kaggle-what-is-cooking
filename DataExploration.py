#For data exploration, examine most-used ingredients in each cuisine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()

train=pd.read_json('train.json')
train.columns
Out[2]:
Index([u'cuisine', u'id', u'ingredients'], dtype='object')
train.shape
Out[3]:
(39774, 3)
train.ingredients[0]
Out[5]:
[u'romaine lettuce',
 u'black olives',
 u'grape tomatoes',
 u'garlic',
 u'pepper',
 u'purple onion',
 u'seasoning',
 u'garbanzo beans',
 u'feta cheese crumbles']

train.cuisine.value_counts()
Out[11]:
italian         7838
mexican         6438
southern_us     4320
indian          3003
chinese         2673
french          2646
cajun_creole    1546
thai            1539
japanese        1423
greek           1175
spanish          989
korean           830
vietnamese       825
moroccan         821
british          804
filipino         755
irish            667
jamaican         526
russian          489
brazilian        467
dtype: int64

#Write function "clean" to lemmatize and clean up strings in the "ingredients" column
def clean(x):
    cleanlist=[]
    cleanlist=[wnl.lemmatize(re.sub('[^a-zA-Z]',' ',item)) for item in x]
    return cleanlist

#Add another column "ingreC", with the cleaned up list of ingredients
train['ingreC']=train.ingredients.apply(lambda x:clean(x))
train.ingreC[0]
Out[7]:
[u'romaine lettuce',
 u'black olives',
 u'grape tomatoes',
 u'garlic',
 u'pepper',
 u'purple onion',
 u'seasoning',
 u'garbanzo beans',
 u'feta cheese crumbles']


#Make a set of all ingredients
all_ingredients=set()
train.ingreC.map(lambda x:[all_ingredients.add(i) for i in list(x)])
len(all_ingredients)
Out[8]:
6709

#Add a column for each ingredient in the set
for ingredient in all_ingredients:
    train[ingredient]=train.ingreC.apply(lambda x:ingredient in x)
train.shape
Out[9]:
(39774, 6713)

#Use groupby.sum() to get the number of times each ingredient appeared in a particular cuisine
train_g1=train.drop(['ingredients','id','ingreC'],axis=1)
train_g2=train_g1.groupby('cuisine').sum()
train_g3=train_g2.tranpose()

#Now the dataframe is ready to be examined and plotted, by each cuisine
train_g3.italian.order(ascending=False)[:10]
train_g3.italian.order(ascending=False)[:10].plot(kind=’bar’)

Out[19]:
salt                      3454
olive oil                 3111
garlic cloves             1619
grated parmesan cheese    1579
garlic                    1471
ground black pepper       1444
extra virgin olive oil    1362
onion                     1240
water                     1052
butter                    1029
Name: italian, dtype: float64

train_g3.chinese.order(ascending=False)[:10]

Out[20]:
soy sauce        1363
sesame oil        915
salt              907
corn starch       906
sugar             824
garlic            763
water             762
green onions      628
vegetable oil     602
scallion          591
Name: chinese, dtype: float64
