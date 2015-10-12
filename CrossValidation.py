#Use GridSearchCV to find the best value for cost in SVC

import nltk
import re
from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()

from sklearn import svm,metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation, grid_search


train=pd.read_json('train.json')

train['ingredients_string'] = [' '.join([wnl.lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['ingredients']]


Xtrain=train.ingredients_string.values

ytrain=train.cuisine.values

vec1=TfidfVectorizer(stop_words='english')

tfidfTrain=vec1.fit_transform(Xtrain).todense()

svm1=svm.LinearSVC()

parameters = {'C':[0.01,1,100]}

grid1=grid_search.GridSearchCV(svm1,parameters)


grid1.fit(tfidfTrain,ytrain)

grid1.best_estimator_

Out[26]: 
LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)

grid1.grid_scores_

Out[27]: 
[mean: 0.68076, std: 0.00073, params: {'C': 0.01},
 mean: 0.78036, std: 0.00099, params: {'C': 1},
 mean: 0.72474, std: 0.00270, params: {'C': 100}]



