#!/usr/bin/python

import sys
import pickle
import matplotlib
import numpy as np
from matplotlib import pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)

data = featureFormat(data_dict, features_list)
maxSalary = max(data[:,1])
maxBonus = max(data[:,2])

for name, info in data_dict.iteritems():
    for feature, values in info.iteritems():
        if feature == "salary":
            if values == maxSalary:
                keyName = name
                #print name, values

        if feature == "bonus":
            if values == maxBonus:
                keyName = name
                #print name, values

data_dict.pop(keyName, 0)

### Task 3: Create new feature(s)

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.0
    if poi_messages == "NaN" and all_messages == "NaN":
        fraction = 0.0
    else:
        fraction = float(poi_messages) / all_messages

    return fraction

submit_dict = {}
for name in data_dict:
    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi

# Add new features to data_dict
for name, info in data_dict.iteritems():
    if name in submit_dict:
        #print name, info
        for f in submit_dict[name].iteritems():
            #print f
            info.update({f})

#print data_dict['METTS MARK']
#print submit_dict['METTS MARK']

### Store to my_dataset for easy export below.
my_dataset = data_dict
# Add new feature to the list
#features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', \
#                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', \
#                 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees', \
#                 'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi']

features_list = ['poi','salary', \
                 'bonus', 'total_stock_value', \
                 'exercised_stock_options', \
                 'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit( data )

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

# I use train_test_split cross validation while the tester.py use StratifiedShuffleSplit
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Algorithm 1: faster
# This produces the precision and recall > .3
#anova_filter = SelectKBest(f_regression, k=7)
#clf = make_pipeline(anova_filter, DecisionTreeClassifier())
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)

# Algorithm 2: longer time to run
# This produces the precision and recall > .3
from sklearn.decomposition import RandomizedPCA
n_components = 7
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(features_train)
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

estimators = [('reduce_dim', RandomizedPCA(n_components=n_components, whiten=True)), ('svm', SVC(kernel='rbf', class_weight='auto'))]
pipe = Pipeline(estimators)

param_grid = {
              'svm__C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'svm__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
clf = GridSearchCV(pipe, param_grid)
clf.fit(features_train,labels_train)
print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
pred = clf.predict(features_test)

# Print out the accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)
print acc
# Print out the precision and recall score
from sklearn.metrics import precision_score, recall_score
print precision_score(labels_test, pred)
print recall_score(labels_test, pred)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

#dump_classifier_and_data(clf, my_dataset, features_list)