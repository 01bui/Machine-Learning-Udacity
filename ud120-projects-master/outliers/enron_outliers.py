#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

maxSalary = max(data[:,0])
maxBonus = max(data[:,1])

#print data_dict

for name, info in data_dict.iteritems():
    for feature, values in info.iteritems():
        if feature == "salary":
            if values == maxSalary:
                keyName = name
                print name, values

        if feature == "bonus":
            if values == maxBonus:
                print name, values

data_dict.pop(keyName, 0)
data = featureFormat(data_dict, features)

import math
for name, info in data_dict.iteritems():
    for feature, values in info.iteritems():
        if (feature == "salary" and values >= 1000000) or (feature == "bonus" and values >= 5000000):
            if math.isnan(float(values)) == 1:
                continue
            else:
                print name, values


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
