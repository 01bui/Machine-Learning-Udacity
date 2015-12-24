#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import re

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print len(enron_data)

print len(enron_data["SKILLING JEFFREY K"])
print (enron_data["SKILLING JEFFREY K"])

poiNum=0
for name in enron_data:
    if enron_data[name]["poi"]==1:
        poiNum+=1
print poiNum

poiNamesNum = 0
f = open('G:/Vy_Data/Vy/Study/CUA/Fall_2015/Machine_Learning_Udacity/ud120-projects-master/final_project/poi_names.txt','r')
f.readline()
for line in f:
    poiNames = line.startswith("(")
    #print poiNames
    if poiNames == True:
        poiNamesNum += 1
print poiNamesNum

print enron_data["PRENTICE JAMES"]["total_stock_value"]

print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

for name in enron_data:
    thisName = name.startswith("SKILLING")
    if thisName == True:
        print enron_data[name]["exercised_stock_options"]

for name in enron_data:
    thisName = name.startswith("LAY")
    if thisName == True:
        print enron_data[name]["total_payments"]
for name in enron_data:
    thisName = name.startswith("SKILLING")
    if thisName == True:
        print enron_data[name]["total_payments"]
for name in enron_data:
    thisName = name.startswith("FASTOW")
    if thisName == True:
        print enron_data[name]["total_payments"]

count=0
for name in enron_data:
     if enron_data[name]["salary"] != "NaN":
         count+=1
print count
count=0
for name in enron_data:
     if enron_data[name]["email_address"] != "NaN":
         count+=1
print count

import sys
sys.path.append("../tools")
from feature_format import featureFormat
from feature_format import targetFeatureSplit

count = 0
for name in enron_data:
     if enron_data[name]["total_payments"] == "NaN":
         count+=1
print count
print count/float(len(enron_data))*100

poiNum=0
for name in enron_data:
    if enron_data[name]["poi"]==1 and enron_data[name]["total_payments"]=="NaN":
        poiNum+=1
print poiNum
print poiNum/float(len(enron_data))*100

poiNum=0
for name in enron_data:
    if enron_data[name]["poi"]==1 and enron_data[name]["total_stock_value"]=="NaN":
        poiNum+=1
print poiNum
print poiNum/float(len(enron_data))*100

