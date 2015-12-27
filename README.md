# Project Machine Learning Udacity
Included my Python codes for all mini-projects and final project of the course. 

1. Dataset: 
In 2000, Enron was one of the largest companies in the United States. 
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. 
In the resulting Federal investigation, there was a significant amount of typically confidential information 
entered into public record, including tens of thousands of emails and detailed financial data for top executives. 

2. Mini-projects:


3. Final Project:
In the final project, I built a person of interest identifier based on financial and email data made public 
as a result of the Enron scandal.

Two new features are implemented (fraction_from_poi and fraction_to_poi 
i.e. the fraction of all messages to this person that come from POIs
the fraction of all messages from this person that are sent to POIs).

Using 2 algorithms to classify POIs: Decision Tree or Support Vector Machine.

Using SelectKBest along with Decision Tree.

Using GridSearchCV to tune SVM.

Performance of the final algorithm selected is assessed by splitting the data into training and testing sets
or through the use of cross validation

When tester.py is used to evaluate performance, precision and recall are both at least 0.3 for both algorithms.

