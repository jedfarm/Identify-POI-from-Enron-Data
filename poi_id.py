#!/usr/bin/python
from __future__ import print_function
import sys
sys.path.append("../tools/")
from sklearn.model_selection import train_test_split
import pickle
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd
import numpy as np
from tester import dump_classifier_and_data
import tester
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
poi_label = ['poi']
financial_feat_list = ['salary', 'bonus', 'long_term_incentive', 'deferred_income',
        'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 
        'total_payments', 'exercised_stock_options', 'restricted_stock',
        'restricted_stock_deferred', 'total_stock_value']
email_feat_list = ['from_messages', 'from_poi_to_this_person',
        'from_this_person_to_poi', 'shared_receipt_with_poi', 'to_messages']

features_list = poi_label + financial_feat_list + email_feat_list 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))
# set the index of df to be the employees series:
df.set_index(employees, inplace=True)

### DATA EXPLORATION


#Those are 25 extrange cases, where actual Enron employees with a valid email 
#address,  do not have email-related features.    
strange_cases = df.index[(df['email_address'] != 'NaN') & (
        df['from_messages']=='NaN')].tolist()
 
# Here we found two entities, that are not real people: 
#THE TRAVEL AGENCY IN THE PARK and TOTAL. These entities do not contribute in 
#any meaningful way to the purpose of this study, so let say that from this 
#moment on we mark them for deletion. 
emailless_people = df.index[df['email_address'] == 'NaN'].tolist()

# The email address is the only field that could not be converted to numeric. 
#We chose to remove it from the data frame because it is of no use to identify 
#poi from the data given.
# Also, in the case of the poi column only zeroes (0) and ones (1) 
#are allowed: 1 = poi, 0 = non-poi    
df=df.apply(lambda x: pd.to_numeric(x, errors='coerse'))
del df['email_address']
df['poi']=df['poi'].astype(int)

df=df[features_list]
    
### Task 2: Remove outliers
df.drop(['TOTAL','THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E'], inplace=True)

# Other data preprocessing steps

# From the financial data, we learned that NaN means zero. 
# Therefore we proceed to make the corresponding changes in our data frame.
df.iloc[:, 1:15] = df.iloc[:, 1:15].fillna(0)

# Impute values to the median
df[email_feat_list]=df[email_feat_list].fillna(
        df.groupby("poi")[email_feat_list].transform("median"))
print("Amount of remaining NaN entries in the dataframe:", df.isnull().sum().sum())

# Fixing errors in the financial data
payments = financial_feat_list[:9]
stock_value = financial_feat_list[10:13]
test_df=df[df[stock_value].sum(axis='columns') != df.total_stock_value][
        financial_feat_list]
test_df.loc[['BELFER ROBERT']] = test_df.loc[['BELFER ROBERT']].shift(
        -1, axis =1).fillna(0)
test_df.loc[['BHATNAGAR SANJAY']] = test_df.loc[['BHATNAGAR SANJAY']].shift(
        1, axis =1).fillna(0)
df.update(test_df)

### Task 3: Create new feature(s)
with open("new_data_df.pickle", "r") as data_file:
    df_new = pickle.load(data_file)
    df_new.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E'], inplace = True)
    

df['to_poi_rate'] = df['from_this_person_to_poi']/df['from_messages']
df['from_poi_rate'] = df['from_poi_to_this_person']/df['to_messages']
new_feat_list = ['from_messages_median_pubIndex', 'to_poi_median_pubIndex']
df = pd.concat([df, df_new], axis=1)
df[new_feat_list]=df[new_feat_list].fillna(df.groupby("poi")[new_feat_list].transform("median"))

features_list = (poi_label + financial_feat_list + email_feat_list + 
                 ['to_poi_rate', 'from_poi_rate' ] + new_feat_list)
print("Total number of features: ", len(features_list)-1)


### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient = 'index')

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.
clf = AdaBoostClassifier(random_state = 45)
clf.fit(features_train, labels_train)

feat_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0.02:
        feat_importance.append([df.columns[i+1], round(clf.feature_importances_[i], 2)])
feat_importance.sort(key=lambda x: x[1], reverse = True)
print("Most relevant features, AdaBoost:")
for item in feat_importance:
    print('{} ===> {}'.format(item[0], item[1]))
print()
boost_feat_list = [x[0] for x in feat_importance]
boost_feat_list.insert(0, 'poi')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.
###StratifiedShuffleSplit.html


features_list = boost_feat_list
### Extract features and labels from dataset for local testing
boost_data = featureFormat(my_dataset, features_list, sort_keys = True)
b_labels, b_features = targetFeatureSplit(boost_data)
b_features_train, b_features_test, b_labels_train, b_labels_test = train_test_split(
        b_features, b_labels, test_size=0.3, random_state=42)


# Cross-validation for parameter tuning in grid search 
sss = StratifiedShuffleSplit(n_splits = 100, test_size = 0.3, random_state = 0)



# Parameters to try in a grid search. A series of two consecutive grid search procedures 
# is more time efficient than one.
param_grid = {              
              "n_estimators": [50, 100, 200, 400, 600, 800],
              "learning_rate": [0.001, 0.01, 0.1, 1]
             }

DTC = DecisionTreeClassifier(random_state = 45)
ABC = AdaBoostClassifier(base_estimator = DTC, random_state = 45)

boost_clf = GridSearchCV(ABC, param_grid=param_grid, scoring = 'f1')

boost_clf.fit(b_features_train, b_labels_train)
b_labels_predictions = boost_clf.predict(b_features_test)
b_classif_report = classification_report(b_labels_test, b_labels_predictions)

param_grid_2 = {
              "base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :  ["best", "random"],
              "base_estimator__max_depth" : [None, 1, 2, 3, 4],
              "base_estimator__min_samples_leaf" : [1, 2, 3, 4], 
              "base_estimator__min_samples_split" : [2, 3, 4, 5, 6]
             }

DTC = DecisionTreeClassifier(random_state = 45, class_weight = None, min_weight_fraction_leaf = 0)
ABC = AdaBoostClassifier(base_estimator = DTC, 
                         random_state = 45,
                         n_estimators = boost_clf.best_params_['n_estimators'], 
                         learning_rate = boost_clf.best_params_['learning_rate'],
                        )

# run grid search
boost_clf_2 = GridSearchCV(ABC, param_grid=param_grid_2, scoring = 'f1')

boost_clf_2.fit(b_features_train, b_labels_train)

DTC = DecisionTreeClassifier(random_state = 45, 
     class_weight = None, 
     min_weight_fraction_leaf = 0, 
     criterion = boost_clf_2.best_params_['base_estimator__criterion'], 
     max_depth = boost_clf_2.best_params_['base_estimator__max_depth'],
     splitter =  boost_clf_2.best_params_['base_estimator__splitter'], 
     min_samples_leaf = boost_clf_2.best_params_['base_estimator__min_samples_leaf'],
     min_samples_split = boost_clf_2.best_params_['base_estimator__min_samples_split']
                            )
clf = AdaBoostClassifier(base_estimator = DTC, learning_rate = boost_clf.best_params_[
        'learning_rate'], n_estimators = boost_clf.best_params_['n_estimators'], random_state = 45)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list )



