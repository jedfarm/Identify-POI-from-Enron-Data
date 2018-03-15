#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:13:55 2018

@author: jedfarm
"""
import pickle
import pandas as pd
import re
import numpy as np

with open('AllEmployeesEmails2.csv', 'r') as f:
    next(f)
    email_name_dict={}
    for line in f:
        w = line.split(',')
        email_name_dict[w[0]] = []
        email_name_dict[w[0]].append(w[1])
        email_name_dict[w[0]].append(w[2])

all_names = []
poi_names = []
poi_emails = []
for key in email_name_dict:
    if email_name_dict[key][0] not in all_names:
        all_names.append(email_name_dict[key][0])
    if (int(email_name_dict[key][1]) == 1):
        poi_emails.append(key)
        if (email_name_dict[key][0] not in poi_names):
            poi_names.append(email_name_dict[key][0])

extra_names = ['charles lemaistre', 'charles p lowry', 'eugene e lockhart', 
               'gareth w walters', 'herbert s winokur jr.', 'james p badum', 
               'jerome j meyer', 'john c baxter', 'john l fugh', 'john mendelsohn', 
               'john wakeham', 'kenneth w cline', 'lawrence reynolds', 
               'michael s cumberland', 'norman p blake jr.', 'paulo v pereiraferraz', 
               'robert belfer', 'robert jaedicke', 'robert s gahn', 'rodney gray', 
               'william d gathmann']
for name in extra_names:
    if name not in all_names:
        all_names.append(name)
 
name_email_dict = {}       
for key in email_name_dict:
    if email_name_dict[key][0] not in name_email_dict:
        name_email_dict[email_name_dict[key][0]] = [key]
    else:
        name_email_dict[email_name_dict[key][0]].append(key)

all_emails = []
for key in email_name_dict:
    all_emails.append(key)

# Loading the email dataframe
emails = pd.read_pickle('all_emails_before_explode.pkl')

     
reduced_from_emails = emails[emails['From'].isin(email_name_dict)]

def names_from_emails(email_name_dict, fromCol):
    return email_name_dict[fromCol][0]    

reduced_from_emails['X-From'] = reduced_from_emails['From'].apply(lambda x: names_from_emails(
        email_name_dict, x))


med_reduced_from_emails = reduced_from_emails.groupby(['X-From']).median() 
med_reduced_from_emails.rename(columns={'pubIndex': 'from_messages_median_pubIndex'}, 
                               inplace=True)
  
med_reduced_to_poi = reduced_from_emails.groupby(['X-From', 'toPoi']).median()   
med_reduced_to_poi = med_reduced_to_poi.reset_index()
med_reduced_to_poi = med_reduced_to_poi[med_reduced_to_poi['toPoi']==1]
med_reduced_to_poi.index = med_reduced_to_poi['X-From']
med_reduced_to_poi.rename(columns={'pubIndex': 'to_poi_median_pubIndex'}, 
                               inplace=True)

med_reduced_from_emails['to_poi_median_pubIndex']= med_reduced_to_poi['to_poi_median_pubIndex']
med_reduced_from_emails = med_reduced_from_emails[['from_messages_median_pubIndex', 
                                                   'to_poi_median_pubIndex']]
med_reduced_from_emails['to_poi_median_pubIndex'].fillna(0, inplace=True)

med_reduced_from_emails.reset_index(inplace = True)

def name_flipper(name):
    """
    Making names compatible with data_dict
    """
  
    pattern = re.compile('(\w?\s?\w*\s?\w?)\s(\w*\-?\w*\s?j?r?\.?)')
    if pattern.match(name.strip()):
       new_name = pattern.match(name).groups()[1].upper(
               ) + ' ' + pattern.match(name).groups()[0].upper() 
    else:
        new_name = name
    return new_name

med_reduced_from_emails['X-From'] = med_reduced_from_emails['X-From'].apply(name_flipper)
med_reduced_from_emails.set_index('X-From', inplace = True)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))
# set the index of df to be the employees series:
df.set_index(employees, inplace=True)

df['from_messages_median_pubIndex']= med_reduced_from_emails['from_messages_median_pubIndex']
df['to_poi_median_pubIndex']= med_reduced_from_emails['to_poi_median_pubIndex']

df = df [['from_messages_median_pubIndex', 'to_poi_median_pubIndex']]

with open('new_data_df.pickle', 'wb') as handle:
     pickle.dump(df, handle, protocol=0) 

