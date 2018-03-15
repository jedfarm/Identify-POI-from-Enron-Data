#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 08:40:20 2018

This script takes several hours running

@author: jedfarm
"""

import pickle
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz
import itertools
import time
import copy

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']

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

emails = pd.read_pickle('all_emails_before_explode.pkl')


def match_list(name_email_dict, key, listCol):
    """
    The To and Cc columns contain list of emails, this function helps to find a (key) match 
    within our dictionary of the known emails.
    """
    match = False
    for email in listCol:
        if email in name_email_dict[key]:
            match = True
            break
    return match



def from_poi_reviewed(email_name_dict, name_email_dict, key, fromCol, toCol, ccCol, fromPoi):
    """
    Eliminates overcount in fromPoi when poi send  emails to themselves. 
    """
    match = False
    if fromPoi == 1:
        for email in toCol:
            if (email in name_email_dict[key]) & (email_name_dict[fromCol][0] != key):
                match = True   
                break
        for email in ccCol:
           if (email in name_email_dict[key]) & (email_name_dict[fromCol][0] != key):
                match = True   
                break
    return match   


def from_poi_to_this_person(email_name_dict, name_email_dict, key, fromCol, toCol,fromPoi):
    """
    A function made to split up the contributions of the from_poi_reviewed function particular
    to the field To of the emails (It does not take into account Cc from poi)
    """
    match = False
    if fromPoi == 1:
        for email in toCol:
            if (email in name_email_dict[key]) & (email_name_dict[fromCol][0] != key):
                match = True   
                break
    return match

def from_poi_cc_this_person(email_name_dict, name_email_dict, key, fromCol, ccCol,fromPoi):
    """
    This is the complement of the from_poi_to_this_person function
    """
    match = False
    if fromPoi == 1:
        for email in ccCol:
               if (email in name_email_dict[key]) & (email_name_dict[fromCol][0] != key):
                    match = True   
                    break
    return match    


   
### Updated version for the email part of the  data_dict
my_data_dict = {}
for key in name_email_dict:   
    from_name = emails.loc[emails['From'].isin(name_email_dict[key])]
    my_data_dict[key] = {}
    my_data_dict[key]['from_messages'] = from_name.groupby('From').count()['Message-ID'].sum()
    if my_data_dict[key]['from_messages'] > 0:
        # cc sent to poi counts as well
        my_data_dict[key]['from_this_person_to_poi'] = from_name['POI'].apply(
                lambda x:x[1]|x[2]).sum()  
    else:
        my_data_dict[key]['from_this_person_to_poi'] = 0
    # just the To column in the dataframe was taken into account for the 'to_messages' key
    my_data_dict[key]['to_messages'] = emails.loc[emails['To'].apply(lambda x: match_list(
            name_email_dict, key, x))].shape[0]
    
    my_data_dict[key]['from_poi_to_this_person'] = emails.loc[emails[['From', 'To', 
            'fromPoi']].apply(lambda x: from_poi_to_this_person(
            email_name_dict, name_email_dict, key, *x), axis=1)].shape[0]          
    my_data_dict[key]['from_poi_cc_this_person'] = emails.loc[emails[['From', 'Cc', 
            'fromPoi']].apply(lambda x: from_poi_cc_this_person(
            email_name_dict, name_email_dict, key, *x), axis=1)].shape[0]   

def share_receipt_with_poi(email_name_dict, name_email_dict, key, toCol, 
                           ccCol, fromPoi, numPoiToAndCc):
    """
    When emails do not come from poi, there could still have a poi or more in the To or Cc
    fields. 
    """
    
    match = False
    if fromPoi == 0:
       for email in toCol:
            if (email in name_email_dict[key]):
                if int(email_name_dict[email][1]) == 0:
                    if numPoiToAndCc > 0:
                        match = True
                        break
                else:  
                    if numPoiToAndCc > 1:
                        match = True
                        break
    
       for email in ccCol:
          if email not in toCol:
              if (email in name_email_dict[key]):
                if int(email_name_dict[email][1]) == 0:
                    if numPoiToAndCc > 0:
                        match = True
                        break
                else:
                    if numPoiToAndCc > 1:
                        match = True
                        break 
    return match



for key in name_email_dict:
    my_data_dict[key]['shared_receipt_with_poi'] = emails.loc[emails[['To', 
           'Cc', 'fromPoi', 'numPoiToAndCc']].apply(lambda x: share_receipt_with_poi(
            email_name_dict, name_email_dict, key, *x), axis=1)].shape[0] 

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

missing_data_df = pd.DataFrame.from_records(list(my_data_dict.values()))
employees = pd.Series(list(my_data_dict.keys()))
missing_data_df.set_index(employees, inplace=True)
missing_data_df.reset_index(inplace = True)
missing_data_df['index'] = missing_data_df['index'].apply(name_flipper)
missing_data_df.set_index('index', inplace = True)

# Save the dictionary to a file for future work
with open('missing_data_df.pickle', 'wb') as handle:
    pickle.dump(missing_data_df, handle, protocol=0) 