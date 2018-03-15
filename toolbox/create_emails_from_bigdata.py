#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 22:45:05 2018

@author: jedfarm
"""

import pickle
import pandas as pd
import numpy as np


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

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

emails = pd.read_pickle('big_emails_df.pkl')

# Parse datetime
emails['Date'] = pd.to_datetime(emails['Date'], infer_datetime_format=True, 
      errors='coerce')       

### Develop a procedure like this for all the names on the list
from_message_dict = {}
for key in name_email_dict:   
    name_sample = emails.loc[emails['From'].isin(name_email_dict[key])]
    from_message_dict[key] = name_sample.groupby('From').count()['Message-ID'].sum()

    

def split_addresses(x):
    s = []
    if (pd.isnull(x)) | (x == ''):
        return s
    else:
        for item in x.split(','):
            s.append(item.strip())
        return s
colsToList = ['Cc', 'To']
for col in colsToList:
    emails[col] = emails[col].apply(split_addresses)


def public_index(email_name_dict, name_email_dict, from_email, toList, ccList):
    """
    public_index equals 1 if the email is sent just to a single person, and goes 
    up according to the number of different people enlisted in the to, cc 
    and bcc fields.
    """
    idx = 0    
    if np.all(pd.notnull(toList)):
        for toemail in toList:
            if toemail in email_name_dict: 
               if from_email not in name_email_dict[email_name_dict[toemail][0]]: 
                   idx += 1
            else:
                if toemail != from_email:
                    idx += 1
    if np.all(pd.notnull(ccList)):
        for ccemail in ccList:
            if ccemail in email_name_dict:
                found_in_toList = False
                for toemail in toList:
                    if toemail in name_email_dict[email_name_dict[ccemail][0]]:
                        found_in_toList = True
                if (from_email not in name_email_dict[email_name_dict[ccemail][0]]
                ) & (found_in_toList == False): 
                    idx += 1
            else:     
                if (ccemail not in toList) & (ccemail != from_email):
                    idx += 1
    return idx


emails['pubIndex'] = emails[['From', 'To', 'Cc']].apply(
        lambda x: public_index(email_name_dict, name_email_dict, *x), axis=1)


def from_poi(poi_emails, fromCol):
    if fromCol in poi_emails:
        output = 1
    else:
        output = 0
    return output

emails['fromPoi'] = emails['From'].apply(lambda x: from_poi(poi_emails, x))

def poi_involved(poi_emails, listCol):
    output = 0
    for email in listCol:
        if email in poi_emails:
            output = 1
            break
        
    return output


emails['toPoi'] = emails['To'].apply(lambda x: poi_involved(poi_emails, x))
emails['ccPoi'] = emails['Cc'].apply(lambda x: poi_involved(poi_emails, x))

def num_poi_to_and_cc(email_name_dict, toCol, ccCol):
    """
    Takes into account how many poi are involved in the To and Cc fields. This is 
    relevant to build the feature share_receipt_with_poi in the cases of poi.
    """
    
    poi_names_list = []
    for email in toCol:
        if (email in email_name_dict):
            if (int(email_name_dict[email][1]) == 1):
                if email_name_dict[email][0] not in poi_names_list:
                    poi_names_list.append(email_name_dict[email][0])
    for email in ccCol:
        if (email in email_name_dict):
            if (int(email_name_dict[email][1]) == 1):
                if email_name_dict[email][0] not in poi_names_list:
                    poi_names_list.append(email_name_dict[email][0])
   
    return  len(poi_names_list)
            
emails['numPoiToAndCc'] = emails[['To', 'Cc']].apply(
        lambda x: num_poi_to_and_cc(email_name_dict, *x), axis = 1) 


def poi_involved_reviewer(email_name_dict, fromCol, toList, ccList,
                          fromPOI, toPOI, ccPOI, numPoiToAndCc):
    """
    Search for meassages poi sent to themselves(not involving other poi) 
    and removes the value 1 in the corresponding column.
    Does not re-write the column, creates new entries in a form of a list instead
    """
    output = [0, 0, 0]
    fromCol = str(fromCol).strip()
    
    if fromPOI == 1:
        output[0]= 1
        if fromCol not in toList:
            match = False
            for email in toList:
                if email in email_name_dict:
                    if (email_name_dict[email][0] == email_name_dict[
                            fromCol][0]) & (numPoiToAndCc < 2):
                        match = True
            if (not match) & (toPOI) == 1:
                output[1] = 1
       
        if fromCol not in ccList:
            match = False
            for email in ccList:
                if email in email_name_dict:
                    if (email_name_dict[email][0] == email_name_dict[
                            fromCol][0]) & (numPoiToAndCc < 2):
                        match = True
            if (not match) & (ccPOI) == 1:
                output[2] = 1      
    else:
        output[1] = toPOI
        output[2] = ccPOI      
    return output

emails['POI'] = emails[['From', 'To', 'Cc', 'fromPoi', 'toPoi', 'ccPoi', 'numPoiToAndCc']].apply(
        lambda x: poi_involved_reviewer(email_name_dict, *x), axis=1)

# Saving the dataframe to work with it elsewhere
emails.to_pickle('all_emails_before_explode.pkl')

