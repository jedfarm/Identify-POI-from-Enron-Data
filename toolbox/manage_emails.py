#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:54:42 2017
 Creates a pickle file that contains a dataframe with data from all the eamils in
 maildir
 This file works with a big amount of data and it takes a lot of time running
@author: jedfarm
"""
import pickle
import os, sys, email
import pandas as pd
from pathlib import Path
import time

start_time = time.time()
p = Path("../maildir")
subdirectories = [x for x in p.iterdir() if x.is_dir()]
file_list = list(p.glob('**/*.'))

error_list = []
email_dict_list = []
for file in file_list:
    msg_to_dict = {}
    try:  
        with open(file, 'r', errors='replace') as f:
            msg = email.message_from_file(f)  
            body = msg.get_payload()
            messages = msg.items()
            messages.append(('Body', body))
            msg_to_dict = dict(messages)
            email_dict_list.append(msg_to_dict)
    except:
        error_list.append(file)
        continue   
        
emails = pd.DataFrame(email_dict_list)




####################   BASIC CLEANING  ##############################################
# Eliminating duplicates
#emails.drop_duplicates(['To', 'From', 'Subject', 'Body', 'Cc', 'Bcc'], inplace=True)

# Removing non-relevant columns 
col_dumper = ['Mime-Version', 'Content-Transfer-Encoding', 'Attendees',
                'Re', 'Time', 'Content-Type', 'X-Folder', 'X-Origin', 'X-FileName']
for item in col_dumper:
    if item in emails.columns:
        emails.drop(item, axis = 1, inplace = True)

print("--- %s seconds ---" % ((time.time() - start_time)))

emails = emails.fillna('')

sample = emails[emails['Bcc']!=emails['Cc']]
sample.shape[0]  # Cc and Bcc columns are equal!

# There is no need to keep Bcc and X-bcc columns
col_dumper = ['Bcc', 'X-bcc']
for item in col_dumper:
    if item in emails.columns:
        emails.drop(item, axis = 1, inplace = True)

# Do all the rows are relevant?
emails.shape[0]==len(emails['Message-ID'].unique())  # YES

emails.to_pickle('big_emails_df.pkl')


