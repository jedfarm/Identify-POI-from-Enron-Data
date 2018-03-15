

```python
from __future__ import print_function
import sys
sys.path.append("../tools/")
from sklearn.model_selection import train_test_split
import pickle
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd
import numpy as np
import itertools
from tester import dump_classifier_and_data
import tester
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
```

Loading the data set


```python
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))
# set the index of df to be the employees series:
df.set_index(employees, inplace=True)
```

# DATA EXPLORATION

We assume that all the email-related data were collected using valid email addresses, therefore if some people have their email addresses missing, that implies the corresponding email features will be missing as well (NaN). The contrary is not true as we are about to see:


```python
print("Number of people with missing email data: ",df['from_messages'].value_counts().max())
print("Number of people with missing email address: ",df['email_address'].value_counts().max())
```

    Number of people with missing email data:  60
    Number of people with missing email address:  35


Who are these people?


```python
strange_cases = df.index[(df['email_address'] != 'NaN') & (df['from_messages']=='NaN')].tolist()
print("People with at least one email address but without email-related features: ")
print()
for i, name in enumerate(strange_cases):
    print(i + 1, name)
```

    People with at least one email address but without email-related features: 
    
    1 ELLIOTT STEVEN
    2 MORDAUNT KRISTINA M
    3 WESTFAHL RICHARD K
    4 WODRASKA JOHN
    5 ECHOLS JOHN B
    6 KOPPER MICHAEL J
    7 BERBERIAN DAVID
    8 DETMERING TIMOTHY J
    9 GOLD JOSEPH
    10 KISHKILL JOSEPH G
    11 LINDHOLM TOD A
    12 BUTTS ROBERT H
    13 HERMANN ROBERT J
    14 SCRIMSHAW MATTHEW
    15 FASTOW ANDREW S
    16 OVERDYKE JR JERE C
    17 STABLER FRANK
    18 PRENTICE JAMES
    19 WHITE JR THOMAS E
    20 CHRISTODOULOU DIOMEDES
    21 DIMICHELE RICHARD G
    22 YEAGER F SCOTT
    23 HIRKO JOSEPH
    24 PAI LOU L
    25 BAY FRANKLIN R


Those are 25 extrange cases, where actual Enron employees with a valid email address,  do not have email-related features.


```python
emailless_people = df.index[df['email_address'] == 'NaN'].tolist()
print("People without an email address: ")
print()
for i, name in enumerate(emailless_people):
    print(i + 1, name)
```

    People without an email address: 
    
    1 BAXTER JOHN C
    2 LOWRY CHARLES P
    3 WALTERS GARETH W
    4 CHAN RONNIE
    5 BELFER ROBERT
    6 URQUHART JOHN A
    7 WHALEY DAVID A
    8 MENDELSOHN JOHN
    9 CLINE KENNETH W
    10 WAKEHAM JOHN
    11 DUNCAN JOHN H
    12 LEMAISTRE CHARLES
    13 SULLIVAN-SHAKLOVITZ COLLEEN
    14 WROBEL BRUCE
    15 MEYER JEROME J
    16 CUMBERLAND MICHAEL S
    17 GAHN ROBERT S
    18 GATHMANN WILLIAM D
    19 GILLIS JOHN
    20 BAZELIDES PHILIP J
    21 LOCKHART EUGENE E
    22 PEREIRA PAULO V. FERRAZ
    23 BLAKE JR. NORMAN P
    24 GRAY RODNEY
    25 THE TRAVEL AGENCY IN THE PARK
    26 NOLES JAMES L
    27 TOTAL
    28 JAEDICKE ROBERT
    29 WINOKUR JR. HERBERT S
    30 BADUM JAMES P
    31 REYNOLDS LAWRENCE
    32 YEAP SOON
    33 FUGH JOHN L
    34 SAVAGE FRANK
    35 GRAMM WENDY L


Here we found two entities, that are not real people: THE TRAVEL AGENCY IN THE PARK and TOTAL. These entities do not contribute in any meaningful way to the purpose of this study, so let say that from this moment on we mark them for deletion. 

The email address is the only field that could not be converted to numeric. We chose to remove it from the data frame because it is of no use to identify poi from the data given.
Also, in the case of the poi column only zeroes (0) and ones (1) are allowed: 1 = poi, 0 = non-poi


```python
df=df.apply(lambda x: pd.to_numeric(x, errors='coerse'))
del df['email_address']
df['poi']=df['poi'].astype(int)
```


```python
poi_label = ['poi']
financial_feat_list = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 
                       'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 
                       'exercised_stock_options', 'restricted_stock','restricted_stock_deferred', 'total_stock_value']
email_feat_list = ['from_messages', 'from_poi_to_this_person','from_this_person_to_poi', 'shared_receipt_with_poi', 
                   'to_messages']
features_list = poi_label + financial_feat_list + email_feat_list
df=df[features_list]
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 146 entries, METTS MARK to GLISAN JR BEN F
    Data columns (total 20 columns):
    poi                          146 non-null int64
    salary                       95 non-null float64
    bonus                        82 non-null float64
    long_term_incentive          66 non-null float64
    deferred_income              49 non-null float64
    deferral_payments            39 non-null float64
    loan_advances                4 non-null float64
    other                        93 non-null float64
    expenses                     95 non-null float64
    director_fees                17 non-null float64
    total_payments               125 non-null float64
    exercised_stock_options      102 non-null float64
    restricted_stock             110 non-null float64
    restricted_stock_deferred    18 non-null float64
    total_stock_value            126 non-null float64
    from_messages                86 non-null float64
    from_poi_to_this_person      86 non-null float64
    from_this_person_to_poi      86 non-null float64
    shared_receipt_with_poi      86 non-null float64
    to_messages                  86 non-null float64
    dtypes: float64(19), int64(1)
    memory usage: 24.0+ KB


Counting the number of poi and non-poi in the dataset.


```python
df['poi'].value_counts()
```




    0    128
    1     18
    Name: poi, dtype: int64




```python
print("Number of poi without email data: ", df[(df['poi']==1) & (~df.to_messages.notnull())].shape[0])
```

    Number of poi without email data:  4


# DATA CLEANSING

Checking if there are people without data associated.


```python
df[df.isnull().sum(axis=1) >= df.shape[1]-1]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>poi</th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>from_messages</th>
      <th>from_poi_to_this_person</th>
      <th>from_this_person_to_poi</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LOCKHART EUGENE E</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



It seems that the only thing we know about Eugene E. Lockhart is that he is not a person of interest.

As we have a relatively low number of data points, we are going to proceed extra-carefully at removing them.
For the moment, we are going to do it just to the items we previously have marked for deletion plus this last one, and we will analyze any further need in a case by case manner as we proceed with our ML algorithms.


```python
df.drop(['TOTAL','THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E'], inplace=True)

```

From the financial data, we learned that NaN means zero. Therefore we proceed to make the corresponding changes in our data frame.


```python
df.iloc[:, 1:15] = df.iloc[:, 1:15].fillna(0)
```

After performing such an operation, the number of NaN values was dramatically reduced from 1323 up to 285, which ultimately is the amount of missing email-related entries.


```python
print("Number of email-related missing data: ", df.isnull().sum().sum())
```

    Number of email-related missing data:  285


Maybe the best way to proceed with the remaining NaN values is to impute them with the median for non-poi people.


```python
df[email_feat_list]=df[email_feat_list].fillna(df.groupby("poi")[email_feat_list].transform("median"))
print("Amount of remaining NaN entries in the dataframe:", df.isnull().sum().sum())
```

    Amount of remaining NaN entries in the dataframe: 0


As found in some of our references, the manual input of the financial data could have been the cause of some observed mistakes. 


```python
payments = financial_feat_list[:9]
df[df[payments].sum(axis = 1) != df.total_payments][financial_feat_list]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-102500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3285.0</td>
      <td>102500.0</td>
      <td>3285.0</td>
      <td>0.0</td>
      <td>44093.0</td>
      <td>-44093.0</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>15456290.0</td>
      <td>2604490.0</td>
      <td>-2604490.0</td>
      <td>15456290.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
stock_value = financial_feat_list[10:13]
test_df=df[df[stock_value].sum(axis='columns') != df.total_stock_value][financial_feat_list]
test_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-102500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3285.0</td>
      <td>102500.0</td>
      <td>3285.0</td>
      <td>0.0</td>
      <td>44093.0</td>
      <td>-44093.0</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>15456290.0</td>
      <td>2604490.0</td>
      <td>-2604490.0</td>
      <td>15456290.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Fortunately, there are errors in just two rows. Checking the .pdf document obtained from FindLaw, we acknowledged that the errors are in fact shifts of one column in each case but opposite directions. Let's correct them.


```python
test_df.loc[['BELFER ROBERT']] = test_df.loc[['BELFER ROBERT']].shift(-1, axis =1).fillna(0)
test_df.loc[['BHATNAGAR SANJAY']] = test_df.loc[['BHATNAGAR SANJAY']].shift(1, axis =1).fillna(0)

df.update(test_df)

if not (df[df[payments].sum(axis = 1) != df.total_payments].shape[0] | df[df[stock_value].sum(
    axis='columns') != df.total_stock_value][financial_feat_list].shape[0]):
    print("All the financial data has been corrected")
else:
    print("Some errors remain")
```

    All the financial data has been corrected


# CREATE NEW FEATURES

The most straightforward manner of creating new features, in this case, is by using the existing ones. For example, we can create meaningful ratios of two features. A more complicated way to achieve the same goal is to work extensively with the full Enron email dataset. As we were curious about those cases of existing emails and no email related data, we decided to dive into the Enron email data.

### Finding the mysterious missing data

Exploring the Enron email dataset proved to be a time-consuming task. After searching with an intricate pattern of regular expressions and using specific search criteria based on the observed email addresses patterns, we were able to find up to 424 different email addresses linked to the people under study. Our search methods were far from optimal as they included final manual adjudications in many cases. That is why we have reasons to believe that there could be more email addresses than the ones we were able to find (but we decided to leave that as a subject of a more detailed study to be carried out in the future). In any case, our search allowed us to find some of the missing email addresses, and with that information, we built the email-based existing features for the employees including those "strange 25 cases". The code is too large to be inserted here, but we provide a text file with the procedure followed alongside with the script files we used. We are going to load a dictionary we created, similar to data_dict in structure, but with the data we processed directly from the Enron email dataset. It also contains new features.


```python
with open("missing_data_df.pickle", "r") as data_file:
    missing_data_df = pickle.load(data_file)

missing_data_df = missing_data_df[missing_data_df.index.isin(strange_cases)]
missing_data_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>from_messages</th>
      <th>from_poi_cc_this_person</th>
      <th>from_poi_to_this_person</th>
      <th>from_this_person_to_poi</th>
      <th>shared_receipt_with_poi</th>
      <th>to_messages</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FASTOW ANDREW S</th>
      <td>9</td>
      <td>15</td>
      <td>49</td>
      <td>5</td>
      <td>1136</td>
      <td>1183</td>
    </tr>
    <tr>
      <th>BERBERIAN DAVID</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>158</td>
      <td>159</td>
    </tr>
    <tr>
      <th>CHRISTODOULOU DIOMEDES</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>448</td>
      <td>521</td>
    </tr>
    <tr>
      <th>YEAGER F SCOTT</th>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>81</td>
      <td>88</td>
    </tr>
    <tr>
      <th>STABLER FRANK</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>36</td>
      <td>89</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>47</td>
      <td>124</td>
    </tr>
    <tr>
      <th>PRENTICE JAMES</th>
      <td>6</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>72</td>
      <td>344</td>
    </tr>
    <tr>
      <th>OVERDYKE JR JERE C</th>
      <td>3</td>
      <td>0</td>
      <td>33</td>
      <td>0</td>
      <td>374</td>
      <td>465</td>
    </tr>
    <tr>
      <th>ECHOLS JOHN B</th>
      <td>8</td>
      <td>0</td>
      <td>8</td>
      <td>5</td>
      <td>78</td>
      <td>90</td>
    </tr>
    <tr>
      <th>WODRASKA JOHN</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>96</td>
    </tr>
    <tr>
      <th>KISHKILL JOSEPH G</th>
      <td>19</td>
      <td>4</td>
      <td>52</td>
      <td>2</td>
      <td>274</td>
      <td>392</td>
    </tr>
    <tr>
      <th>GOLD JOSEPH</th>
      <td>6</td>
      <td>3</td>
      <td>17</td>
      <td>0</td>
      <td>891</td>
      <td>1060</td>
    </tr>
    <tr>
      <th>HIRKO JOSEPH</th>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>104</td>
      <td>106</td>
    </tr>
    <tr>
      <th>MORDAUNT KRISTINA M</th>
      <td>6</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>121</td>
      <td>403</td>
    </tr>
    <tr>
      <th>PAI LOU L</th>
      <td>0</td>
      <td>9</td>
      <td>25</td>
      <td>0</td>
      <td>542</td>
      <td>544</td>
    </tr>
    <tr>
      <th>SCRIMSHAW MATTHEW</th>
      <td>8</td>
      <td>3</td>
      <td>46</td>
      <td>4</td>
      <td>431</td>
      <td>518</td>
    </tr>
    <tr>
      <th>KOPPER MICHAEL J</th>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>183</td>
      <td>192</td>
    </tr>
    <tr>
      <th>DIMICHELE RICHARD G</th>
      <td>8</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>154</td>
      <td>256</td>
    </tr>
    <tr>
      <th>WESTFAHL RICHARD K</th>
      <td>2</td>
      <td>3</td>
      <td>36</td>
      <td>0</td>
      <td>21</td>
      <td>64</td>
    </tr>
    <tr>
      <th>BUTTS ROBERT H</th>
      <td>5</td>
      <td>3</td>
      <td>18</td>
      <td>0</td>
      <td>199</td>
      <td>283</td>
    </tr>
    <tr>
      <th>HERMANN ROBERT J</th>
      <td>3</td>
      <td>0</td>
      <td>14</td>
      <td>3</td>
      <td>194</td>
      <td>174</td>
    </tr>
    <tr>
      <th>ELLIOTT STEVEN</th>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>87</td>
      <td>194</td>
    </tr>
    <tr>
      <th>WHITE JR THOMAS E</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>601</td>
      <td>623</td>
    </tr>
    <tr>
      <th>DETMERING TIMOTHY J</th>
      <td>13</td>
      <td>11</td>
      <td>30</td>
      <td>4</td>
      <td>174</td>
      <td>272</td>
    </tr>
    <tr>
      <th>LINDHOLM TOD A</th>
      <td>18</td>
      <td>0</td>
      <td>17</td>
      <td>4</td>
      <td>170</td>
      <td>260</td>
    </tr>
  </tbody>
</table>
</div>



Mystery solved: We were able to find the email-related data belonging to those 25 people with valid email addresses. Taking a quick look at the data we realized that there is a disproportion between the number of emails sent and received. The amount of messages sent by these people is suspiciously low (or inexistent) for the timeframe considered. We found email data for additional 19 people (from our second list of 35 shown above) that displays the same trend. 

One particular case in the above data frame is worth noticing: Andrew S. Fastow. It is hard to believe that the chief financial officer of a corporation, (who received at least 1183 emails) just sent nine emails in more than a year, including the time when the financial scandal shattered the company.

I believe that the process of emails removal from the dataset due to privacy protection issues that occurred at some point after the first release of the Enron email data might have something to do with this. As this might be an intentional intervention in the data set, it definitively could affect the outcome of any attempt of classification if these data were to be included. It is, therefore, reasonable to assume that this particular situation is the reason behind the absence of the email-related features for those 25 "strange cases" we found earlier. 

### New features

Having the Enron email dataset in a workable shape makes possible to create any number of new features. In this study we are going to try some that are very easy to create. We proposed four new features, belonging to two different kinds. Two were ratios of the existing features as we mentioned earlier, and the other two were the result of working with the entire email dataset. 
In the second case, we created an intermediate feature, called pubIndex.  This one is not going to be used explicitly, but it is part of the process.

 pubIndex accounts for the number of people involved in a given email (To and Cc fields) correcting for when people sent emails to themselves. The lowest possible value for this feature is zero (if someone sent an email just to him or herself with no Cc), it is equal to one if there is only a single person in the To field and none in the Cc field, and so on. It is worth noticing that there is in principle no upper limit for this feature.  

### Ratios of existing features:

- to_poi_rate: ratio of from_this_person_to_poi / from_messages

- from_poi_rate: ratio of from_poi_to_this_person / to_messages

### New features from the email dataset:

- from_messages_median_pubIndex: We grouped all the emails sent by a given person and took the median of the pubIndex feature.

- to_poi_median_pubIndex: The same as above but considering just when sending messages to poi.




```python
with open("new_data_df.pickle", "r") as data_file:
    df_new = pickle.load(data_file)
    df_new.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E'], inplace = True)

```


```python
df['to_poi_rate'] = df['from_this_person_to_poi']/df['from_messages']
df['from_poi_rate'] = df['from_poi_to_this_person']/df['to_messages']
new_feat_list = ['from_messages_median_pubIndex', 'to_poi_median_pubIndex']
df = pd.concat([df, df_new], axis=1)
df[new_feat_list]=df[new_feat_list].fillna(df.groupby("poi")[new_feat_list].transform("median"))
```


```python
print("Number of email-related missing data: ", df.isnull().sum().sum())
```

    Number of email-related missing data:  0



```python
features_list = poi_label + financial_feat_list + email_feat_list + ['to_poi_rate', 'from_poi_rate' ] + new_feat_list
print("Total number of features: ", len(features_list)-1)
```

    Total number of features:  23



```python
### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient = 'index')
```

# ALGORITHM TRAINING

### Extract features and labels from dataset for local testing


```python
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size=0.3, random_state=42)
```


```python
# Cross-validation for parameter tuning in grid search 
sss = StratifiedShuffleSplit(n_splits = 100, test_size = 0.3, random_state = 0)
```

### Naive Bayes


```python
scaler = StandardScaler()
select = SelectKBest()
clf = GaussianNB()

steps = [
		 # Preprocessing
         ('standard_scaler', scaler),
         
         # Feature selection
         ('feature_selection', select),
         
         # Classifier
         ('clf', clf)
         ]
# Create pipeline
pipeline = Pipeline(steps)

parameters = dict(feature_selection__k=[2, 3, 5, 6, 7, 8, 9, 10, 12])


# Create, fit, and make predictions with grid search
gs = GridSearchCV(pipeline,
                  param_grid=parameters,
                  scoring="recall",
                  cv=sss.split(features_train, labels_train),
                  error_score=0)
gs.fit(features_train, labels_train)

labels_predictions = gs.predict(features_test)


print(" Best score: ", gs.best_score_ , "\n")

classif_report = classification_report(labels_test, labels_predictions)

scores = gs.best_estimator_.named_steps['feature_selection'].scores_
mask = gs.best_estimator_.named_steps['feature_selection'].get_support()

kselect_features = [] 
feat_importance = []
for bool, feature, score in zip(mask, features_list[1:], scores):
    if bool:
        kselect_features.append(feature)
        feat_importance.append([feature, round(score, 2)])
feat_importance.sort(key=lambda x: x[1], reverse = True)
print ("\n", "Optimum number of features, KBest: ", gs.best_params_['feature_selection__k'], "\n")
for item in feat_importance:
    print('{} ===> {}'.format(item[0], item[1]))
print()

kselect_features.insert(0, "poi")
```

     Best score:  0.33 
    
    
     Optimum number of features, KBest:  9 
    
    bonus ===> 30.73
    to_poi_rate ===> 23.94
    shared_receipt_with_poi ===> 16.97
    salary ===> 15.86
    total_stock_value ===> 10.63
    to_poi_median_pubIndex ===> 10.53
    exercised_stock_options ===> 9.68
    total_payments ===> 8.96
    deferred_income ===> 8.75
    



```python
best_pipe = Pipeline([('standard_scaler', StandardScaler()),
                          ('feature_selection', SelectKBest(k=gs.best_params_['feature_selection__k'])),
                          ('clf', GaussianNB())])

tester.dump_classifier_and_data(best_pipe, my_dataset, kselect_features)
tester.main();
```

    Pipeline(memory=None,
         steps=[('standard_scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('feature_selection', SelectKBest(k=9, score_func=<function f_classif at 0x1092e37d0>)), ('clf', GaussianNB(priors=None))])
    	Accuracy: 0.86453	Precision: 0.48938	Recall: 0.36850	F1: 0.42042	F2: 0.38765
    	Total predictions: 15000	True positives:  737	False positives:  769	False negatives: 1263	True negatives: 12231
    


### Decision Tree


```python
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

feat_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        feat_importance.append([df.columns[i+1], round(clf.feature_importances_[i], 2)])
feat_importance.sort(key=lambda x: x[1], reverse = True)
print('Most relevant features, Decision Tree:')
for item in feat_importance:
    print('{} ===> {}'.format(item[0], item[1]))
print()
tree_feat_list = [x[0] for x in feat_importance]
tree_feat_list.insert(0, 'poi')
```

    Most relevant features, Decision Tree:
    to_poi_rate ===> 0.41
    shared_receipt_with_poi ===> 0.3
    to_messages ===> 0.13
    from_poi_to_this_person ===> 0.09
    total_payments ===> 0.08
    



```python
tree_data = featureFormat(my_dataset, tree_feat_list, sort_keys = True)
t_labels, t_features = targetFeatureSplit(tree_data)
t_features_train, t_features_test, t_labels_train, t_labels_test = train_test_split(t_features, t_labels, 
                                                                            test_size=0.3, random_state=42)
```


```python
parameters = dict(
                  criterion=['gini', 'entropy'],
                  splitter=['best', 'random'],
                  max_depth=[None, 1, 2, 3, 4],
                  min_samples_split=[2, 3, 4, 5],
                  min_samples_leaf=[1, 2, 3, 4],
                  min_weight_fraction_leaf=[0, 0.25, 0.5],
                  class_weight=[None, 'balanced'],
                  random_state=[45], 
                  )

dt_clf = GridSearchCV(DecisionTreeClassifier(random_state = 45), param_grid = parameters, cv=sss.split(
                                          features_train, labels_train),scoring='f1')
dt_clf.fit(t_features_train, t_labels_train)
t_labels_predictions = dt_clf.predict(t_features_test)
classif_report = classification_report(t_labels_test, t_labels_predictions)
print("Decision Tree, best parameters set: ")
for key in dt_clf.best_params_:
    print(key, "==>", dt_clf.best_params_[key])

```

    Decision Tree, best parameters set: 
    splitter ==> best
    random_state ==> 45
    min_samples_leaf ==> 1
    min_weight_fraction_leaf ==> 0
    criterion ==> gini
    min_samples_split ==> 2
    max_depth ==> 2
    class_weight ==> balanced



```python
clf = DecisionTreeClassifier(class_weight = dt_clf.best_params_['class_weight'], 
                             criterion = dt_clf.best_params_['criterion'], 
                             max_depth = dt_clf.best_params_['max_depth'], 
                             min_samples_leaf = dt_clf.best_params_['min_samples_leaf'], 
                             min_samples_split = dt_clf.best_params_['min_samples_split'], 
                             min_weight_fraction_leaf = dt_clf.best_params_['min_weight_fraction_leaf'],  
                             splitter = dt_clf.best_params_['splitter'], 
                             random_state = 45)
dump_classifier_and_data(clf, my_dataset, tree_feat_list)
tester.main()
```

    DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=2,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0, presort=False, random_state=45,
                splitter='best')
    	Accuracy: 0.90707	Precision: 0.61173	Recall: 0.82950	F1: 0.70416	F2: 0.77437
    	Total predictions: 15000	True positives: 1659	False positives: 1053	False negatives:  341	True negatives: 11947
    


### AdaBoost 


```python
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
```

    Most relevant features, AdaBoost:
    expenses ===> 0.14
    shared_receipt_with_poi ===> 0.12
    to_poi_rate ===> 0.12
    from_poi_rate ===> 0.12
    from_this_person_to_poi ===> 0.1
    long_term_incentive ===> 0.08
    to_poi_median_pubIndex ===> 0.08
    other ===> 0.06
    salary ===> 0.04
    exercised_stock_options ===> 0.04
    



```python
boost_data = featureFormat(my_dataset, boost_feat_list, sort_keys = True)
b_labels, b_features = targetFeatureSplit(boost_data)
b_features_train, b_features_test, b_labels_train, b_labels_test = train_test_split(b_features, b_labels, 
                                                                            test_size=0.3, random_state=42)
```


```python
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

print("AdaBoost, best parameters set:")
for key in boost_clf.best_params_:
    print(key, "==>", boost_clf.best_params_[key])

```

    AdaBoost, best parameters set:
    n_estimators ==> 50
    learning_rate ==> 0.001



```python
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
print("Optimized base estimator parameters set:")
for key in boost_clf_2.best_params_:
    print(key, "==>", boost_clf_2.best_params_[key])
```

    Optimized base estimator parameters set:
    base_estimator__criterion ==> entropy
    base_estimator__max_depth ==> 2
    base_estimator__min_samples_leaf ==> 1
    base_estimator__min_samples_split ==> 2
    base_estimator__splitter ==> best



```python
DTC = DecisionTreeClassifier(random_state = 45, 
                             class_weight = None, 
                             min_weight_fraction_leaf = 0, 
                             criterion = boost_clf_2.best_params_['base_estimator__criterion'], 
                             max_depth = boost_clf_2.best_params_['base_estimator__max_depth'],
                             splitter =  boost_clf_2.best_params_['base_estimator__splitter'], 
                             min_samples_leaf = boost_clf_2.best_params_['base_estimator__min_samples_leaf'],
                             min_samples_split = boost_clf_2.best_params_['base_estimator__min_samples_split']
                            )
clf = AdaBoostClassifier(base_estimator = DTC, learning_rate = boost_clf.best_params_['learning_rate'], 
                         n_estimators = boost_clf.best_params_['n_estimators'], random_state = 45)
tester.dump_classifier_and_data(clf, my_dataset, boost_feat_list)
tester.main();
```

    AdaBoostClassifier(algorithm='SAMME.R',
              base_estimator=DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=2,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0, presort=False, random_state=45,
                splitter='best'),
              learning_rate=0.001, n_estimators=50, random_state=45)
    	Accuracy: 0.92307	Precision: 0.67393	Recall: 0.81950	F1: 0.73962	F2: 0.78556
    	Total predictions: 15000	True positives: 1639	False positives:  793	False negatives:  361	True negatives: 12207
    


After tunning our three classifiers, we obtained in general decent values for all the relevant metrics (as all of them comply with the minimum requirement of being above 0.3). It is worth mentioning that we tried this out using a fixed random state for the sake of reproducibility. If we remove this restriction,  the results obtained after running tester.py will change from one run to the next. However, as an average, we expect them to be close to those shown in the table below. 


```python
print("Final Results")
pd.DataFrame([[0.864, 0.489, 0.368, 0.420], [0.907, 0.612, 0.829, 0.704],[0.923, 0.674, 0.819, 0.739]],
             columns = ['Accuracy','Precision', 'Recall', 'F1'], 
             index = ['GaussianNB', 'Decision Tree', 'AdaBoost'])

```

    Final Results





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GaussianNB</th>
      <td>0.864</td>
      <td>0.489</td>
      <td>0.368</td>
      <td>0.420</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.907</td>
      <td>0.612</td>
      <td>0.829</td>
      <td>0.704</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.923</td>
      <td>0.674</td>
      <td>0.819</td>
      <td>0.739</td>
    </tr>
  </tbody>
</table>
</div>



# CONCLUSIONS

We applied three well-known machine learning algorithms to a combination of financial data (FindLaw) and email-related data from the Enron dataset in an attempt to find persons of interest (poi) in the Enron scandal case.
As it is almost always the case in Data Analysis, the preprocessing of the data played an essential role in the development and final results of our project.

The data set provided contained information about 144 people involved (18 of them were poi); which is, by all means, a small amount of data for the intended task. With an initial assessment of 1323 missing entries within which there were 25 "strange cases" of missing email information, the prospects for success were not precisely enjoyable. Fortunately, we learned that the missing (NaN) values in the financial data were in reality zeros and that dramatically reduced our missing entries to a more manageable amount of 285.  After that, and making use of the Enron email dataset, we were able to unravel the mystery of those 25 missing cases and decided in correspondence. We chose to impute the NaN values with the median of their respective columns making the difference between poi and non-poi. That decision was not taken blindly. It was such that it maximized the performance of our classifiers.

After applying Gaussian Naive Bayes, Decision Tree and AdaBoost classifiers to a reduced number of features, we observed that all three of them fulfilled the minimum requirements for the project. Decision Tree and AdaBoost were very close in their performances. We decided to concede the edge to AdaBoost, not because of its marginally better numbers but taking into account that by using Decision Tree inside boosting algorithms, we make it more robust when facing new data.  We tuned AdaBoost using GridSearchCV in two consecutive steps to minimize the running time. In the end, after applying tester.py, our best values (for a definite random state) were as follows: Accuracy: 0.923, Precision: 0.674, Recall 0.819 and F1 0.739. 
We believe that this set of values is indicative of a solid performance by that classifier in this problem, but there is still room for further improvements. 


## Limitations and future work

As we mentioned above, one of the inherent limitations of this project emanates from the small size of the data set, just 144 people (18 poi). The quality of the data also played a significant role, as the government intervention for privacy issues substantially modified part of the data making them useless for this study.
What is exciting about this project is that by using the full email data set, it is possible to create, at least in principle, any number of new and meaningful features. I believe that if we work this process in more detail,  we could end up creating features that could improve further the efficiency of our classifiers. For instance, some features could take into account the flux of emails around the critical dates, or be the result of sentiment analysis and clustering applied to the emails texts.


# REFERENCES

1. http://www.ahschulz.de/enron-email-data/
2. https://enrondata.readthedocs.io/en/latest/data/custodian-names-and-titles/
3. http://www.infosys.tuwien.ac.at/staff/dschall/email/enron-employees.txt
4. https://marcobonzanini.com/2015/02/25/fuzzy-string-matching-in-python/
5. https://regex101.com
8. https://rodgersnotes.wordpress.com/2013/11/19/enron-email-analysis-persons-of-interest/

