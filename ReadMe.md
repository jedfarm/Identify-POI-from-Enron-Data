# Identify POI from Enron data      <img align="right" src="https://github.com/jedfarm/Identify-POI-from-Enron-Data/blob/master/Logo_Enron.png" width="200" height="200" />

Enron, once called "America's most innovative company," went from reaching dramatic economic heights to bankruptcy in about a year, turning into one of the biggest financial scandals in America's history. 
As part of the trials that followed, a dataset with most of the emails in Enron's servers was released to the public. Almost two decades later, it remains the biggest of its kind in the public domain and one of the most studied.


### Objective:

To identify Persons Of Interest (POI) using email and financial data from the top Enron employees, applying Machine Learning algorithms in Python.

### Algorithms tested:

- Naives Bayes
- Decision Tree
- AdaBoost

### Data preprocessing
Steps to generate the files new_data_df.pickle and missing_data_df.pickle

 All the following scripts run in **Python 3**.  Keeping the order while executing them is of paramount importance because they generate intermediate files.

1. Deploy the content of the toolbox folder into the final_project root folder.
2. Run manage_emails.py (Requires the folder maildir, with all the emails, to exist in the working folder)
3. Run create_emails_from_big_data.py (Requires the file AllEmployeesEmails2.csv to exist in the given path)
4. Run find_missing_data.py
5. Run create_new_features.py (Requires the file AllEmployeesEmails2.csv to exist in the given path).

_Note that some of those files take hours to run in a MacBook Pro laptop (Intel i5 processor) with 16 GB RAM and a solid state hard drive._

## Files

- QUESTIONS RELATED TO THE PROJECT.pdf
- enron61702insiderpay.pdf (financial data)
- final_project_dataset.pkl
- final_project_dataset_modified.pkl
- missing_data_df.pickle (dataframe containing additional email related data found by us digging deeper into the email dataset)
- my_classifier.pkl
- my_dataset.pkl   (dictionary with the last version of the dataset, after cleaned and incorporated new features)
- my_feature_list.pkl
- new_data_df.pickle
- poi_email_addresses.py
- poi_id.ipynb (source code + comments, Python 2.7)
- poi_id.py (source code, Python 2.7)
- poi_names.txt
- tester.py (create a standardized measure of the performance for each of the methods tested, it has been written in Python 2.7 and was the main cause why poi_id.py had to be written in Python 2.7)
- tester.pyc
- ReadMe.md


### Structure of the dataset

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


### Counting the number of poi and non-poi in the dataset.


```python
df['poi'].value_counts()
```

    0    128
    1     18
    Name: poi, dtype: int64


Where 0: Non-poi  and 1: poi

### New features

Having the Enron email dataset in a workable shape makes possible to create any number of new features. In this study we are going to try some that are very easy to create. We proposed four new features, belonging to two different kinds. Two were ratios of the existing features, and the other two were the result of working with the entire email dataset. 
In the second case, we created an intermediate feature, called pubIndex.  This one is not going to be used explicitly, but it is part of the process.

 pubIndex accounts for the number of people involved in a given email (To and Cc fields) correcting for when people sent emails to themselves. The lowest possible value for this feature is zero (if someone sent an email just to him or herself with no Cc), it is equal to one if there is only a single person in the To field and none in the Cc field, and so on. It is worth noticing that there is in principle no upper limit for this feature.  

### Ratios of existing features:

- to_poi_rate: ratio of from_this_person_to_poi / from_messages

- from_poi_rate: ratio of from_poi_to_this_person / to_messages

### New features from the email dataset:

- from_messages_median_pubIndex: We grouped all the emails sent by a given person and took the median of the pubIndex feature.

- to_poi_median_pubIndex: The same as above but considering just when sending messages to poi.


 ###   Total number of features:  23



After tunning our three classifiers, we obtained in general decent values for all the relevant metrics (as all of them comply with the minimum requirement of being above 0.3). It is worth mentioning that we tried this out using a fixed random state for the sake of reproducibility. If we remove this restriction,  the results obtained after running tester.py will change from one run to the next. However, as an average, we expect them to be close to those shown in the table below. 


### Final Results

<div>

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





# REFERENCES

1. http://www.ahschulz.de/enron-email-data/
2. https://enrondata.readthedocs.io/en/latest/data/custodian-names-and-titles/
3. http://www.infosys.tuwien.ac.at/staff/dschall/email/enron-employees.txt
4. https://marcobonzanini.com/2015/02/25/fuzzy-string-matching-in-python/
5. https://regex101.com
6. https://rodgersnotes.wordpress.com/2013/11/19/enron-email-analysis-persons-of-interest/
7. http://scikit-learn.org/stable/documentation.html
8. https://rapidminer.com/blog/validate-models-training-test-error/
9. https://www.udemy.com/machinelearning/
