Steps to generate the files new_data_df.pickle and missing_data_df.pickle

*** All the following scripts must be run in python 3. ***

The script files listed below have to be running in that precise order, because they generate intermediate files. 

1. Deploy the content of the toolbox folder into the final_project root folder.
2. Run manage_emails.py (Requires the folder maildir to exist in ud-120-projects-master)
3. Run create_emails_from_big_data.py (Requires the file AllEmployeesEmails2.csv to exist in the given path)
4. Run find_missing_data.py
5. Run create_new_features.py (Requires the file AllEmployeesEmails2.csv to exist in the given path).

Some of those files take hours to run in a Macbook Pro laptop (Intel i5 processor) with 16 GB RAM and a solid state hard drive.