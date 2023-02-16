import csv
import json
def csv_to_dict(csv_file_path, json_file_path, key_column):
    #create a dictionary
    data_dict = {}
 
    #Step 2
    #open a csv file handler
    with open(csv_file_path, encoding = 'utf-8') as csv_file_handler:
        csv_reader = csv.DictReader(csv_file_handler)
 
        #convert each row into a dictionary
        #and add the converted data to the data_variable
 
        for rows in csv_reader:
 
            #assuming a column named 'No'
            #to be the primary key
            key = rows[key_column]
            data_dict[key] = rows
 
    return data_dict