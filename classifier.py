
#Importing the libraries
import pandas as pd
from utils import read_DataFrame_from_excel, write_DataFrame_to_excel, read_dataFrame_from_csv
from deploy import classify_addresses

DATA_OUTPUT_FILENAME = 'classified.xlsx'


if __name__ == '__main__':
    
    ### Getting filename ###
    filename: str = input('Enter a txt filename with the addresses data that you wish to classify (e.g. input.txt):\n> ')

    ### Reading the dataset ###
    unclassified_addresses = read_DataFrame_from_excel(filename) if filename.endswith('.xlsx') else read_dataFrame_from_csv(filename)
    
    ### Classify addresses ###
    classified_addresses = classify_addresses(unclassified_addresses)

    ### Writing the classified dataset ###
    write_DataFrame_to_excel(classified_addresses, DATA_OUTPUT_FILENAME)
    
    print('\nFirst Five classified addresses:')
    print(classified_addresses.head())
    
    print('\nAll classified addresses saved in ' + DATA_OUTPUT_FILENAME + ' file')
    
    
    
