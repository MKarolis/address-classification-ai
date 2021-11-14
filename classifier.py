
#Importing the libraries
import pandas as pd
from utils import write_DataFrame_to_excel, read_file
from deploy import classify_addresses

DATA_OUTPUT_FILENAME = 'classified.xlsx'


if __name__ == '__main__':
    
    ### Getting filename ###
    filename = input('Enter a txt filename with the addresses data that you wish to classify (e.g. input.txt):\n> ')

    ### Reading the dataset ###
    unclassified_addresses = read_file(filename)
    
    ### Classify addresses ###
    classified_addresses = classify_addresses(raw_data)

    ### Writing the classified dataset ###
    write_DataFrame_to_excel(classified_addresses, DATA_OUTPUT_FILENAME)
    
    print('\nFirst Five classified addresses:')
    print(classified_addresses.head())
    
    print('\nAll classified addresses saved in ' + DATA_OUTPUT_FILENAME + ' file')
    
    
    
