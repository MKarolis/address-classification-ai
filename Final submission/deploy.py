
import pandas as pd
from prepare import PROPERTY_ADDRESS, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS, enrichDataFrameWithProperties
from constants import MODEL_FILENAME, LOGISTIC_REGRESSION_MODEL_FILENAME
import joblib
from utils import read_DataFrame_from_excel, write_DataFrame_to_excel, read_dataFrame_from_csv

DATA_OUTPUT_FILENAME = 'classified.xlsx'


def process_data(dataFrame: pd.DataFrame):
    """

    Function to process a dataset of addresses to a 
    dataset with derivated features that will be 
    used in our model.

    Args:
        dataFrame (pd.DataFrame): dataset to be processed

    Returns:
        processedData: new processed dataset with derivated features from the original dataset

    """
    processedData = pd.DataFrame()
    processedData[PROPERTY_ADDRESS] = dataFrame['person_address']
    enrichDataFrameWithProperties(processedData)
    
    processedData.drop(PROPERTY_ADDRESS, axis=1, inplace=True)
    
    return processedData
    
def classify_addresses(dataFrame: pd.DataFrame):
    """

    Function responsible to classify a dataset of unclassified
    addresses, using a classifying model, to a new dataset with
    the addresses classified as complete or not.
    
    Args:
        dataFrame (pd.DataFrame): dataset of unclassified addresses

    Returns:
        classified_dataset: new dataset with the addresses classified

    """
    
    ### Data Pre-Processing ###
    unclassified_dataset = process_data(dataFrame)
    
    ### Check which model is using so it can drop the rigth features ###

    unclassified_dataset.drop(PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, axis=1, inplace=True)
    ### Uncomment the next line if using logistic regression ###
    # unclassified_dataset.drop([PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS], axis=1, inplace=True)

    ### Load Model ###raw
    model = joblib.load(MODEL_FILENAME)
    
    ### Prediction of dataset ####
    classified_dataset = dataFrame.copy()
    complete_column = model.predict(unclassified_dataset)
    classified_dataset['complete'] = complete_column
    
    return classified_dataset

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
