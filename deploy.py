
#Importing the libraries
import pandas as pd
from prepare import DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS, PROPERTY_ADDRESS, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS, enrichDataFrameWithProperties
from utils import write_DataFrame_to_excel
from constants import MODEL_FILENAME, LOGISTIC_REGRESSION_MODEL_FILENAME
import joblib

MODEL = LOGISTIC_REGRESSION_MODEL_FILENAME

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
    if MODEL == LOGISTIC_REGRESSION_MODEL_FILENAME:
        unclassified_dataset.drop([PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS], axis=1, inplace=True)
    else:
        unclassified_dataset.drop(PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, axis=1, inplace=True)

    ### Load Model ###
    model = joblib.load(MODEL_FILENAME)
    
    ### Prediction of dataset ####
    classified_dataset = dataFrame.copy()
    complete_column = model.predict(unclassified_dataset)
    classified_dataset['completed'] = complete_column
    
    return classified_dataset

    
    
