
#Importing the libraries
import pandas as pd
from prepare import DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS, PROPERTY_ADDRESS, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS, enrichDataFrameWithProperties
from utils import write_DataFrame_to_excel
from constants import LOGISTIC_REGRESSION_MODEL_FILENAME, RANDOM_FOREST_MODEL_FILENAME, NAIVE_BAYES_MODEL_FILENAME
import joblib

DATA_INPUT_FILENAME = 'input.txt'
DATA_OUTPUT_FILENAME = 'classified_addresses.xlsx'

def read_file():
    return pd.read_csv(DATA_INPUT_FILENAME, delimiter='\t', keep_default_na=False)

def process_data(dataFrame: pd.DataFrame):
    
    processedData = pd.DataFrame()
    processedData[PROPERTY_ADDRESS] = dataFrame['person_address']
    enrichDataFrameWithProperties(processedData)
    processedData.drop(PROPERTY_ADDRESS, axis=1, inplace=True)
    return processedData
    
def classifier(dataFrame: pd.DataFrame, model: str):
    
    ### Data Pre-Processing ###
    unclassified_addresses = process_data(dataFrame)
    
    if model == LOGISTIC_REGRESSION_MODEL_FILENAME:
        unclassified_addresses.drop([PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS], axis=1, inplace=True)
    else:
        unclassified_addresses.drop(PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, axis=1, inplace=True)
        print(unclassified_addresses.head())

    ### Load Model ###
    model = joblib.load(model)
    
    ### Prediction of dataset ####
    classified_addresses = raw_data.copy()
    complete_column = model.predict(unclassified_addresses)
    classified_addresses['completed'] = complete_column
    
    return classified_addresses

if __name__ == '__main__':

    ### Reading the dataset ###
    raw_data = read_file()
    
    ### Classify addresses ###
    classified_addresses = classifier(raw_data, RANDOM_FOREST_MODEL_FILENAME)

    ### Writing the classified dataset ###
    write_DataFrame_to_excel(classified_addresses, DATA_OUTPUT_FILENAME)
    
    
