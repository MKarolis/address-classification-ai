import pandas as pd
from prepare import enrichDataFrameWithProperties, DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS, PROPERTY_ADDRESS, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS
from utils import read_DataFrame_from_excel
from constants import MODEL_FILENAME
from sklearn.linear_model import LogisticRegression
import joblib

def process_data(dataFrame: pd.DataFrame):
    """

    Function to process a training dataset of addresses to a 
    dataset with derivated features that will be 
    used to train our model, adding the feature label
    so it can be trained.

    Args:
        dataFrame (pd.DataFrame): dataset to be processed

    Returns:
        processedData: new processed dataset with derivated features from the original dataset

    """
    
    processedData = pd.DataFrame()
    processedData[PROPERTY_ADDRESS] = dataFrame['person_address']
    enrichDataFrameWithProperties(processedData)
    processedData[PROPERTY_LABEL] = dataFrame['label']
    
    processedData.drop(PROPERTY_ADDRESS, axis=1, inplace=True)
    
    return processedData


if __name__ == '__main__':

    ### Reading the dataset ###
    raw_data = read_DataFrame_from_excel(DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS)
    
    ### Data Pre-Processing ###
    addresses = process_data(raw_data)
    
    ### Model Building ###
    
    ### Setting the value for dependent and independent variables for logistic regression ###
    x = addresses.drop([PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS], axis=1)
    y = addresses.label
        
    ### Fitting the Logistic Regression model ###
    lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr_model.fit(x, y)
        
    ### Saving model to a file ###
    joblib.dump(lr_model, MODEL_FILENAME)

