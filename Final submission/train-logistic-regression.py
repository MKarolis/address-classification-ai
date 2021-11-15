import pandas as pd
from prepare import PROPERTY_ADDRESS, enrichDataFrameWithProperties, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS
from utils import read_DataFrame_from_excel
from constants import MODEL_FILENAME, TRAINING_DATA_FILENAME
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def process_data(dataFrame: pd.DataFrame):
    
    processedData = pd.DataFrame()
    processedData[PROPERTY_ADDRESS] = dataFrame[PROPERTY_ADDRESS]
    enrichDataFrameWithProperties(processedData)
    processedData[PROPERTY_LABEL] = dataFrame[PROPERTY_LABEL]
    
    processedData.drop(PROPERTY_ADDRESS, axis=1, inplace=True)
    
    return processedData


if __name__ == '__main__':

    ### Reading the dataset ###
    addresses: pd.DataFrame = process_data(read_DataFrame_from_excel(TRAINING_DATA_FILENAME))
    
    ### Model Building ###
    
    # Setting the value for dependent and independent variables
    x = addresses.drop([PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS], axis=1)
    y = addresses.label
    
    #Fitting the Logistic Regression model
    lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr_model.fit(x, y)
        
    joblib.dump(lr_model, MODEL_FILENAME)
