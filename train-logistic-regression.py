
#Importing the libraries
import pandas as pd
from prepare import DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS, PROPERTY_ADDRESS, enrichDataFrameWithProperties, PROPERTY_LABEL, processData, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS
from utils import read_DataFrame_from_file
from constants import LOGISTIC_REGRESSION_MODEL_FILENAME
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib


if __name__ == '__main__':

    ### Reading the dataset ###
    raw_data = read_DataFrame_from_file(DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS)
    
    
    ### Data Pre-Processing ###
    addresses = processData(raw_data)
    
    ### Model Building ###
    
    # Setting the value for dependent and independent variables
    x = addresses.drop([PROPERTY_LABEL,  PROPERTY_SEPARATED_DIGIT_GROUP_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS_NEAR_WORDS, PROPERTY_DIGITS_GROUP_COUNT, PROPERTY_COMMA_COUNT, PROPERTY_COMMA_SEPARATED_ENTITIES_HAVING_DIGITS], axis=1)
    y = addresses.label
    
    
    # Splitting the dataset
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Fitting the Logistic Regression model
    lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr_model.fit(x, y)
    
    # Prediction of test set
    # y_pred = lr_model.predict(X_test)
    
    joblib.dump(lr_model, LOGISTIC_REGRESSION_MODEL_FILENAME)

