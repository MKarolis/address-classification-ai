import pandas as pd
from utils import read_DataFrame_from_file
from prepare import PROPERTY_ADDRESS, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT
from constants import MODEL_FILENAME, RANDOM_STATE, TRAINING_DATA_FILENAME
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    dataFrame: pd.DataFrame = read_DataFrame_from_file(TRAINING_DATA_FILENAME)
    labels = np.array(dataFrame[PROPERTY_LABEL])

    featureFrame = dataFrame.drop([PROPERTY_LABEL, PROPERTY_ADDRESS, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT], axis = 1)
    feature_list = list(featureFrame.columns)
    features = np.array(featureFrame)

    rf = RandomForestClassifier(n_estimators = 1000, random_state = RANDOM_STATE)
    rf.fit(features, labels)

    predicted = rf.predict([[80, 3, 1, 5, 1, 0, 0]])
    print(predicted)

    #joblib.dump(rf, MODEL_FILENAME)