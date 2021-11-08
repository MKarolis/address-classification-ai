import pandas as pd
from utils import read_DataFrame_from_file
from prepare import PROPERTY_ADDRESS, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT
from constants import MODEL_FILENAME, RANDOM_STATE, TRAINING_DATA_FILENAME
import numpy as np
import pandas as pd
import joblib

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

if __name__ == '__main__':
    le = preprocessing.LabelEncoder()
    model = DecisionTreeClassifier()

    dataFrame: pd.DataFrame = read_DataFrame_from_file(TRAINING_DATA_FILENAME)
    label = np.array(dataFrame[PROPERTY_LABEL])
    print(label)

    featureFrame = dataFrame.drop([PROPERTY_LABEL, PROPERTY_ADDRESS, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT], axis = 1)

    feature_list = list(featureFrame.columns)
    features = np.array(featureFrame)

    #features = list(featureFrame.itertuples(index=False, name=None))
    print(features)

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=RANDOM_STATE)

    model.fit(X_train,y_train)

    predicted = model.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, predicted))

    #joblib.dump(rf, MODEL_FILENAME)

