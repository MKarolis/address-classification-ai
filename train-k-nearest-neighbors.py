import pandas as pd
from utils import read_DataFrame_from_file
from prepare import PROPERTY_ADDRESS, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT
from constants import MODEL_FILENAME, RANDOM_STATE, TRAINING_DATA_FILENAME
import numpy as np
import pandas as pd
import joblib

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    le = preprocessing.LabelEncoder()
    model = KNeighborsClassifier(n_neighbors=3)

    dataFrame: pd.DataFrame = read_DataFrame_from_file(TRAINING_DATA_FILENAME)
    label = np.array(dataFrame[PROPERTY_LABEL])
    print(label)

    featureFrame = dataFrame.drop([PROPERTY_LABEL, PROPERTY_ADDRESS, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT], axis = 1)
    features = list(featureFrame.itertuples(index=False, name=None))
    print(features)

    model.fit(features,label)

    predicted = model.predict([[80, 3, 1, 5, 1, 0, 0]])
    print(predicted)

    #joblib.dump(rf, MODEL_FILENAME)

