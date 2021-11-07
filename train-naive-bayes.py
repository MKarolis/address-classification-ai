import pandas as pd
from utils import read_DataFrame_from_file
from prepare import PROPERTY_ADDRESS, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT
from constants import MODEL_FILENAME, RANDOM_STATE, TRAINING_DATA_FILENAME
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    le = preprocessing.LabelEncoder()
    model = GaussianNB()

    """words_info = pd.DataFrame({
        'chars': [32, 86, 59, 90, 92, 31, 40, 76, 6],
        'tokens': [5, 12, 8, 10, 13, 5, 4, 9, 2],
        'commas': [1, 4, 4, 4, 7, 1, 2, 4, 0],
        'label': [1, 1, 1, 0, 1, 1, 1, 0, 0]
    })

    data = words_info.drop(['label'], axis=1)
    label = np.array(words_info['label'])

    features = list(data.itertuples(index=False, name=None))

    model.fit(features,label)
    predicted = model.predict([[5,3,3]])

    print(features)
    print(label)
    print(predicted)"""

#########################################################################################################

    dataFrame: pd.DataFrame = read_DataFrame_from_file(TRAINING_DATA_FILENAME)
    label = np.array(dataFrame[PROPERTY_LABEL])
    print(label)

    featureFrame = dataFrame.drop([PROPERTY_LABEL, PROPERTY_ADDRESS, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT], axis = 1)
    features = list(featureFrame.itertuples(index=False, name=None))
    print(features)

    model.fit(features,label)

    predicted = model.predict([[90, 0, 0, 0, 0, 0, 0]])
    print(predicted)

    #joblib.dump(rf, MODEL_FILENAME)

