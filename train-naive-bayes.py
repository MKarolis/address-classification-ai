import pandas as pd
from utils import read_DataFrame_from_file
from utils import get_precision_recall_f1
from prepare import PROPERTY_ADDRESS, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT
from constants import MODEL_FILENAME, RANDOM_STATE, TRAINING_DATA_FILENAME
import numpy as np
import pandas as pd
import joblib


from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics


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

    """model.fit(features,label)

    predicted = model.predict([[80, 3, 1, 5, 1, 0, 0]])
    print(predicted)"""

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=RANDOM_STATE)

    model.fit(X_train, y_train)

    predicted = model.predict(X_test)

    #print("Accuracy:", metrics.accuracy_score(y_test, predicted))

    train_features, test_features, train_labels, test_labels = train_test_split(features, label, test_size=0.3,
                                                                                random_state=109)

    train_predictions = model.predict(train_features)
    train_accuracy = model.score(train_features, train_labels) * 100.0
    [train_precision, train_recall, train_f1] = get_precision_recall_f1(train_labels, train_predictions)

    test_predictions = model.predict(test_features)
    test_accuracy = model.score(test_features, test_labels) * 100.0
    [test_precision, test_recall, test_f1] = get_precision_recall_f1(test_labels, test_predictions)

    print('For training data:')
    print('Accuracy: {:.2f} | Precision: {:.2f} | Recall: {:.2f} | F1: {:.2f}'
          .format(train_accuracy, train_precision, train_recall, train_f1))
    print('For test data:')
    print('Accuracy: {:.2f} | Precision: {:.2f} | Recall: {:.2f} | F1: {:.2f}'
          .format(test_accuracy, test_precision, test_recall, test_f1))

    #joblib.dump(rf, MODEL_FILENAME)

