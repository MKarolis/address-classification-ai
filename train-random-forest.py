import pandas as pd
from utils import read_DataFrame_from_file, write_DataFrame_to_excel, get_precision_recall_f1
from prepare import PROPERTY_ADDRESS, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot

RANDOM_STATE = 420
TRAINING_DATA_FILENAME = 'training_data.xlsx'

dataFrame: pd.DataFrame = read_DataFrame_from_file(TRAINING_DATA_FILENAME)
labels = np.array(dataFrame[PROPERTY_LABEL])

featureFrame = dataFrame.drop([PROPERTY_LABEL, PROPERTY_ADDRESS, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT], axis = 1)
feature_list = list(featureFrame.columns)
features = np.array(featureFrame)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size = 0.2, random_state = RANDOM_STATE
)

rf = RandomForestClassifier(n_estimators = 1000, random_state = RANDOM_STATE)
rf.fit(train_features, train_labels)

train_predictions = rf.predict(train_features)
train_accuracy = rf.score(train_features, train_labels) * 100.0
[train_precision, train_recall, train_f1] = get_precision_recall_f1(train_labels, train_predictions)

test_predictions = rf.predict(test_features)
test_accuracy = rf.score(test_features, test_labels) * 100.0
[test_precision, test_recall, test_f1] = get_precision_recall_f1(test_labels, test_predictions)

print('For training data:')
print('Accuracy: {:.2f} | Precision: {:.2f} | Recall: {:.2f} | F1: {:.2f}'
      .format(train_accuracy, train_precision, train_recall, train_f1))
print('For test data:')
print('Accuracy: {:.2f} | Precision: {:.2f} | Recall: {:.2f} | F1: {:.2f}'
      .format(test_accuracy, test_precision, test_recall, test_f1))

importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:50} Importance: {}'.format(*pair)) for pair in feature_importances]

tree = rf.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

predictions = rf.predict(features)
dataFrame['predicted'] = predictions
write_DataFrame_to_excel(dataFrame, 'predicted.xlsx')