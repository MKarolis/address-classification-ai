
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prepare import DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS, PROPERTY_ADDRESS, enrichDataFrameWithProperties, PROPERTY_LABEL, processData
from utils import read_DataFrame_from_file

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

pd.set_option('display.expand_frame_repr', True)
# pd.set_option('display.max_columns', None)
pd.reset_option('display.max_columns')

### Reading the dataset ###

raw_data = read_DataFrame_from_file(DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS)

print('\nReading the dataset')
print('\nRaw Data:')
print(raw_data.head())

### Data Pre-Processing ###

print('\nProcessed Data:')
addresses = processData(raw_data)
print(addresses.head())

# Checking for missing values
print('\nMissing values:')
print(addresses.isnull().sum())


### Exploratory Data Analysis ###

# Number of rows and columns of train set
print('\nNumber of rows and columns:')
print(addresses.shape)

# Dataset info
print('Dataset info: ')
print(addresses.info())

### Dataset description ###
print('Dataset description: ')
description = addresses.describe()
print(description)



## Analysis of length feature

# length distribution
# addresses.length.hist()
# print("The Median length of addres is:", int(dataset.length.median()))


# length group which is more likely to be completed
# sns.lmplot(x='length',y='label',data=addresses)


## Analysis of digit_count feature

# digit_count distribution
# addresses.digit_count.hist()
# sns.countplot('digit_count',data=addresses)
# print(addresses['digit_count'].value_counts())

# digit_count group which is more likely to be complted
# sns.lmplot(x='digit_count',y='label',data=addresses)

# Percentage of addresses completed grouped by digit_count
# sns.barplot(x='digit_count',y='label',data=addresses)
# print(addresses.groupby('digit_count',as_index=False).label.mean())
# sns.countplot(x='label', hue='digit_count', data=addresses)



## Analysis of digits_group_count feature

# addresses = addresses[addresses['digits_group_count'] != 0]

# count of addresses based of digits_group_count
# sns.countplot('digits_group_count',data=addresses)
# print(addresses['digits_group_count'].value_counts())

# Percentage of addresses completed grouped by digits_group_count
# sns.barplot(x='digits_group_count',y='label',data=addresses)
# print(addresses.groupby('digits_group_count',as_index=False).label.mean())
# sns.lmplot(x='digits_group_count',y='label',data=addresses)


# Count of addresses completed based on digits_group_count
# sns.countplot(x='label', hue='digits_group_count', data=addresses)



## Analysis of separated_digits_group_count feature

# count of addresses based of separated_digits_group_count
# sns.countplot('separated_digits_group_count',data=addresses)
# print(addresses['separated_digits_group_count'].value_counts())

# Percentage of addresses completed grouped by separated_digits_group_count
# sns.barplot(x='separated_digits_group_count',y='label',data=addresses)
# print(addresses.groupby('separated_digits_group_count',as_index=False).label.mean())
# sns.lmplot(x='separated_digits_group_count',y='label',data=addresses)

# Count of addresses completed based on separated_digits_group_count
# sns.countplot(x='label', hue='separated_digits_group_count', data=addresses)



## Analysis of token_count feature

# addresses = addresses[addresses['token_count'] > 3]

# count of addresses based of token_count
# sns.countplot('token_count',data=addresses)
# print(addresses['token_count'].value_counts())

# Percentage of addresses completed grouped by token_count
# sns.barplot(x='token_count',y='label',data=addresses)
# print(addresses.groupby('token_count',as_index=False).label.mean())
# sns.lmplot(x='token_count',y='label',data=addresses)

# Count of addresses completed based on token_count
# sns.countplot(x='label', hue='token_count', data=addresses)



## Analysis of comma_count feature

# count of addresses based of comma_count
# sns.countplot('comma_count',data=addresses)
# print(addresses['comma_count'].value_counts())

# Percentage of addresses completed grouped by comma_count
# sns.barplot(x='comma_count',y='label',data=addresses)
# print(addresses.groupby('comma_count',as_index=False).label.mean())
# sns.lmplot(x='comma_count',y='label',data=addresses)

# Count of addresses completed based on comma_count
# sns.countplot(x='label', hue='comma_count', data=addresses)



## Analysis of comma_separated_entities_having_numbers feature

# count of addresses based of comma_separated_entities_having_numbers
# sns.countplot('comma_separated_entities_having_numbers',data=addresses)
# print(addresses['comma_separated_entities_having_numbers'].value_counts())

# Percentage of addresses completed grouped by comma_separated_entities_having_numbers
# sns.barplot(x='comma_separated_entities_having_numbers',y='label',data=addresses)
# print(addresses.groupby('comma_separated_entities_having_numbers',as_index=False).label.mean())
# sns.lmplot(x='comma_separated_entities_having_numbers',y='label',data=addresses)

# Count of addresses completed based on comma_separated_entities_having_numbers
# sns.countplot(x='label', hue='comma_separated_entities_having_numbers', data=addresses)


## Analysis of comma_separated_entities_having_numbers_near_words feature

# count of addresses based of comma_separated_entities_having_numbers_near_words
# sns.countplot('comma_separated_entities_having_numbers_near_words',data=addresses)
# print(addresses['comma_separated_entities_having_numbers_near_words'].value_counts())

# Percentage of addresses completed grouped by comma_separated_entities_having_numbers_near_words
# sns.barplot(x='comma_separated_entities_having_numbers_near_words',y='label',data=addresses)
# print(addresses.groupby('comma_separated_entities_having_numbers_near_words',as_index=False).label.mean())
# sns.lmplot(x='comma_separated_entities_having_numbers_near_words',y='label',data=addresses)

# Count of addresses completed based on comma_separated_entities_having_numbers_near_words
# sns.countplot(x='label', hue='comma_separated_entities_having_numbers_near_words', data=addresses)

## Analysis of label feature

# Count of the passengers survived
# sns.countplot('label',data=addresses)
# print(addresses['label'].value_counts())


## Correlation Matrix
corrMatrix = addresses.corr()
print(corrMatrix)



### Model Building ###

# Setting the value for dependent and independent variables
X = addresses.drop('label', 1)
y = addresses.label

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Fitting the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

#Prediction of test set
y_pred = lr_model.predict(X_test)

#Predicted values
print(y_pred)

#Actual value and the predicted value
compareValues = pd.DataFrame({'Actual value': y_test, 'Predicted value':y_pred})
print(compareValues.head())

#Confusion matrix 
cnf_matrix = confusion_matrix(y_test, y_pred)

# sns.heatmap(matrix, annot=True, fmt="d")
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')

# class_names=[0,1] # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')


# Classification report
print(classification_report(y_test, y_pred))

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("f1-score:", metrics.f1_score(y_test, y_pred))






