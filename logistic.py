
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prepare import DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS, PROPERTY_ADDRESS, enrichDataFrameWithProperties, PROPERTY_LABEL, processData
from utils import read_DataFrame_from_excel

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

pd.set_option('display.expand_frame_repr', True)
# pd.set_option('display.max_columns', None)
pd.reset_option('display.max_columns')

### Reading the dataset ###

raw_data = read_DataFrame_from_excel(DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS)

# print('\nReading the dataset')
# print('\nRaw Data:')
# print(raw_data.head())

### Data Pre-Processing ###

# print('\nProcessed Data:')
addresses = processData(raw_data)
# print(addresses.head())

# Checking for missing values
# print('\nMissing values:')
# print(addresses.isnull().sum())


### Exploratory Data Analysis ###

# Number of rows and columns of train set
# print('\nNumber of rows and columns:')
# print(addresses.shape)

# Dataset info
# print('Dataset info: ')
# print(addresses.info())

### Dataset description ###
# print('Dataset description: ')
# description = addresses.describe()
# print(description)



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
# print(corrMatrix)




import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = addresses.drop('label', axis=1)
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))  #print 10 best features



import pandas as pd
import numpy as np
data = addresses.drop('label', axis=1)
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()



### Model Building ###

# Setting the value for dependent and independent variables
# 'label', 'digit_count', 'digits_group_count', 'separated_digits_group_count', 'token_count', 'comma_count', 'comma_separated_entities_having_numbers', 'comma_separated_entities_having_numbers_near_words'
X = addresses.drop(['label',  'separated_digits_group_count', 'comma_separated_entities_having_numbers_near_words', 'digits_group_count', 'comma_count', 'comma_separated_entities_having_numbers'], axis=1)
y = addresses.label

print(X.columns.values)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Fitting the Logistic Regression model
lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_model.fit(X_train, y_train)

#Prediction of test set
y_pred = lr_model.predict(X_test)

#Predicted values
# print(y_pred)

#Actual value and the predicted value
# compareValues = pd.DataFrame({'Actual value': y_test, 'Predicted value':y_pred})
# print(compareValues.head())

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
# print(classification_report(y_test, y_pred))


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("f1-score:", metrics.f1_score(y_test, y_pred))


# ROC Curve

y_pred_proba = lr_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
print("AUC: ", auc)


# Accuracy: 0.952
# Precision: 0.9152542372881356
# Recall: 0.9818181818181818
# f1-score: 0.9473684210526316
# AUC:  0.9724350649350649

# ['length' 'digit_count' 'token_count']
# Accuracy: 0.952
# Precision: 0.9152542372881356
# Recall: 0.9818181818181818
# f1-score: 0.9473684210526316
# AUC:  0.9725