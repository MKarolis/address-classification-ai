import pandas as pd
from utils import read_DataFrame_from_excel
from prepare import PROPERTY_ADDRESS, PROPERTY_LABEL, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT
from constants import MODEL_FILENAME, TRAINING_DATA_FILENAME
import numpy as np
import pandas as pd
import joblib
from sklearn.naive_bayes import GaussianNB


if __name__ == '__main__':
    model = GaussianNB()

    dataFrame: pd.DataFrame = read_DataFrame_from_excel(TRAINING_DATA_FILENAME)
    label = np.array(dataFrame[PROPERTY_LABEL])

    featureFrame = dataFrame.drop([PROPERTY_LABEL, PROPERTY_ADDRESS, PROPERTY_SEPARATED_DIGIT_GROUP_COUNT], axis = 1)
    features = list(featureFrame.itertuples(index=False, name=None))

    model.fit(features, label)

    joblib.dump(model, MODEL_FILENAME)

