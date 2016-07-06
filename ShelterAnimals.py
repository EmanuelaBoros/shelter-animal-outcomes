import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import sys
from datetime import datetime


def munge(data, train):
    data['HasName'] = data['Name'].fillna(0)
    data.loc[data['HasName'] != 0, "HasName"] = 1
    data['HasName'] = data['HasName'].astype(int)
    data['AnimalType'] = data['AnimalType'].map({'Cat': 0, 'Dog': 1})

    if (train):
        data.drop(['AnimalID', 'OutcomeSubtype'], axis=1, inplace=True)
        data['OutcomeType'] = data['OutcomeType'].map(
            {'Return_to_owner': 4, 'Euthanasia': 3, 'Adoption': 0, 'Transfer': 5, 'Died': 2})

    gender = {'Neutered Male': 1, 'Spayed Female': 2, 'Intact Male': 3, 'Intact Female': 4, 'Unknown': 5, np.nan: 0}
    data['SexuponOutcome'] = data['SexuponOutcome'].map(gender)

    def agetodays(x):
        try:
            y = x.split()
        except:
            return None
        if 'year' in y[1]:
            return float(y[0]) * 365
        elif 'month' in y[1]:
            return float(y[0]) * (365 / 12)
        elif 'week' in y[1]:
            return float(y[0]) * 7
        elif 'day' in y[1]:
            return float(y[0])

    data['AgeInDays'] = data['AgeuponOutcome'].map(agetodays)
    data.loc[(data['AgeInDays'].isnull()), 'AgeInDays'] = data['AgeInDays'].median()

    data['Year'] = data['DateTime'].str[:4].astype(int)
    data['Month'] = data['DateTime'].str[5:7].astype(int)
    data['Day'] = data['DateTime'].str[8:10].astype(int)
    data['Hour'] = data['DateTime'].str[11:13].astype(int)
    data['Minute'] = data['DateTime'].str[14:16].astype(int)

    data['Name+Gender'] = data['HasName'] + data['SexuponOutcome']
    data['Type+Gender'] = data['AnimalType'] + data['SexuponOutcome']
    data['IsMix'] = data['Breed'].str.contains('mix', case=False).astype(int)

    return data.drop(['AgeuponOutcome', 'Name', 'Breed', 'Color', 'DateTime'], axis=1)


def best_params(data):
    rfc = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 400],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(data[0::, 1::], data[0::, 0])
    return CV_rfc.best_params_


if __name__ == "__main__":
    in_file_train = 'data/train.csv'
    in_file_test = 'data/test.csv'

    from keras.layers import Dense, Dropout, Activation, Lambda
    from keras.datasets import reuters
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.utils import np_utils

    print("Loading data...\n")
    pd_train = pd.read_csv(in_file_train)
    pd_test = pd.read_csv(in_file_test)

    print("Munging data...\n")
    pd_train = munge(pd_train, True)
    pd_test = munge(pd_test, False)

    pd_test.drop('ID', inplace=True, axis=1)

    train = pd_train.values
    test = pd_test.values

    print("Calculating best case params...\n")
    #print(best_params(train))

    nb_classes = np.max(train[0::, 0]) + 1
    print(nb_classes, 'classes')

    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    Y_train = np_utils.to_categorical(train[0::, 0], nb_classes)
    #Y_test = np_utils.to_categorical(test, nb_classes)
    print('Y_train shape:', Y_train.shape)
    #print('Y_test shape:', Y_test.shape)
    batch_size = 32
    nb_epoch = 5

    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(len(train[0::, 1::][0]),)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(train[0::, 1::], Y_train,
                        nb_epoch=1, batch_size=batch_size,
                        verbose=1, validation_split=0.1)

    print("Predicting... \n")

    predictions = model.predict_proba(test)

    #forest = RandomForestClassifier(n_estimators=400, max_features='auto')
    #forest = forest.fit(train[0::, 1::], train[0::, 0])
    #predictions = forest.predict_proba(test)

    output = pd.DataFrame(predictions, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
    output.columns.names = ['ID']
    output.index.names = ['ID']
    output.index += 1

    print("Writing predictions.csv\n")

    print(output)

    output.to_csv('data/predictions_nn.csv')

    print("Done.\n")
