import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import sys
from datetime import datetime


def preprocess(data, isTrain):
#    print(data.columns)
#    print(data.head())
    data['HasName'] = data['Name'].fillna(0)
    data.loc[data['HasName'] != 0, "HasName"] = 1
    data['HasName'] = data['HasName'].astype(int)
    data['AnimalType'] = data['AnimalType'].map({'Cat': 0, 'Dog': 1})
    if (isTrain):
        data.drop(['AnimalID', 'OutcomeSubtype'], axis=1, inplace=True)
        import matplotlib
        matplotlib.rcParams['backend'] = "Qt4Agg"
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import pylab
#        for category in np.unique(data["OutcomeType"]):
#            print(category)
#            dfCategory = data[data["OutcomeType"] == category]
#            groups = dfCategory.groupby("Breed")["OutcomeType"].count()
#            colors = np.unique(data['Breed'][:100])
#            groups = groups[colors]
#            plt.figure()
#            groups.plot(kind="bar", title=category + " count by color")
#            pylab.show()
        data['OutcomeType'] = data['OutcomeType'].map(
            {'Return_to_owner': 3, 'Euthanasia': 2, 'Adoption': 0, 'Transfer': 4, 'Died': 1})

#    print(data['SexuponOutcome'].head())
    data['SexuponOutcome'] = data['SexuponOutcome'].fillna('Not specified')
    data['IsNeutered'] = data['SexuponOutcome'].str.contains('Neutered', case=False).astype(int)
    data['IsIntact'] = data['SexuponOutcome'].str.contains('Intact', case=False).astype(int)
    data['IsSpayed'] = data['SexuponOutcome'].str.contains('Spayed', case=False).astype(int)
#
    gender = {'Neutered Male': 1, 'Spayed Female': 2, 'Intact Male': 3, 'Intact Female': 4, 'Unknown': 5, 'Not specified': 0}
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
    data['IsMix'] = data['Breed'].str.contains('Mix', case=False).astype(int)
    data['IsBlack'] = data['Color'].str.contains('Black', case=False).astype(int)
    data['IsWhite'] = data['Color'].str.contains('White', case=False).astype(int)
    data['IsBrown'] = data['Color'].str.contains('Brown', case=False).astype(int)
    data['IsDomestic'] = data['Breed'].str.contains('Domestic', case=False).astype(int)
    data['IsPitbull'] = data['Breed'].str.contains('Pit Bull', case=False).astype(int)
    data['IsShorthair'] = data['Breed'].str.contains('Shorthair', case=False).astype(int)
    data['KindaColor'] = data['IsBlack'] + data['IsWhite']
    breed_split = pd.DataFrame([x.split('/') for x in data['Breed'].tolist()], columns=['Breed1', 'Breed2', 'Breed3'])
    breed_split = breed_split.drop(breed_split.columns[[2]], axis=1).reset_index(drop=True)
    print(breed_split['Breed1'].dtype)
    breed_split['Breed1'] = breed_split['Breed1'].str.replace(' Mix','')
    
    data = pd.concat([data, pd.get_dummies(breed_split['Breed1'], prefix='Breed1'), \
                pd.get_dummies(breed_split['Breed2'], prefix='Breed2')], axis=1)
    
    return data.drop(['AgeuponOutcome', 'Breed', 'Name', 'Color', 'DateTime'], axis=1)


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
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, Embedding
    from keras.utils import np_utils

    print("Loading data...\n")
    pd_train = pd.read_csv(in_file_train)
    pd_test = pd.read_csv(in_file_test)

    print("Preprocess data...\n")
    pd_train = preprocess(pd_train, True)
    pd_test = preprocess(pd_test, False)
    print(pd_train['OutcomeType'])
    test_ids = pd_test['ID']
    pd_test.drop('ID', inplace=True, axis=1)
    
    missing_test = np.setdiff1d(np.array(pd_train.columns), np.array(pd_test.columns))
    missing_train = np.setdiff1d(np.array(pd_test.columns), np.array(pd_train.columns))
    pd_test = pd.concat([pd_test, pd.DataFrame(0, index=np.arange(len(pd_test)), \
            columns=missing_test)], axis=2)
    
    pd_train = pd.concat([pd_train, pd.DataFrame(0, index=np.arange(len(pd_train)), \
            columns=missing_train)], axis=2)
    pd_test.drop('OutcomeType', inplace=True, axis=1)
    targets = pd_train['OutcomeType']
    pd_train.drop('OutcomeType', inplace=True, axis=1)
    pd_train.fillna(0)
    pd_test.fillna(0)
#    pd_test = pd_test.sort(pd_test.columns)
#    pd_train = pd_train.sort(pd_train.columns)
#    
#    print(list(pd_train.columns))
#    print(list(pd_test.columns))
    print(pd_train.shape, pd_test.shape)
    train = pd_train.values
    test = pd_test.values
    
    print(targets)
    print("Calculating best case params...\n")
    #print(best_params(train))
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import decomposition
    
    np.random.seed(5)
    
    centers = [[1, 1], [-1, -1], [1, -1]]
    import matplotlib
    matplotlib.rcParams['backend'] = "Qt4Agg"
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import pylab
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=164)
    
    plt.cla()
    pca = decomposition.PCA(n_components=5)
    pca.fit(train[0::, 0::])
    X = pca.transform(train[0::, 0::][500:700])
    print(type(train[0::, 0]))
    y = targets.astype(int)[500:700]
    print(y)
    for name, label in [('Adoption', 0), ('Died', 1), ('Euthanasia', 2), ('Return_to_owner', 3), ('Transfer', 4)]:
                  ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    print(y)
    y = np.choose(y, [3, 4, 1, 2, 0]).astype(np.int64)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)
    
    x_surf = [X[:, 0].min(), X[:, 0].max(),
              X[:, 0].min(), X[:, 0].max()]
    y_surf = [X[:, 0].max(), X[:, 0].max(),
              X[:, 0].min(), X[:, 0].min()]
    x_surf = np.array(x_surf)
    y_surf = np.array(y_surf)
    v0 = pca.transform(pca.components_[[0]])
    v0 /= v0[-1]
    v1 = pca.transform(pca.components_[[1]])
    v1 /= v1[-1]
    
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    
    plt.show()
    nb_classes = np.max(targets) + 1
    print(nb_classes, 'classes')

    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    Y_train = np_utils.to_categorical(targets, nb_classes)
    #print(train[0::, 0])
    #print(pd_train['OutcomeType'])
    #Y_test = np_utils.to_categorical(test, nb_classes)
    print('Y_train shape:', Y_train.shape)
    #print('Y_test shape:', Y_test.shape)
    batch_size = 32
    nb_epoch = 5

    print(train.shape, test.shape)
    
#    
    train = train.reshape(train.shape + (1,))
    print('Building model...', train[0::, 0::].shape)
    model = Sequential()
    model.add(Convolution1D(
        nb_filter=64,
        filter_length=3,
        input_shape=(413, 1),
        input_length=26729,
        activation='tanh',
        init='uniform'))
    model.add(Activation('relu'))
    model.add(Convolution1D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.25))

#    model.add(Convolution1D(128, 3, border_mode='same'))
#    model.add(Activation('relu'))
#    model.add(Convolution1D(128, 3))
#    model.add(Activation('relu'))
#    model.add(MaxPooling1D(pool_length=2))
#    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    #print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

    history = model.fit(train[0::, 0::], Y_train,
                        nb_epoch=25, batch_size=batch_size,
                        verbose=1, validation_split=0.4)
    print("Predicting... \n")

    test = test.reshape(test.shape + (1,))
    predictions = model.predict_proba(test)
    print(predictions)
#    forest = RandomForestClassifier(n_estimators=1, max_features='auto')
#    forest = forest.fit(train[0::, 1::], train[0::, 0])
#    predictions = forest.predict_proba(test)

    #print(test_ids)


    output = pd.DataFrame(predictions, index=test_ids, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
    #output.columns.names = ['ID']
    #output.index.names = ['ID']
    #output.index += 1

    print("Writing predictions.csv\n")

    print(output)

    output.to_csv('data/predictions_nn_cc_15.csv')

    print("Done.\n")
