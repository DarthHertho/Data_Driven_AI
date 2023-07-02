
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from pandas import read_csv
from datetime import datetime

from keras import models
from keras import layers
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error

def Classification_MLP():
    data = pd.read_csv("iris.data", header= None,
                       names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])


    x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

    y = data["class"].values
    y = label_binarize(y, classes=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    # x = data[data["class"] == "Iris-setosa"]["sepal_length"]
    # y = data[data["class"] == "Iris-setosa"]["sepal_width"]
    #


    # Reshape the data according to the requirements of sklearn models
    # x = np.array(x)
    # print(x)
    # x = x.reshape(-1, 1)
    # print(x)
    #
    # y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(4,)))
    network.add(layers.Dense(3, activation='softmax'))
    # Compile network
    network.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    history = network.fit(x_train, y_train, epochs=20, batch_size=40)
    # Accuracy
    plt.plot(history.history['accuracy'])
    plt.title('Network Train Accuracy')
    plt.show()
    # Loss
    plt.plot(history.history['loss'])
    plt.title('Network Train Loss')
    plt.show()
    test_loss, test_acc = network.evaluate(x_test, y_test)

    print('Test Accuracy: ', test_acc, '\nTest Loss: ', test_loss)

    # # Train the model using the training sets
    # regressor.fit(x_train, y_train)
    #
    # # Make predictions using the testing set
    # y_pred = regressor.predict(x_test)

    # # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # # The coefficient of determination: 1 is perfect prediction
    # print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
    #
    # # Plot outputs
    # plt.scatter(x_test, y_test, color="black")
    # plt.plot(x_test, y_pred, color="blue", linewidth=3)



    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)
    #
    # print(f'Train set shape = {x_train.shape}')
    # print(f'Test set shape = {x_test.shape}')
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')




def time_series_fc_LSTM():
    dataset= pd.read_csv("airquality.csv", parse_dates= [['year','month','day', 'hour']], index_col=0, date_parser= parse)

    dataset.drop('No', axis=1, inplace=True)

    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'

    # the pollution values for the first 24 hours are NaN
    # the corresponding rows can be removed as they are at the beginning of the data
    dataset = dataset [24:]
    print(dataset)

    # other NaN values cannot be removed as the time sequentiality would be lost
    # mark them with mean of the column
    dataset['pollution'].fillna(dataset['pollution'].mean(), inplace=True)

    # summarize first 5 rows
    dataset.head()

    # prepare data for lstm

    # get values from dataframe as numpy array
    values = dataset.values
    print(values)

    # encode wind direction to integer values
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    print(values[:, 4])

    # ensure all data is float
    values = values.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # convert series to supervised learning
    # n_in: number of timestep to consider as input
    # n_out: number of timestep to be predicted
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = data.shape[1]
        df = DataFrame(data)
        cols, names = [], []

        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df[0].shift(-i))
            if i == 0:
                #             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                names += [('var1(t)')]
            else:
                #             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
                names += [('var1(t+%d)' % (i))]

        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names

        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        return agg

    reframed = series_to_supervised(scaled, 1, 1)

    print(reframed)

    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24
    n_val_hours = 365 * 24
    train = values[:n_train_hours, :]
    val = values[n_train_hours:n_train_hours + n_val_hours, :]
    test = values[n_train_hours + n_val_hours:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    val_X, val_y = val[:, :-1], val[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=64, validation_data=(val_X, val_y), verbose=2,
                        shuffle=False)


    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


def main():
    # Classification_MLP()
    time_series_fc_LSTM()

main()