import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

def exercise_1_hist():
    data = pd.read_csv("iris.data", header=1,
                       names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    print(data)

    print(len(data[data["class"] == "Iris-setosa"]))

    data["class"].hist()
    data.groupby("class").hist()

    plt.show()


def exercise_1_scatter():
    data = pd.read_csv("iris.data", header=1,
                       names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

    data[data["class"] == "Iris-setosa"].plot.scatter("sepal_length", "sepal_width")

    plt.show()


def exercise_1_lrm():
    data = pd.read_csv("iris.data", header=1,
                       names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

    x = data[data["class"] == "Iris-setosa"]["sepal_length"]
    y = data[data["class"] == "Iris-setosa"]["sepal_width"]

    # Reshape the data according to the requirements of sklearn models
    x = np.array(x)
    print(x)
    x = x.reshape(-1, 1)
    print(x)

    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)

    regressor = LinearRegression()

    # Train the model using the training sets
    regressor.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regressor.predict(x_test)

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.scatter(x_test, y_test, color="black")
    plt.plot(x_test, y_pred, color="blue", linewidth=3)

    plt.show()

    x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = label_binarize(data[['class']], classes=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)

    print(f'Train set shape = {x_train.shape}')
    print(f'Test set shape = {x_test.shape}')

def exercise_2_dtm():

    data = pd.read_csv("iris.data", header=1,
                       names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

    x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = label_binarize(data[['class']], classes=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)

    print(f'Train set shape = {x_train.shape}')
    print(f'Test set shape = {x_test.shape}')

    tree = DecisionTreeClassifier(min_samples_split=10)
    tree.fit(x_train, y_train)

    predictions = tree.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    # accuracy = sum([pred == target for (pred, target) in zip(predictions, y_test)]) / len(predictions)
    print(accuracy)

    scores = cross_val_score(estimator=tree, X=x_train, y=y_train, cv=10)
    print(scores.mean())


    fpr = {}  # false positive rate
    tpr = {}  # true positive rate
    roc_auc = []
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
        roc_auc.append(auc(fpr[i], tpr[i]))

    roc_auc_lib = roc_auc_score(y_test, predictions, average=None)

    print(roc_auc_lib)
    print(roc_auc)

    min_split = []
    for i in range(2, 11):
        clf = DecisionTreeClassifier(
            min_samples_split=i)  # for minimum samples in nodes use  clf = DecisionTreeClassifier(min_samples_split=i)

        # Perform 10-fold cross validation
        scores = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10)
        min_split.append((i, scores.mean()))

        best_result = max([s[1] for s in min_split])

    best_split = [s[0] for s in min_split if s[1] == best_result]
    print(min_split)
    print(best_split, best_result)



    x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()
    y = data['class'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)

    # Create and fit the model
    classifier = KNeighborsClassifier()
    # classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Predict probabilities
    probs_y = classifier.predict_proba(x_test)
    probs_y = np.round(probs_y, 2)
    print(probs_y)

    # Print classifier scores
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred, average="weighted")}')
    print(f'Recall: {recall_score(y_test, y_pred, average="weighted")}')
    print(f'F1: {f1_score(y_test, y_pred, average="weighted")}')


def main():
    # exercise_1_hist()
    # exercise_1_scatter()
    exercise_1_lrm()
    # exercise_2_dtm()


main()
