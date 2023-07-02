import sklearn
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# sns.set_style('whitegrid')
import pandas as pd
import numpy as np
import shap
shap.initjs()

seed = 42

# loading and spliting the data
diabetes = load_diabetes()
x = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

# building two models without hyperparameter tuning
tree = DecisionTreeRegressor(random_state=seed)
mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500, random_state=seed))

# printing the coeficient of determination R^2
print(tree.fit(x_train, y_train).score(x_test, y_test))
print(mlp.fit(x_train, y_train).score(x_test, y_test))

fig = plt.subplots(figsize=(16, 8))
ax = plt.subplots(figsize=(16, 8))
plot_tree(tree, filled=True)
plt.title("Decision tree for Diabetes dataset")
plt.show()



# Gini feature importance and permutation feature importance for the DT

feature_importance = tree.feature_importances_
sorted_idx = np.argsort(feature_importance) # we sort the feature importances for visualization purposes
pos = np.arange(sorted_idx.shape[0]) + 0.5

fig = plt.figure(figsize=(12, 6))

# in the first subplot we will visualize the gini importance in a bar plot
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(diabetes.feature_names)[sorted_idx])
plt.title("Gini Feature Importance for the DT")

# now we compute the permutation feature importance using the test set
result = permutation_importance(tree, x_test, y_test, n_repeats=10, random_state=seed, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

# in the second subplot we will visualize the permutation feature importance in a box plot
# (since we repeated the process 10 times)
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(diabetes.feature_names)[sorted_idx])
plt.title("Permutation Feature Importance for the DT")
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=(6, 6))

# now we compute the permutation feature importance using the test set for the MLP
result = permutation_importance(mlp, x_test, y_test, n_repeats=10, random_state=seed, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(diabetes.feature_names)[sorted_idx])
plt.title("Permutation Feature Importance for the MLP")
fig.tight_layout()
plt.show()

fig= plt.subplots(figsize=(12, 6))
ax = plt.subplots(figsize=(12, 6))
ax.set_title("Decision Tree")
tree_disp = plot_partial_dependence(tree, x, ["sex", "bmi"], ax=ax)