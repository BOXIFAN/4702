import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score

# get the data from files
df_classif = pd.read_csv('w3classif.csv',names=['x','y','label'])
df_regr = pd.read_csv('w3regr.csv', names=['x', 'y'])

# classification dataset
print("Classfication dataset head: ")
print(df_classif.head())
# regression dataset
print("Regression dataset head: ")
print(df_regr.head())
print(df_classif.columns)
print()

# Q1.  Make scatterplots of each dataset so you can see what they look like.
# Make classfication dataset visible
def plot_classification_data(df_classif):
    plt.figure()
    plt.title('Classification dataset scatterplotes')
    class_0 = df_classif[df_classif['label'] == 0]
    class_1 = df_classif[df_classif['label'] == 1]

    plt.scatter(class_0['x'], class_0['y'], label='Class 0')
    plt.scatter(class_1['x'], class_1['y'], label='Class 1')
#draw picture
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_classification_data(df_classif)
# Make regression dataset visible
def plot_regression_data(df_regr):
    plt.figure()
    plt.title('Regression dataset scatterplotes')
    # draw(x,y)point
    plt.scatter(df_regr['x'], df_regr['y'], label='Regression dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()
plot_regression_data(df_regr)

# Q2.  Randomly shuffle the datasets (i.e. the order of the rows) and split them each into
#  70% (for training) and 30% (for testing).
x_classif = df_classif[['x','y']] # get the features and labels from data frame
y_classif = df_classif['label']

# use train_test_split get the training set and test set
#for classification
x_classif_train, x_classif_test, y_classif_train, y_classif_test = train_test_split(
    x_classif, y_classif, test_size=0.3, random_state=42, shuffle=True) # random value to prevent sequence error
# for regression
x_regr = df_regr[['x']]
y_regr = df_regr[['y']]
x_regr_train, x_regr_test, y_regr_train, y_regr_test = train_test_split(
    x_regr, y_regr, test_size=0.3, random_state=42, shuffle=True)
# test data size
print("Classification training set size: ", len(x_classif_train))
print("Classification test set size: ", len(x_classif_test))
print("Regression training set size: ", len(x_regr_train))
print("Regression test set size: ", len(x_regr_test))
print("")

# Q3(a)Build a k-NN classifier with k = 3 for dataset w3classif.csv and find the training
#  and test loss (i.e. misclassification rate).
knn_classif = KNeighborsClassifier(n_neighbors=3) # create k-mn classfier when k=3
knn_classif.fit(x_classif_train, y_classif_train)
# use model to predict the training set and test set
y_train_pre = knn_classif.predict(x_classif_train)
y_test_pre = knn_classif.predict(x_classif_test)

# calculate accuracy
train_accuracy = accuracy_score(y_classif_train, y_train_pre)
test_accuracy = accuracy_score(y_classif_test, y_test_pre)

# error rate = 1 - accuracy rate
train_error = 1 - train_accuracy
test_error = 1 - test_accuracy

print("Training error rate: ", train_error) # 4.3%
print("Test error rate: ", test_error) #0.83% those data has very low error rate, it means k=3 are good clasifer
print("")

#Q3 (b)Plot the decision regions for your classifier together with the training and/or test
#data points.
def plot_decision_boundary(model, x, y, title = "Decision Boundary " ):
    h = 0.05
    # get the boundary coordinate range
    x_min, x_max = x['x'].min() - 1, x['x'].max() + 1
    y_min, y_max = x['y'].min() - 1, x['y'].max() + 1
    # create grid points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # stitching grid points as model input
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(pd.DataFrame(grid_points, columns=['x', 'y']))
    Z = Z.reshape(xx.shape)
    # draw graph
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(x['x'], x['y'], c=y)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
plot_decision_boundary(knn_classif, x_classif_train, y_classif_train,title="k-NN Decision Region (k=3)")

# Q3（c）Experiment with different k values and see how it affects the loss values and the
#  decision regions
def plot_decision_boundary_diffKNN(x,y,title = "KNN Decision Region"):
    k_values = [1,5,7,9,13,15,20]
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x,y)
        # create grid
        h = 0.05
        x_min, x_max = x['x'].min() - 1, x['x'].max() + 1
        y_min, y_max = x['y'].min() - 1, x['y'].max() + 1
        # create grid points
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # stitching grid points as model input
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(pd.DataFrame(grid_points, columns=['x', 'y']))
        Z = Z.reshape(xx.shape)
        # draw graph
        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(x['x'], x['y'], c=y)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
plot_decision_boundary_diffKNN(x_classif_train, y_classif_train) # while k=5 it is more smooth, better than k=3

# Q4(a) Build a k-NN regression model with k = 3 for dataset w3regr.csv and find the
#  training and test loss (i.e. sum of squared error)
# create k-mn regression
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(x_regr_train, y_regr_train.values.ravel())
# predict on training and test set
y_train_pre = knn_reg.predict(x_regr_train)
y_test_pre = knn_reg.predict(x_regr_test)
# calculate SSE
# training

train_sse = np.sum((y_regr_train.values.ravel() - y_train_pre) ** 2)

# testing
test_sse = np.sum((y_regr_test.values.ravel() - y_test_pre) ** 2)



print("Trining SSE: ", train_sse)
print("Test SSE: ", test_sse)


# Q4(b) Plot the training and/or test data together with the predicted “function” of the
#  model.
# sort the data make the graph looks consistent
x_plot  = x_regr_test.sort_values(by='x')
y_true = y_regr_test.loc[x_plot.index]
y_pred = knn_reg.predict(x_plot)

#draw
plt.figure()
plt.scatter(x_plot, y_true, color='blue', label='True y (Test Data)')
plt.plot(x_plot, y_pred, color='red', label='Predicted y (kNN Regression)')
plt.title('k-NN Regression: True vs Predicted on Test Set')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Q4(c) Experiment with different k values and see how it affects the loss values and the
#  predicted function
k_values = [1, 5, 7, 9, 13, 15, 20]
plt.figure(figsize=(15, 10))

for i, k in enumerate(k_values):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(x_regr_train, y_regr_train.values.ravel())

    # predict after sort
    x_sorted = x_regr_test.sort_values(by='x')
    y_true = y_regr_test.loc[x_sorted.index]
    y_pred = model.predict(x_sorted)

    # plot each subplot
    plt.subplot(3, 3, i + 1)
    plt.plot(x_sorted, y_true, color='blue', label='True y (Test Data)')
    plt.plot(x_sorted, y_pred, color='red', label='Predicted y (kNN Regression)')
    plt.title(f'k = {k}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle('k-NN Regression with Different k values', fontsize=16, y=1.02)
plt.show()
