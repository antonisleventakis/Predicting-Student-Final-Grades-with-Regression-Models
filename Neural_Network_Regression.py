import time
import warnings
from math import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import (MinMaxScaler)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
start = time.time()
# load datasets for two subjects, Math and Portuguese
mat = pd.read_csv(
    "Set this to your pathname for the file: student-mat.csv", sep=';')
por = pd.read_csv(
    "Set this to your pathname for the file : student-por.csv", sep=';')
end = time.time()
print("\nTime to load data:", end - start, "seconds")
print()
# preprocess the data
start = time.time()

mat['subject'] = 'Maths'
por['subject'] = 'Portuguese'
df = pd.concat([mat, por])

# Set the random seed for reproducibility
np.random.seed(123)

# 1. Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)


# 2. Removing duplicated rows and missing values
duplicated_rows = df.duplicated()


# Print the number of duplicated rows
print(f"Number of duplicated rows: {duplicated_rows.sum()}")

# Drop the duplicated rows
df = df.drop_duplicates()

# 3. Drop the missing values
df.dropna(inplace=True)
# Print the number of remaining rows
print(f"Number of remaining rows: {len(df)}")

# metonomazoume ta xarakrthristika gia na einai pio katanaohta
df.columns = ['school', 'sex', 'age', 'address', 'family_size', 'parents_status', 'mother_education', 'father_education',
              'mother_job', 'father_job', 'reason', 'guardian', 'commute_time', 'study_time', 'failures', 'school_support',
              'family_support', 'paid_classes', 'activities', 'nursery', 'desire_higher_edu', 'internet', 'romantic', 'family_quality',
              'free_time', 'go_out', 'weekday_alcohol_usage', 'weekend_alcohol_usage', 'health', 'absences', 'p1_score', 'p2_score', 'final_score', 'subject']

# Feature Engineering
# final_grade = weighted sum of p1, p2, final score
df["final_grade"] = (0.15*df["p1_score"]) + \
    (0.20*df["p2_score"]) + (0.65*df["final_score"])

# Student Group: 1,2,3,4 me vash ton teliko vathmo tous
# df['Student_Group'] = 0  # dhmiourgia neas sthlhs me times 'na'
# df.loc[(df.final_grade >= 0) & (df.final_grade < 10), 'Student_Group'] = 4
# df.loc[(df.final_grade >= 10) & (df.final_grade < 14), 'Student_Group'] = 3
# df.loc[(df.final_grade >= 14) & (df.final_grade < 17), 'Student_Group'] = 2
# df.loc[(df.final_grade >= 17) & (df.final_grade <= 20), 'Student_Group'] = 1
df['Student_Group'] = pd.cut(
    df.final_grade, bins=[-1, 10, 14, 17, 20], labels=[4, 3, 2, 1])


# 4. Kanonikopoihsh synexwn metavlitwn

cont_cols = ['age', 'mother_education', 'father_education', 'commute_time', 'study_time', 'failures', 'family_quality', 'free_time', 'go_out',
             'weekday_alcohol_usage', 'weekend_alcohol_usage', 'health', 'absences', 'p1_score', 'p2_score', 'final_score', 'Student_Group']
# for col in cont_cols:
#    df[col] = (df[col]-min(df[col]))/(max(df[col])-min(df[col]))

# normalize continuous variables
df[cont_cols] = MinMaxScaler().fit_transform(df[cont_cols])

# 5. 'One Hot Encoding' twn katyhgorikwn metavlitwn

ohe_cols = ['school', 'sex', 'address', 'family_size', 'parents_status', 'mother_job', 'father_job',
            'reason', 'guardian', 'school_support', 'family_support',
            'paid_classes', 'activities', 'nursery',
            'desire_higher_edu', 'internet', 'romantic', 'subject']

df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# 6. Afairesh twn  'Outliers'
print("Shape before removing outliers: ", df.shape)


def remove_outliers(df):
    for col in df.columns:
        if df[col].dtype != 'object':
            # Check the skewness of the column
            skewness = df[col].skew()
            if abs(skewness) > 1:
                # Use z-score method for outlier detection
                threshold = 3
                z_scores = (df[col] - df[col].mean()) / df[col].std()
                df.loc[abs(z_scores) > threshold, col] = np.nan
            else:
                # Use Tukey's method for outlier detection
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df.loc[(df[col] < lower_bound) | (
                    df[col] > upper_bound), col] = np.nan

    # Drop rows with missing values
    df.dropna(inplace=True)

    return df


df = remove_outliers(df)

print("Shape after removing outliers: ", df.shape)
print()

# 7. Feature Selection

print("Shape before feature selection: ", df.shape)


def select_features(df, num_features):
    X = df.drop('final_grade', axis=1)
    y = df['final_grade']

    # instantiate SelectKBest with f_regression as the score function
    selector = SelectKBest(f_regression, k=num_features)

    # fit selector to data
    selector.fit(X, y)

    # create a boolean mask to select the columns that were selected by the selector
    mask = selector.get_support()

    # get the column names for the selected columns
    selected_columns = X.columns[mask]

    # return a dataframe with the selected columns and the target column
    return df[selected_columns.append(pd.Index(['final_grade']))]


df = select_features(df, 23)

print("Shape after feature selection: ", df.shape)
print()

# 8. Anazhthsh "highly corrrelated" xarakthristikwn:

# Compute the correlation matrix
corr_matrix = df.corr().abs()

# Iterate over all the columns to find pairs with a correlation coefficient above the threshold
cols_to_drop = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > 0.90:
            colname_i = corr_matrix.columns[i]
            colname_j = corr_matrix.columns[j]
            cols_to_drop.add(colname_i)
            cols_to_drop.add(colname_j)

# Convert the set of columns to drop to a list
cols_to_drop = list(cols_to_drop)
print("\nColumns with 'high correlation' (> 92%): \n", cols_to_drop)
print()

# Storing the target variable before dropping
y = df['final_grade']

# Dropping highly correlated variables
df.drop(cols_to_drop, axis=1, inplace=True)
df.drop(['Student_Group'], axis=1, inplace=True)
print(df.shape)
# print(df.columns)

end = time.time()
print("\nTime to preprocess data:", end - start, "seconds")
print()
###############################################################################
# FF Neural Network for modelling
start = time.time()

x_train, x_test, y_train, y_test = train_test_split(
    df, y, test_size=0.3, random_state=20)


# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(14, activation='relu',
                          input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0025)

model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, validation_data=(
    x_test, y_test), batch_size=9, epochs=50)

end = time.time()

print("\nTime to model data using NN:", end - start, "seconds")
print()
# plot the training and validation loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# predict on test data
predictions = model.predict(x_test)
# diorthwnontas tis diastaseis gia kalh ektiposi
y_test = y_test.to_numpy()
# y_pred = y_pred.reshape(len(y_pred), 1)
y_test = np.squeeze(y_test)
# y_test.shape
predictions = np.squeeze(predictions)
# y_pred.shape

fdf = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(fdf)


mse_nn = mean_squared_error(y_test, predictions)
n = len(y_test)
y_mean = np.mean(y_test)
variance = sum((predictions - y_mean) ** 2) / (n - 1)
std = sqrt(variance)
print("Variance of the NN model:", variance)
print("Std of the NN model:", std)
print("MAE from NN: ", mean_absolute_error(y_test, predictions))
print("MSE from NN: ", mse_nn)
print("RMSE from NN: ", np.sqrt(mse_nn))
print("R2 Score from NN: ", r2_score(y_test, predictions))
print()

# Assuming y_test and predictions are numpy arrays
plt.scatter(y_test, predictions)
plt.plot([0, 20], [0, 20], 'r--')  # plot the y = x line in red dashed line
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.title('Actual Grades vs. Predicted Grades')

plt.show()

###############################################################################

# Linear Regression
start = time.time()

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)

end = time.time()
print("Time to model data using LR:", end - start, "seconds")
print()
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
variance_lr = sum((y_pred_lr - y_mean) ** 2) / (n - 1)
std_lr = sqrt(variance_lr)
print("Variance of the LR model:", variance_lr)
print("Std of the LR model:", std_lr)
print("MAE from LR: ", mae_lr)
print("MSE from LR: ", mse_lr)
print("RMSE from LR: ", np.sqrt(mse_lr))
print("R2 Score from LR: ", r2_score(y_test, y_pred_lr))
print()

###############################################################################

# Decision Tree Regression
dt_model = DecisionTreeRegressor()
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
variance_dt = sum((y_pred_dt - y_mean) ** 2) / (n - 1)
std_dt = sqrt(variance_dt)
print("Variance of DT model: ", variance_dt)
print("Std deviation of DT model: ", std_dt)
print("MAE from DT: ", mae_dt)
print("MSE from DT: ", mse_dt)
print("RMSE from DT: ", np.sqrt(mse_dt))
print("R2 Score from DT: ", r2_score(y_test, y_pred_dt))
print()

###############################################################################

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=20)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
variance_rf = sum((y_pred_rf - y_mean) ** 2) / (n - 1)
std_rf = sqrt(variance_rf)
print("Variance of RF model: ", variance_rf)
print("Std deviation of RF model: ", std_rf)
print("MAE from RF: ", mae_rf)
print("MSE from RF: ", mse_rf)
print("RMSE from RF: ", np.sqrt(mse_rf))
print("R2 Score from RF: ", r2_score(y_test, y_pred_rf))
print()

###############################################################################

# Support Vector Machine Regression

svm_model = SVR(kernel='linear')
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
mse_svm = mean_squared_error(y_test, y_pred_svm)
variance_svm = sum((y_pred_svm - y_mean) ** 2) / (n - 1)
std_svm = sqrt(variance_svm)
print("Variance of SVM model: ", variance_svm)
print("Std deviation of SVM model: ", std_svm)
print("MAE from SVM: ", mae_svm)
print("MSE from SVM: ", mse_svm)
print("RMSE from SVM: ", np.sqrt(mse_svm))
print("R2 Score from SVM: ", r2_score(y_test, y_pred_svm))
print()
###############################################################################

# K Nearest Neighbor Regression
knn_model = KNeighborsRegressor(n_neighbors=6)
knn_model.fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
variance_knn = sum((y_pred_knn - y_mean) ** 2) / (n - 1)
std_knn = sqrt(variance_knn)
print("Variance of KNN model: ", variance_knn)
print("Std deviation of KNN model: ", std_knn)
print("MAE from KNN: ", mae_knn)
print("MSE from KNN: ", mse_knn)
print("RMSE from KNN: ", np.sqrt(mse_knn))
print("R2 Score from KNN: ", r2_score(y_test, y_pred_knn))
print()

# plot linear regression
plt.plot(y_test, y_pred_lr, 'yo', label='Linear Regression')

# plot decision tree regression
plt.plot(y_test, y_pred_dt, 'bo', label='Decision Tree Regression')

# plot random forest regression
plt.plot(y_test, y_pred_rf, 'go', label='Random Forest Regression')

# plot support vector machine regression
plt.plot(y_test, y_pred_svm, 'ro', label='Support Vector Machine Regression')

# plot k nearest neighbor regression
plt.plot(y_test, y_pred_knn, 'co', label='K Nearest Neighbor Regression')

# plot neural network regression
plt.plot(y_test, predictions, 'mo', label='Neural Network Regression')

plt.plot([0, 20], [0, 20], 'k--', label='Ideal Line')

# set labels and title
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.title('Actual vs Predicted Grades for Different Regression Models')

# set legend
plt.legend()

# show plot
plt.show()

# Feature Ranking
print("Feature Ranking: \n")
feature_list = list(df.columns)
feature_imp = pd.Series(rf_model.feature_importances_,
                        index=feature_list).sort_values(ascending=False)
print(feature_imp)
