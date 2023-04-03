from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from scipy.stats import skew
from scipy.stats import zscore, iqr
from sklearn.preprocessing import RobustScaler
from scipy.stats import zscore
from sklearn.feature_selection import SelectKBest, f_regression
from keras import regularizers
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import math
from sklearn import neighbors
from math import *
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
import scipy.stats as stats
from tqdm import tqdm
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from locale import normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# load datasets for two subjects, Math and Portuguese
mat = pd.read_csv(
    "pathname of student-mat.csv file", sep=';')
por = pd.read_csv(
    "pathname of student-por.csv file", sep=';')

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

# final_grade = weighted sum of p1, p2, final score
df["final_grade"] = (0.15*df["p1_score"]) + \
    (0.20*df["p2_score"]) + (0.65*df["final_score"])

# Student Group based on final grades
df['Student_Group'] = pd.cut(
    df.final_grade, bins=[-1, 10, 14, 17, 20], labels=[4, 3, 2, 1])


# 4. Normalization of continuous variables

cont_cols = ['age', 'mother_education', 'father_education', 'commute_time', 'study_time', 'failures', 'family_quality', 'free_time', 'go_out',
             'weekday_alcohol_usage', 'weekend_alcohol_usage', 'health', 'absences', 'p1_score', 'p2_score', 'final_score', 'Student_Group']
# for col in cont_cols:
#    df[col] = (df[col]-min(df[col]))/(max(df[col])-min(df[col]))
df[cont_cols] = MinMaxScaler().fit_transform(df[cont_cols])



# 5. 'One Hot Encoding' of categorical variables

ohe_cols = ['school', 'sex', 'address', 'family_size', 'parents_status', 'mother_job', 'father_job',
            'reason', 'guardian', 'school_support', 'family_support',
            'paid_classes', 'activities', 'nursery',
            'desire_higher_edu', 'internet', 'romantic', 'subject']

df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# 6. Outliers removal
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

# """


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

# """
print("Shape after feature selection: ", df.shape)
print()

# 8. Removal of "highly corrrelated" features

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
print(df.columns)

###############################################################################


# Modelling the data using Semi-Supervised Learning

x_train, x_test, y_train, y_test = train_test_split(
    df, y, test_size=0.65, random_state=42)

Lx = x_train
Ly = y_train
Ux = x_test
Uy = y_test
T = 0.1
MaxIter = 15
m = [T * len(Lx)]
m = [round(m) for m in m]
print("m: ", m[0])

model_1 = neighbors.KNeighborsRegressor(
    n_neighbors=2, metric='minkowski', weights='distance')
model_2 = neighbors.KNeighborsRegressor(
    n_neighbors=3, metric='minkowski', weights='distance')
model_3 = neighbors.KNeighborsRegressor(
    n_neighbors=4, metric='minkowski', weights='distance')

h1 = model_1.fit(Lx, Ly)
h2 = model_2.fit(Lx, Ly)
h3 = model_3.fit(Lx, Ly)

h1_y = h1.predict(Ux)
h2_y = h2.predict(Ux)
h3_y = h3.predict(Ux)

print("\nL shape before learning: \n", Lx.shape, Ly.shape)
print("\nU shape before learning: \n", Ux.shape, Uy.shape)

# Defining S
Sx = pd.DataFrame(columns=Lx.columns)
dropped_rows = []

for j in range(MaxIter):
    # Dx: array with the differences of the estimations of the 3 models
    Dx = np.random.uniform(low=0.0, high=1.0, size=(len(Ux)))
    for i in range(len(Ux)):
        Dx[i] = max(h1_y[i], h2_y[i], h3_y[i]) - min(h1_y[i], h2_y[i], h3_y[i])

    # argpartition: Every value below the m[0] is sorted to be less or equal than m[0]
    idx = np.argpartition(Dx, m[0])

    # Storing in S the "m" more "confident" rows
    for k in range(0, m[0]):
        Sx.loc[k] = Ux.iloc[idx[k]]
        dropped_rows.append(idx[k])

    # Each model predicts the values for the "confident" rows
    S1y = h1.predict(Sx)
    S2y = h2.predict(Sx)
    S3y = h3.predict(Sx)

    # We store in S_y the average value for each row
    S_y = np.random.uniform(low=0.0, high=1.0, size=m[0])
    for l in range(m[0]):
        S_y[l] = (S1y[l]+S2y[l]+S3y[l])/3

    # Sx rows are removed from Ux
    print("\nUx shape before Sx removal: \n", Ux.shape)

    Ux_new = pd.concat([Ux, Sx])
    Ux_new = Ux_new.drop_duplicates(keep=False)
    Ux = Ux_new
    print("\nUx shape after S removal: \n", Ux.shape)

    print("\nUy shape before Sy removal: \n", Uy.shape)
    
    # create boolean mask indicating which rows were dropped
    mask = np.zeros(Uy.shape[0], dtype=bool)
    mask[dropped_rows] = True
    # drop corresponding rows from Uy
    Uy = Uy[~mask]
    print("\nUy shape after Sy removal: \n", Uy.shape)
    dropped_rows = []
    # S is added to L
    S_y = pd.DataFrame(S_y)

    print("\nL shape before S addition: \n", Lx.shape, Ly.shape)
    Lx = pd.concat([Lx, Sx])
    Ly = pd.concat([Ly, S_y])
    print("\nL shape after S addition: \n", Lx.shape, Ly.shape)
    print()

    # Each model is retrained on the new L
    h1 = model_1.fit(Lx, Ly)
    h2 = model_2.fit(Lx, Ly)
    h3 = model_3.fit(Lx, Ly)

    h1_y = h1.predict(Ux)
    h2_y = h2.predict(Ux)
    h3_y = h3.predict(Ux)

print("Final L shape after learning: \n", Lx.shape, Ly.shape)
print("Final U shape after learning: \n", Ux.shape, Uy.shape)

print()

# After Semi-Supervised Learning we are using a Random Forrest Regressor on the final L set

# Split the data into training and test sets
Lx_train, Lx_test, Ly_train, Ly_test = train_test_split(
    Lx, Ly, test_size=0.20, random_state=42)  # stratify na dw


# Split the training data into training and validation sets
Lx_train, Lx_val, Ly_train, Ly_val = train_test_split(
    Lx_train, Ly_train, test_size=0.15, random_state=42)

# Print the sizes of the resulting sets
print("Training set size:", len(Lx_train), len(Ly_train))
print("Testing set size:", len(Lx_test), len(Ly_test))
print("Validation set size:", len(Lx_val), len(Ly_val))
print()

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Defining the model
RFR = RandomForestRegressor(random_state=42)
# Fitting the model to the training set
RFR.fit(Lx_train, Ly_train)

# Predicts and compares the validation set with the actual values
Ly_val_pred = RFR.predict(Lx_val)
print("MAE from SSR in validation data: ",
      mean_absolute_error(Ly_val, Ly_val_pred))
print("MSE from SSR in validation data: ",
      mean_squared_error(Ly_val, Ly_val_pred))
print("R2 Score from SSR in validation data: ", r2_score(Ly_val, Ly_val_pred))
print()
# Predicts the remaining labels from U and compares the with actual values
Uy_pred = RFR.predict(Ux)
mse = mean_squared_error(Uy, Uy_pred)
print("MAE from SSR in U: ", mean_absolute_error(Uy, Uy_pred))
print("MSE from SSR in U: ", mse)
print("RMSE from SSR in U: ", np.sqrt(mse))
print("R2 Score from SSR in U: ", r2_score(Uy, Uy_pred))
print()

# reshaping for plotting
Uy = Uy.to_numpy()
Uy_pred = Uy_pred.reshape(len(Uy_pred), 1)
Uy = np.squeeze(Uy)
Uy.shape
Uy_pred = np.squeeze(Uy_pred)

fdf = pd.DataFrame({'Actual': Uy, 'Predicted': Uy_pred})
print(fdf)

plt.plot(Uy, Uy_pred, 'y*')
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.show()

###############################################################################
# """
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(Lx_train, Ly_train)
y_pred_lr = lr_model.predict(Ux)
print("MAE from LR in U: ", mean_absolute_error(Uy, y_pred_lr))
print("MSE from LR in U: ", mean_squared_error(Uy, y_pred_lr))
print("R2 Score from LR in U: ", r2_score(Uy, y_pred_lr))
print()
###############################################################################

# Decision Tree Regression
dt_model = DecisionTreeRegressor()
dt_model.fit(Lx_train, Ly_train)
y_pred_dt = dt_model.predict(Ux)
print("MAE from DT in U: ", mean_absolute_error(Uy, y_pred_dt))
print("MSE from DT in U: ", mean_squared_error(Uy, y_pred_dt))
print("R2 Score from DT in U: ", r2_score(Uy, y_pred_dt))
print()
###############################################################################
# Feature Ranking
feature_list = list(df.columns)
feature_imp = pd.Series(RFR.feature_importances_,
                        index=feature_list).sort_values(ascending=False)
print(feature_imp)
# """
