#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 18:47:00 2025

@author: genti
"""

#import libraries
import sys
import scipy
import matplotlib
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import pywt

# scikit-learn models and scaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Models from TensorFlow Keras for the LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint

import os, matplotlib
matplotlib.use('Agg')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(SCRIPT_DIR, 'Report/Assignment/flu/charts/')
os.makedirs(CHART_DIR, exist_ok=True)

sns.set_style("whitegrid")

color_pal = sns.color_palette('coolwarm')


# Start the timer
start_time = time.perf_counter()

# Set display options
pd.set_option('display.max_columns', None)    # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping
pd.set_option('display.max_colwidth', None)    # Show full column contents

#Check All Library Versions
print('\nVersion Numbers\n')
print('Python       : {}'.format(sys.version))
print('scipy        : {}'.format(scipy.__version__))
print('numpy        : {}'.format(np.__version__))
print('matplotlib   : {}'.format(matplotlib.__version__))
print('pandas       : {}'.format(pd.__version__))
print('sklearn      : {}'.format(sklearn.__version__))

# Load Influenza Data for North America
file_path = "DataExport_AMR_NA_020425.xlsx"
df = pd.read_excel(file_path, sheet_name="DataExport")

# Print the shape, head, tail and a saple of the dataset
print("\n")
print("Shape of Data in (Rows, Columns) is " + str(df.shape))

print("\n")
print("\nThe First 10 Rows of the Dataset: \n", df.head(10))
print("\nThe Last 10 Rows of the Dataset: \n", df.tail(10))
print("\nA Sample of 10 Rows of the Dataset: \n", df.sample(10))

print("\n")
print("\nThe Entire Dataset: \n", df)

print("\n")
print("\nA Slice of Rows 10 to 21 of the Dataset: \n", df[10:21])

# Statistical Summary
print("\n")
print(df.describe())

# How Many Columns Have Missing Values
print("\n\nNull Value Count by Column : \n", df.isnull)

# Convert date columns to datetime format
df['ISO_SDATE'] = pd.to_datetime(df['ISO_SDATE'])

# Function to determine season based on month
def determine_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Create new "Season" column
df['Season'] = df['ISO_SDATE'].dt.month.apply(determine_season)

# Selecting relevant columns for prediction
selected_columns = ['ISO_YEAR', 'ISO_WEEK', 'ISO_SDATE', 'INF_ALL', 'INF_A', 'INF_B', 'ILI_ACTIVITY', 'AH1', 'AH3', 'Season', 'SPEC_PROCESSED_NB']
df = df[selected_columns]

# Data Cleaning: Handling missing values
df.ffill(inplace=True)  # Forward-fill missing values


# Creating new attributes for new time-based features
df['Month'] = df['ISO_SDATE'].dt.month
df['Day_of_Week'] = df['ISO_SDATE'].dt.dayofweek
df['Total_Influenza_Cases'] = df['INF_ALL']

# Creating Lag Features for trend analysis
df['INF_ALL_LAG_1'] = df['INF_ALL'].shift(1)
df['INF_ALL_LAG_2'] = df['INF_ALL'].shift(2)
df['INF_ALL_LAG_4'] = df['INF_ALL'].shift(4)

# Removing rows where lag features are NaN
df.dropna(inplace=True)

# For skewed or non-normal data the IQR method is more preferred
# Detecting Outliers using IQR (Interquartile Range) method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Calculating all the outliers for INF_ALL, INF_A, and INF_B
outliers_inf_all = detect_outliers_iqr(df, 'INF_ALL')
outliers_inf_a = detect_outliers_iqr(df, 'INF_A')
outliers_inf_b = detect_outliers_iqr(df, 'INF_B')
print(f"Outliers detected in INF_ALL: {len(outliers_inf_all)}")
print(f"Outliers detected in INF_A: {len(outliers_inf_a)}")
print(f"Outliers detected in INF_B: {len(outliers_inf_b)}")

# Function to plot a bar chart
def plot_bar_chart(dataframe, label):
    # Plotting the distribution of data points per season
    plt.figure(figsize=(12, 6))
    dataframe.plot(kind='bar', rot=0)
    plt.title(f'Distribution of Influenza Cases by {label}')
    plt.xlabel(f'{label}')
    plt.ylabel('Number of Influenza Cases')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()

# Start creating different datasets for checking the validity of the outliers 
# calculate above.
# Count occurrences for each season
# Group by Season and Calculate the sum of INF_ALL for each season
seasonal_inf_all_sum = df.groupby('Season')['INF_ALL'].sum()

# Plotting the distribution of data points per season
plot_bar_chart(seasonal_inf_all_sum, 'Season')
plt.savefig(f'{CHART_DIR}01_seasonal_cases.png', dpi=150, bbox_inches='tight'); plt.show(); plt.close()

# Group by "Season" and compute mean for numeric columns
season_mean = df.groupby('Season')['INF_ALL'].mean(numeric_only=True)

# Plotting the distribution of data points per season
plot_bar_chart(season_mean, 'Mean')
plt.savefig(f'{CHART_DIR}02_mean_seasonal_cases.png', dpi=150, bbox_inches='tight'); plt.show(); plt.close()

#create anew dataframe without outliers
df_without_outliers = df.merge(outliers_inf_all, how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])

# Count occurrences for each season
# Group by Season and Calculate the sum of INF_ALL for each season
seasonal_inf_all_sum_without_outliers = df_without_outliers.groupby('Season')['INF_ALL'].sum()

# Plotting the distribution of data points per season
plot_bar_chart(seasonal_inf_all_sum_without_outliers, 'Total Without Outliers')
plt.savefig(f'{CHART_DIR}03_seasonal_no_outliers.png', dpi=150, bbox_inches='tight'); plt.show(); plt.close()

#Plotting the winter season group by year including the outliers
filtered_df = df[df['Season'] == 'Winter']
filtered_df_winter_without_outliers = df_without_outliers[df_without_outliers['Season'] == 'Winter']

# Calculate mean Influenza cases across all seasons grouped by year
mean_inf_all_by_year_all_seasons = df.groupby('ISO_YEAR', as_index=False)['INF_ALL'].mean().reset_index()
sum_winter_inf_all_by_year = filtered_df.groupby('ISO_YEAR', as_index=False)['INF_ALL'].sum().reset_index()
mean_inf_all_without_outliers_by_year_all_seasons = df_without_outliers.groupby('ISO_YEAR', as_index=False)['INF_ALL'].mean().reset_index()
sum_inf_all_without_outliers_by_year_winter = filtered_df_winter_without_outliers.groupby('ISO_YEAR', as_index=False)['INF_ALL'].sum().reset_index()

# Plotting the lineplots
plt.figure(figsize=(12, 6))

# Line plot for Winter Influenza cases
sns.lineplot(data=sum_winter_inf_all_by_year, x='ISO_YEAR', y='INF_ALL', label='Winter Total Influenza Cases')

# Line plot for mean Influenza cases by year across all seasons
sns.lineplot(data=mean_inf_all_by_year_all_seasons, x='ISO_YEAR', y='INF_ALL', label='Mean Influenza Cases (All Seasons)', linestyle='--')

# Line plot for mean Influenza cases by year across all seasons
sns.lineplot(data=mean_inf_all_without_outliers_by_year_all_seasons, x='ISO_YEAR', y='INF_ALL', label='Mean Influenza Cases Without Outliers (All Seasons)', linestyle='-.')

# Line plot for mean Influenza cases by year across all seasons
sns.lineplot(data=sum_inf_all_without_outliers_by_year_winter, x='ISO_YEAR', y='INF_ALL', label='Winter Total Influenza Cases Without Outliers', linestyle=':')

# Labels and title
plt.xlabel('Year')
plt.ylabel('Influenza Cases')
plt.title('Influenza Cases for Winter vs Mean All Seasons by Year With or Without Outliers')
plt.legend()

plt.savefig(f'{CHART_DIR}04_winter_vs_mean.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
df.to_csv("processed_influenza_data.csv", index=False)

# Data Scaling for model readiness, to normalize the columns individually
scaler = StandardScaler()
df[['INF_ALL', 'INF_A', 'INF_B', 'ILI_ACTIVITY']] = scaler.fit_transform(df[['INF_ALL', 'INF_A', 'INF_B', 'ILI_ACTIVITY']])

# Visualizing Data Trends
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='ISO_SDATE', y='INF_ALL', label='Total Influenza Cases')
sns.lineplot(data=df, x='ISO_SDATE', y='INF_A', label='Influenza A Cases')
sns.lineplot(data=df, x='ISO_SDATE', y='INF_B', label='Influenza B Cases')
plt.legend()
plt.title('Influenza Cases Trend in North America')
plt.savefig(f'{CHART_DIR}05_influenza_trend.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()


# =============================================================================
# Candidate Feature Sets
# =============================================================================
# Define several candidate feature sets to explore which combination of features
# best predicts the total number of influenza cases.
feature_sets = {
    'All_Features': ['ISO_YEAR', 'ISO_WEEK', 'Month', 'INF_ALL_LAG_1', 
                     'INF_ALL_LAG_2', 'INF_ALL_LAG_4', 'INF_A', 'INF_B', 
                     'ILI_ACTIVITY', 'INF_ALL', 'SPEC_PROCESSED_NB', 
                     'Total_Influenza_Cases'],
    'All_Features_Without_Target': ['ISO_YEAR', 'ISO_WEEK', 'Month', 
                    'INF_ALL_LAG_1', 'INF_ALL_LAG_2', 'INF_ALL_LAG_4', 
                    'INF_A', 'INF_B', 'ILI_ACTIVITY', 'INF_ALL', 
                    'SPEC_PROCESSED_NB'],
    'Season_Features': ['Season_numeric', 'INF_ALL', 'SPEC_PROCESSED_NB', 
                        'Total_Influenza_Cases'],
    'Season_Features_Without_Target': ['Season_numeric', 'INF_ALL', 'SPEC_PROCESSED_NB']
}

# Define the target column and the look-back period (number of past time steps).
target = 'Total_Influenza_Cases'
look_back = 60

# =============================================================================
# Helper Functions
# =============================================================================
def mae(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE) between actual and predicted values.
    
    Parameters:
      y_true: Actual target values.
      y_pred: Predicted target values.
      
    Returns:
      MAE: The mean absolute error as a float.
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def mse(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) between actual and predicted values.
    
    Parameters:
      y_true: Actual target values.
      y_pred: Predicted target values.
      
    Returns:
      MSE: The mean squared error as a float.
    """
    return np.mean((y_true - y_pred) ** 2)
    

# =============================================================================
# Data Loading
# =============================================================================
# Load the CSV file that contains the processed influenza data.
data = pd.read_csv('processed_influenza_data.csv', parse_dates=["ISO_SDATE"], index_col="ISO_SDATE")
# Convert a categorical Season column to numeric codes
data['Season_numeric'] = data['Season'].astype('category').cat.codes

# =============================================================================
# Pearson Correlation Heatmaps for Each Candidate Feature Set
# =============================================================================
print("\nPearson Correlation Heatmaps for Each Candidate Feature Set:")
colormap = sns.diverging_palette(h_neg=45, h_pos=225, s=81, l=68, sep=30, center='light', as_cmap=True)
_fig_heatmap_num = 6
# Loop over each candidate feature set defined earlier.
for name, feature_list in feature_sets.items():
    # Create a subset of the data including the current feature set and the target.
    subset = data[feature_list + [target]]
    # Calculate the Pearson correlation matrix.
    corr_matrix = subset.corr(method='pearson')
    
    # Create a heatmap using seaborn.
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap=colormap, fmt=".2f")
    plt.title(f"Pearson Correlation Heatmap for Feature Set: {name}")
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}{_fig_heatmap_num:02d}_corr_heatmap_{name}.png', dpi=150, bbox_inches='tight')
    _fig_heatmap_num += 1
    plt.show()
    plt.close()
    

# -------------------------------
# Date-based Splitting Logic
# -------------------------------
# Define the splitting durations and cutoff years
test_val_duration = 2
test_start = 2023
test_end = test_start + test_val_duration
val_start = test_start - test_val_duration
val_end = test_start
train_start = 1997
train_end = val_start

# setting the year 2018 since there are no significant pandemics 
wavelet_start_year = 2018
wavelet_end_date = '2018-07-01'

# Wavelet
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    x = np.copy(x)  # ensure writable
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    
    wave = pywt.waverec(coeff, wavelet, mode='per')
    
    if len(x) == len(wave):
        return wave
    else:
        return wave[:-1]
    
data['Wavelet'] = denoise_signal(data[target])

df_slice = data.loc[data.index.year == wavelet_start_year]
df_slice = df_slice.loc[df_slice.index < wavelet_end_date]

fig, ax = plt.subplots(figsize=(14,3) )
sns.lineplot(data=df_slice, 
             x=df_slice.index,
             y=target,
             linewidth=3,
             color=color_pal[0],
             ax=ax)

sns.lineplot(data=df_slice, 
             x=df_slice.index,
             y='Wavelet',
             linewidth=1.5,
             color='black',
             ax=ax)

ax.legend([target, 'Wavelet'], loc='lower left')
plt.savefig(f'{CHART_DIR}10_wavelet_denoising.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Comparison of DWT to a simple moving average:
data['MA5'] = data[target].rolling(5).mean()

df_slice = data.loc[data.index.year == wavelet_start_year]
df_slice = df_slice.loc[df_slice.index < wavelet_end_date]

fig, ax = plt.subplots(figsize=(14,3) )
sns.lineplot(data=df_slice, 
             x=df_slice.index,
             y=target,
             linewidth=3,
             color=color_pal[0],
             ax=ax)

sns.lineplot(data=df_slice, 
             x=df_slice.index,
             y='MA5',
             linewidth=3,
             color='gray',
             ax=ax)

sns.lineplot(data=df_slice, 
             x=df_slice.index,
             y='Wavelet',
             linewidth=1.5,
             color='black',
             ax=ax)


ax.legend([target,'MA', 'Wavelet'], loc='lower left')
plt.savefig(f'{CHART_DIR}11_ma_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Comparison of DWT to an exponentially weighted moving average:
data['EMA5'] = data[target].ewm(span=5).mean()

df_slice = data.loc[data.index.year == wavelet_start_year]
df_slice = df_slice.loc[df_slice.index < wavelet_end_date]

fig, ax = plt.subplots(figsize=(14,3) )
sns.lineplot(data=df_slice, 
             x=df_slice.index,
             y=target,
             linewidth=3,
             color=color_pal[0],
             ax=ax)

sns.lineplot(data=df_slice, 
             x=df_slice.index,
             y='EMA5',
             linewidth=3,
             color='gray',
             ax=ax)

sns.lineplot(data=df_slice, 
             x=df_slice.index,
             y='Wavelet',
             linewidth=1.5,
             color='black',
             ax=ax)


ax.legend([target,'EWA', 'Wavelet'], loc='lower left')
plt.savefig(f'{CHART_DIR}12_ewa_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()


# Initialize dictionaries to store MAE and MSE results
results = {'Feature_Set': [], 'Model': [], 'MAE': []}
results_mse = {'Feature_Set': [], 'Model': [], 'MSE': []}


# Define the two feature sets to loop over
feature_set_keys = ['All_Features', 'Season_Features']
_model_fig_num = 13

# Loop over each feature set
for fs_key in feature_set_keys:
    print("\n==============================")
    print(f"Processing feature set: {fs_key}")
    print("==============================\n")
    
    # Use the selected feature set to build a full dataframe
    full_df = data[feature_sets[fs_key]]
    
    # -------------------------------
    # Date-based Splitting Logic
    # -------------------------------
    train_df = full_df.loc[(full_df.index.year >= train_start) & (full_df.index.year < train_end)]
    val_df   = full_df.loc[(full_df.index.year >= val_start)   & (full_df.index.year < val_end)]
    test_df  = full_df.loc[(full_df.index.year >= test_start)  & (full_df.index.year < test_end)]
    
    # For plotting, define a color palette (adjust colors as desired)
    color_pal = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Plot the dataset splits using the 'Close' column
    f, ax = plt.subplots(figsize=(14,6.5))
    train_df['INF_ALL'].plot(ax=ax, label='Training Set', color='gray')
    val_df['INF_ALL'].plot(ax=ax, label='Validation Set', color=color_pal[0])
    test_df['INF_ALL'].plot(ax=ax, label='Testing Set', color=color_pal[2])
    plt.axvline(x=str(val_start), color='gray', linestyle='--', linewidth=1.3)
    plt.axvline(x=str(test_start), color='gray', linestyle='--', linewidth=1.3)
    ax.legend(['Training Set', 'Validation Set', 'Testing Set'])
    plt.title(f"Dataset Split by Date ({fs_key})")
    plt.savefig(f'{CHART_DIR}{_model_fig_num:02d}_dataset_split_{fs_key}.png', dpi=150, bbox_inches='tight')
    _model_fig_num += 1
    plt.show()
    plt.close()
    
    # -------------------------------
    # Linear Regression Model
    # -------------------------------
    # Prepare training, validation, and testing sets
    X_train = train_df.copy()
    y_train = X_train.pop(target)
    
    X_val = val_df.copy()
    y_val = X_val.pop(target)
    
    X_test = test_df.copy()
    y_test = X_test.pop(target)
    
    # Train linear regression
    model_linreg = LinearRegression()
    model_linreg.fit(X_train, y_train)
    
    pred_linreg = model_linreg.predict(X_val)
    error_linreg_mae = mae(y_val, pred_linreg)
    error_linreg_mse = mse(y_val, pred_linreg)
    print(f"Linear Regression ({fs_key}) MAE: {error_linreg_mae}")
    print(f"Linear Regression ({fs_key}) MSE: {error_linreg_mse}")
    
    # Store linear regression results
    results['Feature_Set'].append(fs_key)
    results['Model'].append('Linear Regression')
    results['MAE'].append(error_linreg_mae)
    results_mse['Feature_Set'].append(fs_key)
    results_mse['Model'].append('Linear Regression')
    results_mse['MSE'].append(error_linreg_mse)
    
    # Print number of predictions made by Linear Regression
    print(f"Linear Regression ({fs_key}) predicted {len(pred_linreg)} cases.\n")
    
    
    # Plot linear regression predictions
    def plot_predictions(model, X_val, y_val, start_date=None, end_date=None):
        # If dates are not provided, use the full validation set.
        if start_date is None or end_date is None:
            true_val = y_val
            pred_val = model.predict(X_val)
            plot_dates = y_val.index
        else:
            true_val = y_val.loc[(y_val.index >= start_date) & (y_val.index <= end_date)]
            pred_val = model.predict(X_val.loc[(X_val.index >= start_date) & (X_val.index <= end_date)])
            plot_dates = y_val.loc[(y_val.index >= start_date) & (y_val.index <= end_date)].index

        fig, ax = plt.subplots(figsize=(14,6))
        sns.lineplot(x=plot_dates, y=true_val, color='#bbbbbb', ax=ax)
        sns.lineplot(x=plot_dates, y=pred_val, color=color_pal[0], ax=ax)
        ax.set(xlabel='Date', ylabel='Total Influenza Cases', title=f'Linear Regression: True vs Predicted Values ({fs_key})')
        ax.legend(['True Values', 'Predicted Values'])
        plt.savefig(f'{CHART_DIR}{_model_fig_num:02d}_lr_predictions_{fs_key}.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    
    plot_predictions(model_linreg, X_val, y_val)
    _model_fig_num += 1
    plt.close()
    
    # -------------------------------
    # Models on % Change Data (Decision Tree & NN)
    # -------------------------------
    # Calculate percentage change for specific columns and the target
    for df in [X_train, X_val, X_test]:
        df['Open_pct_change'] = df['SPEC_PROCESSED_NB'].pct_change()
        df['Open_pct_change'].iloc[0] = 0 
        df['Close_pct_change'] = df['INF_ALL'].pct_change()
        df['Close_pct_change'].iloc[0] = 0 
    
    y_train_pct = y_train.pct_change()
    y_train_pct.iloc[0] = 0
    y_val_pct = y_val.pct_change()
    y_val_pct.iloc[0] = 0
    y_test_pct = y_test.pct_change()
    y_test_pct.iloc[0] = 0
    
    # For scaling features for the decision tree and NN models,
    # choose the appropriate "without target" feature set.
    if fs_key == 'All_Features':
         features_without_target = feature_sets['All_Features_Without_Target']
    else:
         features_without_target = feature_sets['Season_Features_Without_Target']
    
    scaler = StandardScaler()
    scaler.fit(X_train[features_without_target])
    
    X_train_scaled = scaler.transform(X_train[features_without_target])
    X_val_scaled = scaler.transform(X_val[features_without_target])
    X_test_scaled = scaler.transform(X_test[features_without_target])
    
    # -------------------------------
    # Decision Tree Model
    # -------------------------------
    model_dtree = DecisionTreeRegressor()
    model_dtree.fit(X_train_scaled, y_train_pct)
    pred_dtree = model_dtree.predict(X_val_scaled)
    error_dtree = mae(y_val_pct, pred_dtree)
    error_dtree_mse = mse(y_val_pct, pred_dtree)
    print(f"Decision Tree ({fs_key}) MAE (% change): {error_dtree}")
    print(f"Decision Tree ({fs_key}) MSE (% change): {error_dtree_mse}")
    
    results['Feature_Set'].append(fs_key)
    results['Model'].append('Decision Tree')
    results['MAE'].append(error_dtree)
    results_mse['Feature_Set'].append(fs_key)
    results_mse['Model'].append('Decision Tree')
    results_mse['MSE'].append(error_dtree_mse)
    
    # Print number of predictions made by Decision Tree
    print(f"Decision Tree ({fs_key}) predicted {len(pred_dtree)} cases.\n")
    
    # Plot Decision Tree predictions
    plt.figure(figsize=(14, 6))
    plt.plot(y_val_pct.index, y_val_pct, label='True % Change', color='blue', linewidth=2)
    plt.plot(y_val_pct.index, pred_dtree, label='Predicted % Change', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('% Change in Total Influenza Cases')
    plt.title(f'Decision Tree Regressor: True vs Predicted % Change (Validation Set) - {fs_key}')
    plt.legend()
    plt.savefig(f'{CHART_DIR}{_model_fig_num:02d}_dt_predictions_{fs_key}.png', dpi=150, bbox_inches='tight')
    _model_fig_num += 1
    plt.show()
    plt.close()
    
    # -------------------------------
    # Neural Network Regressor
    # -------------------------------
    model_mlp = Sequential()
    model_mlp.add(Flatten(input_shape=(1, X_train_scaled.shape[-1])))
    model_mlp.add(BatchNormalization())
    model_mlp.add(Dense(64, activation='elu'))
    model_mlp.add(BatchNormalization())
    model_mlp.add(Dropout(0.2))
    model_mlp.add(Dense(32, activation='elu'))
    model_mlp.add(BatchNormalization())
    model_mlp.add(Dropout(0.2))
    model_mlp.add(Dense(16, activation='elu'))
    model_mlp.add(BatchNormalization())
    model_mlp.add(Dropout(0.2))
    model_mlp.add(Dense(1))
    model_mlp.compile(optimizer=RMSprop(learning_rate=0.001), loss='mae')
    
    cb_early_stopping = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=10)
    # Use a different checkpoint file name per feature set
    cb_checkpoint = ModelCheckpoint(f'best_model_{fs_key}.h5', monitor='val_loss', mode='min', save_best_only=True)
    
    # Reshape data for the neural network (expects 3D input)
    X_train_nn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_nn = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
    
    history = model_mlp.fit(X_train_nn, y_train_pct,
                            batch_size=5,
                            epochs=120,
                            validation_data=(X_val_nn, y_val_pct),
                            callbacks=[cb_early_stopping, cb_checkpoint])
    
    # Custom MAE function for model loading
    def mae_custom(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    from keras.models import load_model
    model_mlp = load_model(f'best_model_{fs_key}.h5', custom_objects={'mae': mae_custom})
    
    pred_mlp = model_mlp.predict(X_val_nn)
    # Flatten the predictions list
    pred_mlp = [item for sublist in pred_mlp for item in sublist]
    error_mlp = mae(y_val_pct, pred_mlp)
    error_mlp_mse = mse(y_val_pct, pred_mlp)
    print(f"Neural Network ({fs_key}) MAE (% change): {error_mlp}")
    print(f"Neural Network ({fs_key}) MSE (% change): {error_mlp_mse}")
    
    results['Feature_Set'].append(fs_key)
    results['Model'].append('Neural Network')
    results['MAE'].append(error_mlp)
    results_mse['Feature_Set'].append(fs_key)
    results_mse['Model'].append('Neural Network')
    results_mse['MSE'].append(error_mlp_mse)
    
    # Print number of predictions made by Neural Network
    print(f"Neural Network ({fs_key}) predicted {len(pred_mlp)} cases.\n")
    
    # Plot Neural Network predictions
    plt.figure(figsize=(14, 6))
    plt.plot(y_val_pct.index, y_val_pct, label='True % Change', color='blue', linewidth=2)
    plt.plot(y_val_pct.index, pred_mlp, label='Predicted % Change', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('% Change in Total Influenza Cases')
    plt.title(f'Neural Network Regressor: True vs Predicted % Change (Validation Set) - {fs_key}')
    plt.legend()
    plt.savefig(f'{CHART_DIR}{_model_fig_num:02d}_nn_predictions_{fs_key}.png', dpi=150, bbox_inches='tight')
    _model_fig_num += 1
    plt.show()
    plt.close()

# =============================================================================
# Bar Plot Comparing MAE for All Models
# =============================================================================
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', hue='Feature_Set', data=results_df)
plt.title('MAE Comparison for All Models by Feature Set')
plt.ylabel('MAE')
plt.xlabel('Model')
plt.savefig(f'{CHART_DIR}21_mae_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# =============================================================================
# Bar Plot Comparing MSE for All Models
# =============================================================================
results_mse_df = pd.DataFrame(results_mse)
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', hue='Feature_Set', data=results_mse_df)
plt.title('MSE Comparison for All Models by Feature Set')
plt.ylabel('MSE')
plt.xlabel('Model')
plt.savefig(f'{CHART_DIR}22_mse_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Combine the MAE and MSE results into one DataFrame for comparison
comparison_df = pd.merge(results_df, results_mse_df, on=['Feature_Set', 'Model'])
print("\nComparison of MAE and MSE for All Models by Feature Set:")
print(comparison_df.to_string(index=False))

CSV_PATH = os.path.join(SCRIPT_DIR, 'KPM/model_performance_metrics.csv')
comparison_df.to_csv(CSV_PATH, index=False)
print(f"\nMetrics saved to {CSV_PATH}")

# =============================================================================
# Timer Stop
# =============================================================================
end_time = time.perf_counter()

# Calculate the time for the model
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")