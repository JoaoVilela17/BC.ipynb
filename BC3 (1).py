#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from yellowbrick.model_selection import LearningCurve
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import re
from sklearn.preprocessing import LabelEncoder


#to plot in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# # Importation of the data

# In[2]:


#Creating dataframe by loading the csv file
#This data consists of the daily sales betwee 01-10-2018 to 27-08-2019
df_sales = pd.DataFrame(pd.read_csv("Case3_Sales data.csv",sep=";"))
df_sales


# In[3]:


#Creating dataframe by loading the excel file
#This data is composed various macroeconomic indexes that involve a few countries that includes Germany, and the World
df_market = pd.DataFrame(pd.read_excel("Case3_Market data.xlsx"))
df_market


# In[4]:


#Creating dataframe by loading the csv file
#This is the data in which we will submit our final predictions
df_test_data = pd.DataFrame(pd.read_csv("Case3_Test Set Template.csv",sep=";"))
df_test_data


# The first issue we can observe from the 3 dataframes is the fact that all of them have different ways to present dates.
# 
# While df_sales has the format "01.10.2018", df_market has "2004m2", and df_test_data has "Mai 22".
# 
# We will start the code with fixing this issue.

# In[5]:


#We start off with the dataframe df_market by diving it into two parts:
df_market_name = df_market.iloc[:2,:] #one that does not consider the dates
df_market_value = df_market.iloc[2:, :]#one that only considers the dates to change


# In[6]:


#Here is the new dataframe without any dates nor numeric values
df_market_name


# In[7]:


#The dataframe that has all the dates and numerica values
df_market_value


# In[8]:


#Now we will grab df_market_value and change the dates by not only formating this way "YYYY-MM-DD"...
#but also turning these values from strings to datetime.date
#The original dataframe only had year and month on the values, we will just put all the days as "01"...
#...if this becomes a issue, it will be resolved further below

#Here is the function that will convert the values into datetime and also convert it to the format above indicated
def convert_time_months(time_str):
    datetime_obj = dt.datetime.strptime(time_str.strip(), '%Ym%m')
    return datetime_obj.date()

#Now we will apply the function above
df_market_value['Unnamed: 0'] = df_market_value['Unnamed: 0'].apply(convert_time_months) #creation of a new column from the original one wiht the function

#Here is the result
df_market_value


# In[9]:


#Merging the two dataframes to create the original one with the alteration fo the date
df_market = pd.concat([df_market_name, df_market_value])

#Removing columns with nan values
df_market = df_market.dropna(axis=1, how='all')

#Making the column with the dates the index to not interfere with the algorithms that will be used...
#...the algorithms in question do not work with strings, only numeric values
df_market = df_market.set_index('Unnamed: 0')
df_market.index.name = 'Date' #Changing the name of the index

#Changing the index value of the row named "date" to "Abbreviaton" that better fits the values of the row
df_market.rename(index={'date': 'Abbreviation'}, inplace=True)


# In[10]:


#This is how the dataframe looks like after the alterations
df_market


# In[11]:


#Now we will make the exact same process for the dataframe df_sales
#For this one we do not need to divide the dataframe and there are days in the original date

#We are creating a very similar function to the one above, but this one considers that the value entered as formated as "DD.MM.YYYY"
def convert_time_days(time_str):
    datetime_obj = dt.datetime.strptime(time_str.strip(), '%d.%m.%Y')
    return datetime_obj.date()

#Using the function to apply the changes as explained in the cells above
df_sales['DATE'] = df_sales['DATE'].apply(convert_time_days)
df_sales = df_sales.set_index('DATE')

#Changing the name of the index from "DATE" to "date" to be the same as df_market
df_market.rename(index={'DATE': 'Date'}, inplace=True)


# In[12]:


#This is how the dataframe looks like after the alterations
df_sales


# In[13]:


#This code alters the first row that describes the data to add the value of the column, for example, "China", adds ": " and then it's original value
#The reason for this change is to remove the first row and make the column more redundant whilst keeping all the information
df_market.iloc[0] = df_market.columns.map(lambda col: col + ": " + str(df_market[col].iloc[0]))


# In[14]:


df_market


# Considering the output, whilst it does give all the information needed to interpret the data, it is not consistent.
# 
# For example, whilst the first column is "China: Production Index Machinery & Electricals" the penultimate is "production index.14: France: Electrical equipment".
# 
# We will change it to keep this format: "[country] [category] [sub-category]"

# In[15]:


#This list was created considering the values in the dataframe and the format indicated above
first_row_replacement = ['China: Production Index Machinery & Electricals',
'China: Shipments Index Machinery & Electricals',
'France: Production Index Machinery & Electricals',
'France: Shipments Index Machinery & Electricals',
'Germany: Production Index Machinery & Electricals',
'Germany: Shipments Index Machinery & Electricals',
'Italy: Production Index Machinery & Electricals',
'Italy: Shipments Index Machinery & Electricals',
'Japan: Production Index Machinery & Electricals',
'Japan: Shipments Index Machinery & Electricals',
'Switzerland: Production Index Machinery & Electricals',
'Switzerland: Shipments Index Machinery & Electricals',
'United Kingdom: Production Index Machinery & Electricals',
'United Kingdom: Shipments Index Machinery & Electricals',
'United States: Production Index Machinery & Electricals',
'United States: Shipments Index Machinery & Electricals',
'Europe: Production Index Machinery & Electricals',
'Europe: Shipments Index Machinery & Electricals',
'World: Price of Base Metals',
'World: Price of Energy',
'World: Price of Metals & Minerals',
'World: Price of Natural Gas Index',
'World: Price of Crude Oil, Average',
'World: Price of Copper',
'United States: EUR in LCU',
'United States: Producer Prices Electrical Equipment',
'United Kingdom: Producer Prices Electrical Equipment',
'Italy: Producer Prices Electrical Equipment',
'France: Producer Prices Electrical Equipment',
'Germany: Producer Prices Electrical Equipment',
'China: Producer Prices Electrical Equipment',
'United States: Production Index Machinery and Equipment n.e.c.',
'World: Production Index Machinery and Equipment n.e.c.',
'Switzerland: Production Index Machinery and Equipment n.e.c.',
'United Kingdom: Production Index Machinery and Equipment n.e.c.',
'Italy: Production Index Machinery and Equipment n.e.c.',
'Japan: Production Index Machinery and Equipment n.e.c.',
'France: Production Index Machinery and Equipment n.e.c.',
'Germany: Production Index Machinery and Equipment n.e.c.',
'United States: Production Index Electrical Equipment',
'World: Production Index Electrical Equipment',
'Switzerland: Production Index Electrical Equipment',
'United Kingdom: Production Index Electrical Equipment',
'Italy: Production Index Electrical Equipment',
'Japan: Production Index Electrical Equipment',
'France: Production Index Electrical Equipment',
'Germany: Production Index Electrical Equipment']

#Now we will replace the first row's values with this new list to make it easier to interpret and read
df_market.loc["Index 2010=100 (if not otherwise noted)"] = first_row_replacement

#We will replace the column values with this first row that has both the values...
df_market.columns = df_market.iloc[0]
#...and drop the first row (which is now the column names)
df_market = df_market.drop(df_market.index[0])


# In[16]:


df_market


# The restructuring of the dataframe df_market is almost complete. There is a few issues that will be addressed now that we can take away from the output above.
# 
# The first issue is the fact that whilst we did all possible to remove redundant parts, we still have the row with the index "Abbreviation" that do not represent numerica values. This will make the algorithim unable to analyse the data. So we will remove this row and, if needed in the future, add it back
# 
# Another possible is issue is that the indexes that indicate the date also appear to show the the hours, we will also remove these whilst keeping them as datetime.

# In[17]:


#First we will convert the first row to save it if needed further below
abbreviations = df_market.loc["Abbreviation"]

#Second, we remove it from the dataframe
df_market = df_market.drop(index='Abbreviation')

#Third, we now remove the hours from the index of the dataframe
df_market.index = pd.to_datetime(df_market.index).date
#We will do the same for df_sales just in case it suffers the same problem
df_sales.index = pd.to_datetime(df_sales.index).date


# In[18]:


df_market


# The df_market, in terms of strucutre, is finished. Now we must merge df_sales with it.
# 
# The biggest issue that has to do with index. Whilst df_sales' index shows the specific day of the sale, df_market only has data related to month and all the days are set to "01".
# 
# To resolve this issue, we will group the index and column Mapped_GCK of df_sales whilst summing them to both dataframes to be in months.

# In[19]:


df_sales


# In[20]:


#First, we will sum the values of the sales, grouping by their type, and the the year and month...
#...after this we will merge both the dataframes

#To sum the different values of the sales, we must turn the values into floats, right now they are strings and have...
#...commas instead of points
df_sales.Sales_EUR = df_sales.Sales_EUR.str.replace(",",".").astype(float)

#We will turn the index into datetime to be able to group by month
df_sales.index = pd.to_datetime(df_sales.index)
#Grouping by month and also type of product
monthly_sales = df_sales.groupby([pd.Grouper(freq='M'), 'Mapped_GCK']).sum()

#Now we reset the index to be able to remove the day of the date
monthly_sales = monthly_sales.reset_index()
#Removing the days from the new formed column
monthly_sales['Date'] = monthly_sales['level_0'].dt.to_period('M')
#Setting it back the date to be the index
monthly_sales = monthly_sales.set_index(['Date'])
#Dropping the column that we just created
monthly_sales = monthly_sales.drop('level_0', axis=1)
#Converting the dataframe created to be the new df_sales
df_sales = monthly_sales


# In[21]:


df_sales


# The dataframe, df_sales, looks ready to merge with df_market.

# In[22]:


df_sales


# In[23]:


#Now we will rewrite the code to adjust to the dataframe df_market

#We will turn the index into datetime to be able to group by month
df_market.index = pd.to_datetime(df_market.index)

#Now we reset the index to be able to remove the day of the date
monthly_sales = df_market.reset_index()
#Removing the days from the new formed column
monthly_sales['Date'] = monthly_sales['index'].dt.to_period('M')
#Setting it back the date to be the index
monthly_sales = monthly_sales.set_index(['Date'])
#Dropping the column that we just created
monthly_sales = monthly_sales.drop('index', axis=1)
#Converting the dataframe created to be the new df_market
df_market = monthly_sales


# In[24]:


df_market


# Now that both dataframes have the same index, we can now merge them together to create our final dataframe

# In[25]:


df = pd.merge(df_sales, df_market, on='Date', how="inner")
df


# # 1 - Exploration

# ## 1.1. - Data Analysis
# 
# Now that the dataframes are combined and ready to be used, we will explore it to better understand it and fix any issues.

# In[26]:


#We will start by checking how many rows and columns our dataframe has
df.shape


# In[27]:


#Now we will verify what data types our columns are adn check for any missing values
df.info()


# As we can verify above, the only issue derives from the data from df_market.
# 
# All are objects and some have missing values.
# 
# To fix this problem we will first conver them to float and fill the missing values with knn.
# 
# The reason for the use of knn derives from the fact that we are looking at a time series. Any other simpler method would consider all the data, independently of date, to be as important as the closest dates.

# In[28]:


#We start by creating a list to indicate which columns we want to change
#In this case it's just the columns of df_market
cols_to_convert = df_market.columns.tolist()
#We convert said columns to floats
df[cols_to_convert] = df[cols_to_convert].astype(float)

#Then we start using knn
#Although knn only considers data within the column to replace the empty values...
#...the algorithim itself does not allow to any column that isn't numeric to be within the dataframe...
#...so, to make it easier, we will jus remove Mapped_GCK and put it back in the end
product_group = df.Mapped_GCK
df = df.drop('Mapped_GCK', axis=1)
#Imputting the missing values of the dataframe while using the 5 nearest neighboors
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
#Adding back Mapped_GCK
df['Mapped_GCK'] = product_group


# In[29]:


df.describe()


# The only part of the output that could be considered to be incoherent is the fact that sales have negative values. But this can be explained with the devolution and creation of credit to the customers in question.

# In[30]:


# convert the 'Date' index from Period to datetime
df.index = df.index.to_timestamp()

filtered_df = df[df['Mapped_GCK'] == '#1']

# plot the 'Sales_EUR' column
plt.figure(figsize=(15,5))
plt.plot(filtered_df['Sales_EUR'])
plt.show()


# ## DELETE THIS
# One advantage of additive decomposition is that it can be easier to interpret the individual components since they are added together to create the original time-series. Additionally, additive decomposition tends to work well for time-series data where the magnitude of the seasonal variation is roughly constant over time.
# 
# On the other hand, one advantage of multiplicative decomposition is that it can be more appropriate for time-series data where the seasonal variation is proportional to the level of the time-series. This can often be the case with economic or financial data, where the seasonal variation increases with the overall level of economic activity.

# In[31]:


df[df['Mapped_GCK'] == '#1']


# ## Have different views of the data

# In[32]:


# Set period and extrapolate trend
period = 12
extrapolate_trend = 'freq'

# Multiplicative decomposition
result_mul = seasonal_decompose(df[df['Mapped_GCK'] == '#1'].Sales_EUR, model='multiplicative', period=period, extrapolate_trend=extrapolate_trend)

# Additive decomposition
result_add = seasonal_decompose(df[df['Mapped_GCK'] == '#1'].Sales_EUR, model='additive', period=period, extrapolate_trend=extrapolate_trend)

# Plot the results
plt.rcParams.update({'figure.figsize': (15,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()


# ## Data preparation

# In[33]:


#ONLY HAS SALES VALUES
gck1 = df.loc[df['Mapped_GCK'] == '#1', ['Sales_EUR']]


# In[34]:


# Function to create lag features
# Adapted from https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, varNames=None):
    """"
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
        varNames: List of column names (same size as the number of variables).
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(varNames[j]+'(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(varNames[j]+'(t)' ) for j in range(n_vars)]
        else:
            names += [(varNames[j]+'(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[35]:


df


# In[36]:


# Create lag features
lagDays = 10
X = series_to_supervised(gck1.values, lagDays, 1, varNames=gck1.columns.values.tolist())
X.set_index(gck1.index[lagDays:], inplace=True)
X


# In[37]:


# Replace 'df.columns[3]' with the name of your desired column
columns_of_interest = df.columns[3:]

top_corr_columns = []

for column_name in columns_of_interest:
    de_ppee = df.loc[df['Mapped_GCK'] == '#1', [column_name]]

    # Create lag features
    corr_de_ppee = series_to_supervised(de_ppee, lagDays, 1, varNames=de_ppee.columns.values.tolist())
    corr_de_ppee.set_index(de_ppee.index[lagDays:], inplace=True)

    corr_de_ppee = pd.merge(gck1, corr_de_ppee, on='Date', how="inner")

    # Select the first column and all other columns
    corr_de_ppee_sub = corr_de_ppee.iloc[:, [0] + list(range(1, corr_de_ppee.shape[1]))]

    # Calculate the Spearman correlation coefficients between the first column and all other columns
    corr = corr_de_ppee_sub.corrwith(corr_de_ppee_sub.iloc[:, 0], method='spearman')

    # Remove the correlation coefficient between the first column and itself
    corr = corr.iloc[1:]

    if corr.empty:
        top_corr_columns.append("No correlation found")
    else:
        # Get the column name with the highest correlation
        highest_corr_column = corr.idxmax()

        # Add the column name to the list of top correlated columns
        top_corr_columns.append(highest_corr_column)


# In[38]:


top_corr_columns


# In[39]:


top_corr_columns.remove("No correlation found")


# In[40]:


top_corr_columns = list(filter(lambda x: 'Germany' in x, top_corr_columns))


# In[41]:


top_corr_columns


# In[42]:


for column in top_corr_columns:
    de_ppee = df.loc[df['Mapped_GCK'] == '#1', [re.sub(r'\(.*\)', '', column)]]

    # Create lag features
    corr_de_ppee = series_to_supervised(de_ppee, lagDays, 1, varNames=de_ppee.columns.values.tolist())
    corr_de_ppee.set_index(de_ppee.index[lagDays:], inplace=True)
    corr_de_ppee = pd.merge(gck1, corr_de_ppee, on='Date', how="inner")

    #gck1.index = pd.to_datetime(gck1.index)
    X[column] = corr_de_ppee[column]


# In[43]:


X


# In[44]:


subPlots_Title_fontSize = 12
subPlots_xAxis_fontSize = 10
subPlots_yAxis_fontSize = 10
subPlots_label_fontSize = 10
heatmaps_text_fontSize = 8

plots_Title_fontSize = 14
plots_Title_textColour = 'black'

plots_Legend_fontSize = 12
plots_Legend_textColour = 'black'

plots_barTexts_fontSize = 8


# In[45]:


#Now we will create a correlation matrix

#Considering the amount of variables presented, most won't have a linear realitionship between each other, therefore we will use the Speraman's correlation
#We will still check the Pearson one further below
corr = X.corr(method='spearman')
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True

# Draw
fig , ax = plt.subplots(figsize=(15, 20))
heatmap = sns.heatmap(corr,
                      mask = mask,
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4,
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      fmt='.4f',
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      annot_kws = {'size': heatmaps_text_fontSize})

# Decoration
plt.title("Spearman correlation between numeric variables", fontsize=plots_Title_fontSize)
ax.set_yticklabels(corr.columns, rotation = 0)
ax.set_xticklabels(corr.columns, rotation = 45)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Modeling

# In[46]:


# Create the Target
y = X['Sales_EUR(t)']


# ### WHAT SHOULD BE THE NUMBER OF MONTHS OF THE FORECAST?

# In[47]:


# Split into train and test

#this is the number of forecasts, in this case I'm putting 1 year, replace the terms
days_forecast = 12

y_train = y[:-days_forecast]
y_test = y[-days_forecast:]

X_train = X[:-days_forecast]
X_test = X[-days_forecast:]


# In[48]:


# Remove the Target from the training
X_train = X_train.drop(labels=['Sales_EUR(t)'],axis=1)
X_test = X_test.drop(labels=['Sales_EUR(t)'],axis=1)


# In[49]:


# List maker function
def listmaker(value, n):
    listofv = [value] * n
    return listofv

# Create a temporary DF to scale data to predifined min and max
tempDF = pd.DataFrame(columns=X_train.columns)
values = listmaker(0, tempDF.shape[1]-1)  # set a row with 0 as minimum
values.append(0)
tempDF.loc[len(tempDF)] = values
values = listmaker(100, tempDF.shape[1]-1)  # set a row with 100 as maximum
values.append(1)
tempDF.loc[len(tempDF)] = values

# Normalize training data
scaler = MinMaxScaler(feature_range=(0, 1))
tempDF_scaled = scaler.fit_transform(tempDF)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[50]:


X_test


# In[51]:


model = MLPRegressor(random_state=21, hidden_layer_sizes=6, max_iter=400)


# ## There is no cross validation?!!?

# In[52]:


# Plot the learning curve
cv = 10
sizes = np.linspace(0.1, 1.0, 5)
visualizer = LearningCurve(estimator=model, cv=cv, scoring='r2', train_sizes=sizes, n_jobs=4, random_state=123)
visualizer.fit(X_train_scaled, y_train)
visualizer.show()


# In[53]:


# Create and train the model
model.fit(X_train_scaled, y_train)


# ## Evaluation

# In[54]:


# predict y for X_train and X_test
y_pred_train = model.predict(X_train_scaled) 
y_pred_test = model.predict(X_test_scaled) 


# In[55]:


# Function to create dataframe with metrics
def performanceMetricsDF(metricsObj, yTrain, yPredTrain, yTest, yPredTest,set1='Train', set2='Test'):
  measures_list = ['MAE','RMSE', 'R^2','MAPE (%)','MAX Error']
  train_results = [metricsObj.mean_absolute_error(yTrain, yPredTrain),
                np.sqrt(metricsObj.mean_squared_error(yTrain, yPredTrain)),
                metricsObj.r2_score(yTrain, yPredTrain),
                metricsObj.mean_absolute_percentage_error(yTrain, yPredTrain),
                metricsObj.max_error(yTrain, yPredTrain)]
  test_results = [metricsObj.mean_absolute_error(yTest, yPredTest),
                np.sqrt(metricsObj.mean_squared_error(yTest, yPredTest)),
                metricsObj.r2_score(yTest, yPredTest),
                  metricsObj.mean_absolute_percentage_error(yTest, yPredTest),
                metricsObj.max_error(yTest, yPredTest)]
  resultsDF = pd.DataFrame({'Measure': measures_list, set1: train_results, set2:test_results})
  return(resultsDF)


# In[56]:


# Show performance results
resultsDF = performanceMetricsDF(metrics, y_train, y_pred_train, y_test, y_pred_test)
resultsDF


# In[57]:


# Visualize predictions vs close values 
temp = y_test.to_frame()
temp['Prediction'] = y_pred_test
temp['Residual'] = y_test - temp.Prediction
temp


# In[58]:


# Plot close price time series
plt.figure(figsize=(15,5))
plt.plot(X[:-days_forecast]['Sales_EUR(t)'])
plt.plot(temp[['Sales_EUR(t)', 'Prediction']])


# #### With walk-forward approach

# In[59]:


# Create first element based on first row of X_test
X_valid = X_test[:1]
y_valid = pd.Series([],dtype=np.float)
n = 0
for date in pd.to_datetime(X_test.index.values):
    # Get current day values
    currentDay = X_valid[X_valid.index==date]
    currentDay_scaled = scaler.transform(currentDay.values) # scaled using the same scaler, not fiting a new one

    # Predict an y for each date and add it to results series
    y_valid[n] = model.predict(currentDay_scaled)[0]
    n = n + 1

    # Add a new row to X_valid
    if n < days_forecast:
        X_valid = X_valid.append(X_test.iloc[n])
        newDate = X_valid.index[-1]

        # Shift values 1 day
        copyToCol = ['Sales_EUR(t-'+str(j)+')' for j in range(lagDays,2, -1)]
        copyFromCol = ['Sales_EUR(t-'+str(j)+')' for j in range(lagDays-1,1, -1)]
        shiftValues = X_valid[X_valid.index==date][copyFromCol].values.tolist()
        X_valid.loc[X_valid.index==newDate, copyToCol] = shiftValues
    
        # Set last day with the value predicted
        X_valid.loc[X_valid.index==newDate,'Sales_EUR(t-1)'] = y_valid[n-1]


# In[60]:


# Show performance results
resultsDF = performanceMetricsDF(metrics, y_test, y_pred_test, y_test, y_valid, 'Test', 'Valid')
resultsDF


# In[61]:


# Visualize predictions vs close values 
temp = y_test.to_frame()
temp['Prediction_WalkForward'] = y_valid.values
temp['Residual'] = temp['Sales_EUR(t)'] - y_valid.values
temp


# In[62]:


# Plot forecast
plt.figure(figsize=(15,5))
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(temp['Prediction_WalkForward'], label='Prediction WF')
plt.legend(loc='upper left')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




