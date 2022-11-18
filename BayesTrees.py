from bartpy.sklearnmodel import SklearnModel
from sklearn.model_selection import cross_validate
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model._base import LinearRegression
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Fetching the data from API
housing_data, housing_target = fetch_california_housing(as_frame=True, return_X_y=True)
# Pulling it in
data = pd.DataFrame(pd.concat([housing_data, housing_target], axis=1))


# This gives you a look at the data
# print(pd.DataFrame.head(data, n=10))
#Descriptions of the data
# print(data.describe())


# More statistics
# print(housing_data.agg(
#     ('count','sum', 'min', 'mean', 'max')
# )
# )



#This is for time dependent variables
#train_set, test_set= np.split(data, [round(int(.70 *len(data)))])


# This is for a random split
train, test = train_test_split(data, test_size=0.2)


# Splitting into test and train
# Train
y_train= train['MedHouseVal']
x_train = train[['MedInc','HouseAge', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]

# Test
y_test= test['MedHouseVal']
x_test = test[['MedInc','HouseAge', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]



# Making the linear model
model = sm.OLS(y_train, x_train).fit()
predictions = model.predict(x_test)
predictions = pd.Series(predictions)
rms0= mean_squared_error(y_test, predictions, squared=False)
print(rms0) #0.795878






# Making the BART model
model2 = SklearnModel() # Use default parameters
model2.fit(x_train, y_train) # Fit the model
y_prediction = model2.predict(x_test)# Make predictions on the train set

#Checking to see if the values were predicted
print(y_prediction)
print("Y Predictions \n\n")

#just to provide a check both are the same size
print(len(y_prediction)); print(len(y_test))

#resetting the index so I dont get NaN values
y_prediction = pd.Series(y_prediction)
y_test= pd.Series(y_test) 



#Root Mean Squared Error
rms = mean_squared_error(y_test, y_prediction, squared=False)
print(rms) #0.578796





# Making the BART model on normalized X predictors
normalized_x_train=(x_train-x_train.mean())/x_train.std()
normalized_x_test = (x_test-x_test.mean())/x_test.std()


model3 = SklearnModel() # Use default parameters
model3.fit(normalized_x_train, y_train) # Fit the model
y_prediction_normalized = model3.predict(normalized_x_test)# Make predictions on the train set

#Checking to see if the values were predicted
print(y_prediction_normalized)
print("Y Predictions \n\n")


#making them series
y_prediction_normalized = pd.Series(y_prediction_normalized)
y_test= pd.Series(y_test) 



#Root Mean Squared Error
rms3 = mean_squared_error(y_test, y_prediction_normalized, squared=False)
print(rms3) #.60700136