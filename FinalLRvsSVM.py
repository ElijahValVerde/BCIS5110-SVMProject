import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import mean_absolute_error,mean_squared_error

#Pandas setting to show all columns when we print dataframe
pd.set_option('display.max_columns', None,'display.max_rows', None)
#Reading the dataframe (data) from the excel file
data = pd.read_excel(r"C:\Users\18178\OneDrive\Documents\CS Masters\BCIS 5110\Group Project\Logistics Trucking Dataset.xlsx")  #For the group prject change these 2 lines to data = pd.readcsv and the file name

#completely empty columns were removed from the excel file
data = data.dropna()
#Printing information from the dataset
print('After dropping columns:',data.shape)
print(data.head())
print(data.columns)
print(data.describe())
print(data.info())
corrMatrix = data.corr()
print('Correlation Matrix:',corrMatrix)



#Q1 code to calculate the cost per order in a truckload and cost per order in a LTL (Less than truckload) to differentiate which cost is cheaper in either delivery.
#For LTL
X = data[data.LTL == "LTL"]
totalcount_Orderid_LTL = float(X["LTL"].value_counts()) #to get the total count of Order Id for LTL
print("The total count of the Order ID for LTL is:",totalcount_Orderid_LTL)
TotalCharges = X.astype({'Total Charges':'float'}) #changing to float
sum_TotalCharges_LTL = TotalCharges[["Total Charges"]].sum() #Calculating the sum of Total Charges for LTL
#Calculating Final Total charges that is sum of total charges in LTL divided by total order id count for TL
Final_TotalCharges_LTL = sum_TotalCharges_LTL/totalcount_Orderid_LTL
print('The cost per order in less than a truckload is:', Final_TotalCharges_LTL)

#For TL
Y = data[data.LTL == "TL"]
totalcount_Orderid_TL = float(Y["LTL"].value_counts()) #to get the total count of Order Id for TL
print("The total count of the Order ID for TL is:",totalcount_Orderid_TL)
TotalCharges = Y.astype({'Total Charges':'float'}) #changing to float
sum_TotalCharges_TL = TotalCharges[["Total Charges"]].sum() #Calculating the sum of Total Charges for TL
#Calculating Final Total charges that is sum of total charges in TL divided by total order id count for TL
Final_TotalCharges_TL = sum_TotalCharges_TL/totalcount_Orderid_TL
print('The cost per order in a truckload is:', Final_TotalCharges_TL)



#Q2 Code to compare the top three branches of VMC,using an analysis of how much cost, pallets, cases, weight and miles per order.
#For WI ST
X = data[data.OrigState == "WI"]
totalcount_Orderid_WI = float(X["OrigState"].value_counts()) #to get the total order counts for WI
TotalCharges = X.astype({'Total Charges':'float'}) #changing to float
sum_TotalCharges_WI = TotalCharges[["Total Charges"]].sum() #Calculating the sum of Total Charges for WI
TotalPallets = X.astype({'# Pallets':'int'})
sum_TotalPallets = TotalPallets[['# Pallets']].sum() #Calculating the sum of Total Pallets for WI
TotalWeight = X.astype({'Weight':'float'})
sum_TotalWeight = TotalWeight[['Weight']].sum() #Calculating the sum of Total Weight for WI
TotalCases = X.astype({'Cases Pieces':'int'})
sum_Cases = TotalCases[['Cases Pieces']].sum() #Calculating the sum of Total Cases for WI
TotalMiles = X.astype({'Total Miles':'float'})
sum_TM = TotalMiles[['Total Miles']].sum() #Calculating the sum of Total Miles for WI

#Calculating Total Charges per order
Final_TotalCharges_WI = sum_TotalCharges_WI/totalcount_Orderid_WI
#Calculating Total Pallets per order
Final_TotalPallets_WI = sum_TotalPallets/totalcount_Orderid_WI
#Calculating Total Weight per order
Final_TotalWeight_WI = sum_TotalWeight/totalcount_Orderid_WI
#Calculating Total Cases per order
Final_TotalCases_WI = sum_Cases/totalcount_Orderid_WI
#Calculating Total Miles per order
Final_TotalMiles_WI = sum_TM /totalcount_Orderid_WI

print("The total count of the orders for WI is:",totalcount_Orderid_WI)
print('The total cost per order in WI is: ', Final_TotalCharges_WI)
print('The total pallets per order in WI is: ', Final_TotalPallets_WI)
print('The total weight per order in WI is: ', Final_TotalWeight_WI)
print('The total cases per order in WI is: ', Final_TotalCases_WI)
print('The total miles per order in WI is: ', Final_TotalMiles_WI)

#For MN ST
Y = data[data.OrigState == "MN"]
totalcount_Orderid_MN = float(Y["OrigState"].value_counts()) #to get the total order counts for MN
TotalCharges = Y.astype({'Total Charges':'float'}) #changing to float
sum_TotalCharges_MN = TotalCharges[["Total Charges"]].sum() #Calculating the sum of Total Charges for MN
TotalPallets = Y.astype({'# Pallets':'int'})
sum_TotalPallets_MN = TotalPallets[['# Pallets']].sum() #Calculating the sum of Total Pallets for MN
TotalWeight = Y.astype({'Weight':'float'})
sum_TotalWeight_MN = TotalWeight[['Weight']].sum() #Calculating the sum of Total Weight for MN
TotalCases = Y.astype({'Cases Pieces':'int'})
sum_Cases_MN = TotalCases[['Cases Pieces']].sum() #Calculating the sum of Total Cases for MN
TotalMiles = Y.astype({'Total Miles':'float'})
sum_TM_MN = TotalMiles[['Total Miles']].sum() #Calculating the sum of Total Miles for MN

#Calculating Total Charges per order
Final_TotalCharges_MN = sum_TotalCharges_MN/totalcount_Orderid_MN
#Calculating Total Pallets per order
Final_TotalPallets_MN = sum_TotalPallets_MN/totalcount_Orderid_MN
#Calculating Total Weight per order
Final_TotalWeight_MN = sum_TotalWeight_MN/totalcount_Orderid_MN
#Calculating Total Cases per order
Final_TotalCases_MN = sum_Cases_MN/totalcount_Orderid_MN
#Calculating Total Miles per order
Final_TotalMiles_MN = sum_TM_MN /totalcount_Orderid_MN

print("The total count of the orders for MN is:",totalcount_Orderid_MN)
print('The total cost per order in MN is: ', Final_TotalCharges_MN)
print('The total pallets per order in MN is: ', Final_TotalPallets_MN)
print('The total weight per order in MN is: ', Final_TotalWeight_MN)
print('The total cases per order in MN is: ', Final_TotalCases_MN)
print('The total miles per order in MN is: ', Final_TotalMiles_MN)

#For NJ ST
Z = data[data.OrigState == "NJ"]

totalcount_Orderid_NJ = float(Z["OrigState"].value_counts()) #to get the total count of Order Id for NJ
TotalCharges = Z.astype({'Total Charges':'float'}) #changing to float
sum_TotalCharges_NJ = TotalCharges[["Total Charges"]].sum() #Calculating the sum of Total Charges for NJ
TotalPallets = Z.astype({'# Pallets':'int'})
sum_TotalPallets_NJ = TotalPallets[['# Pallets']].sum() #Calculating the sum of Total Pallets for NJ
TotalWeight = Z.astype({'Weight':'float'})
sum_TotalWeight_NJ = TotalWeight[['Weight']].sum() #Calculating the sum of Total Weight for NJ
TotalCases = Z.astype({'Cases Pieces':'int'})
sum_Cases_NJ = TotalCases[['Cases Pieces']].sum() #Calculating the sum of Total Cases for NJ
TotalMiles = Z.astype({'Total Miles':'float'})
sum_TM_NJ = TotalMiles[['Total Miles']].sum() #Calculating the sum of Total Miles for NJ

#Calculating Total Charges per order
Final_TotalCharges_NJ = sum_TotalCharges_NJ/totalcount_Orderid_NJ
#Calculating Total Pallets per order
Final_TotalPallets_NJ = sum_TotalPallets_NJ/totalcount_Orderid_NJ
#Calculating Total Weight per order
Final_TotalWeight_NJ = sum_TotalWeight_NJ/totalcount_Orderid_NJ
#Calculating Total Cases per order
Final_TotalCases_NJ = sum_Cases_NJ/totalcount_Orderid_NJ
#Calculating Total Miles per order
Final_TotalMiles_NJ = sum_TM_NJ /totalcount_Orderid_NJ

print("The total count of the orders for NJ is:",totalcount_Orderid_NJ)
print('The total cost per order in NJ is: ', Final_TotalCharges_NJ)
print('The total pallets per order in NJ is: ', Final_TotalPallets_NJ)
print('The total weight per order in NJ is: ', Final_TotalWeight_NJ)
print('The total cases per order in NJ is: ', Final_TotalCases_NJ)
print('The total miles per order in NJ is: ', Final_TotalMiles_NJ)



#Q3 Code to calculate/predict the trend of  fuel charge to predict how much fuel cost has affected the total charges
#select "target" column (Y) as a variable to predict
y = data['Fuel']
# Pick X variables based off correlation matrix
x = data[['Weight','# Orders', 'Cases Pieces', '# Pallets', 'Total Miles', 'Total Charges', 'Linehaul Charge', 'Lumper']]

#Create train and test datasets from x and y
#test_size -->  represent the proportion of the dataset to include in the test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
print("x_train",type(x_train),x_train.shape) #Getting the type of each variable
print("y_train",type(y_train))
print("x_test",type(x_test),x_test.shape)
print("y_test",type(y_test))

alg_names = ["Linear Regression","Support Vector Machine"]
alg_models = [LinearRegression(),svm.SVR()]

for alg_name,alg_model in zip(alg_names,alg_models):
    print("For algorithm",alg_name)
    alg_model.fit(x_train,y_train)
    predictions = alg_model.predict(x_test)
    print(type(predictions), predictions.shape, type(y_test), y_test.shape)
    print("Mean absolute error for",alg_name,mean_absolute_error(y_test, predictions))
    print("Mean squared error for",alg_name,mean_squared_error(y_test, predictions))

plt.plot(y_test,predictions,'.')
plt.xlabel("Actual")
plt.ylabel("Predictions")
plt.title("Compare Actual and Predictions")
#Add straight line to plot
x = np.linspace(0,350,500)
y = x
plt.plot(x,y)
#show the plot
plt.show()
