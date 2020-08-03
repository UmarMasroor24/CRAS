import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from datetime import datetime
from datetime import date
from time import time
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from sklearn.preprocessing import StandardScaler
import time
import config

print("success")

class Pipeline:
	def __init__(self):


		self.df_list = None
		self.train_df_list = None
		self.test_df_list = None
		self.Ex_time = None





	def profile_raw_data(self, df, log):

		# Get the last row number
		last_row_number = log.shape[0]

		# Get the last row
		last_row = log.loc[[last_row_number - 1]]

		# Extract Backlog Id
		ProcessID = last_row.iloc[0]['ProcessID']

		# Importing Data From File
		data = pd.read_csv("Variable_Table.csv")

		# Fetch names of variables from the variables column
		s1 = data['Variables'].values

		# Select the columns of only the variables we need
		df = df[s1]

		# save data profiles in variables
		prof_median = df.median(numeric_only=bool)
		prof_unique = df.nunique(dropna=True)
		prof_mean = df.mean(numeric_only=bool)
		prof_mode = df.mode(numeric_only=bool)
		prof_min = df.min(numeric_only=bool)
		prof_max = df.max(numeric_only=bool)
		prof_quant_25 = df.quantile(numeric_only=bool,q=0.25)
		prof_quant_75 = df.quantile(numeric_only=bool,q=0.75)
		prof_std = df.std(numeric_only=bool)
		prof_nan_count = len(df) - df.count()

		# create a dataframe to be exported
		result = pd.concat([prof_median,prof_unique, prof_mean, prof_min, prof_max, prof_quant_25, prof_quant_75,prof_std,prof_nan_count], axis=1)

		# reset index
		result.reset_index(level = 0, inplace =True)

		# rename columns
		result = result.set_axis(['VariableName','Median','Unique_values','Mean','Minimum','Maximum', 'Quantile_25', 'Quantile_75', 'Std_Dev','Null_Count'], axis=1, inplace=False)

		# insert current date into the column
		result['Date'] = str(date.today())

		# insert current time into the column
		result['Time'] = str(datetime.now().time())

		# insert backlog id into column
		result['ProcessID'] = ProcessID

		# insert Data Stage
		result['Data_Stage'] = 'Raw'

		# insert mode column as NULL (For now)
		result['Mode'] = ''

		file_name = "Profile_Table.csv"

		# Exporting the file
		# Check the availability of the file
		if os.path.exists(file_name):

			# don't include headers if the file already exists
			result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a', header=False)
		else:

			# include headers if the file does not exist
			result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a')

	def profile_processed_data(self, df, log):

		# Get the last row number
		last_row_number = log.shape[0]

		# Get the last row
		last_row = log.loc[[last_row_number - 1]]

		# Extract Backlog Id
		ProcessID = last_row.iloc[0]['ProcessID']

		# Importing Data From File
		data = pd.read_csv("Variable_Table.csv")

		# Fetch names of variables from the variables column
		s1 = data['Variables'].values

		# Select the columns of only the variables we need
		df = df[s1]

		# save data profiles in variables
		prof_median = df.median(numeric_only=bool)
		prof_unique = df.nunique(dropna=True)
		prof_mean = df.mean(numeric_only=bool)
		prof_mode = df.mode(numeric_only=bool)
		prof_min = df.min(numeric_only=bool)
		prof_max = df.max(numeric_only=bool)
		prof_quant_25 = df.quantile(numeric_only=bool,q=0.25)
		prof_quant_75 = df.quantile(numeric_only=bool,q=0.75)
		prof_std = df.std(numeric_only=bool)
		prof_nan_count = len(df) - df.count()

		# create a dataframe to be exported
		result = pd.concat([prof_median,prof_unique, prof_mean, prof_min, prof_max, prof_quant_25, prof_quant_75,prof_std,prof_nan_count], axis=1)

		# reset index
		result.reset_index(level = 0, inplace =True)

		# rename columns
		result = result.set_axis(['VariableName','Median','Unique_values','Mean','Minimum','Maximum', 'Quantile_25', 'Quantile_75', 'Std_Dev','Null_Count'], axis=1, inplace=False)

		# insert current date into the column
		result['Date'] = str(date.today())

		# insert current time into the column
		result['Time'] = str(datetime.now().time())

		# insert backlog id into column
		result['ProcessID'] = ProcessID

		# insert Data Stage
		result['Data_Stage'] = 'Processed'

		# insert mode column as NULL (For now)
		result['Mode'] = ''

		file_name = "Profile_Table.csv"

		# Exporting the file
		# Check the availability of the file
		if os.path.exists(file_name):

			# don't include headers if the file already exists
			result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a', header=False)
		else:

			# include headers if the file does not exist
			result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a')

	#below function does imputation by filling nulls with mean of the column
	def Imputation(self, df):
		#loop through dataframe and check for nulls in each column
		for (columnName, columnData) in df.iteritems():
			temp = df[[columnName]]
			#if nulls is less than the threshold then fill all the nulls of column with mean of the column
			if temp.isnull().sum() < self.Threshold:
				temp = fillna(temp.mean())
				df[columnName] = temp
		#return the dataframe after imputation
		return df

	def Remove_Null(self,df):
		#remove all the nulls and return the dataframe

		df = df.dropna()
		return df

	# def Data_Aggregation(self, df):
		#Aggregate data on specific level according to grain
		#Values of Media variables will remain same
		#Price will be same
		#Incentives will add up
		#above Approach is valid if aggregation is being done on category level or product level for same period
		#if aggregation is done on according to grain of period then approach will be changed

	# def var_encoding(self,df):
	# 	dum_df = pd.get_dummies(df, columns=[var_encode] )
	# 	# merge with main df bridge_df on key values
	# 	df = df.join(dum_df)
	# 	return df

	# def varaible_scaling(self,df):
	# 	ss = StandardScaler()
	# 	d = df[reduced_Var]
	# 	df_scaled = pd.DataFrame(ss.fit_transform(df),columns = reduced_Var)
	# 	return df_scaled

	#def Drived_Variables(self,df):
		#we will create variables if needed in this function

	#def feature_selection(self,df):
		#Feature Selection will be done in this function and selected
		#features will be written in variable table.

	#This function checks for erroneous_column and returns column name containing error
	def Erroneous_Column(self,df,Threshold):
		#make a list of erroneous columns
		Err_col = []
		#Iterate through the dataframe and calculate percentage null points in each columns
		for (columnName, columnData) in df.iteritems():
			temp = df[columnName]
			Per_Nulls = temp.isnull().sum()/temp.shape[0]
			#if the percentage of null points is greater than the threshold mark it as erroneous column
			if Per_Nulls > Threshold:
				Err_col = Err_col + [columnName]
		#if only one column is erroneous return name of the column
		if len(Err_col) == 1:
			return Err_col[0]
		#if no column is erroneous return False
		if len(Err_col) == 0:
			return False
		#return the erroneous columns if exist
		else:
			return Err_col

	#This function removes the erroneous_column and returns dataframe after removal of the erroneous column
	def Remove_Err_column(self,df, col):
		#drop the erroneous column and return dataframe
		df = df.drop(col,axis = 1)
		return df

	#This function checks for empty dataset and returns true if dataset is empty
	def Empty_Dataset(self, df):
		#reeturn true if the dataset is empty
		return len(df.index) == 0

#'PackageName','UserStory','ErrorType','ExecutionTime','ProcessID'
	def Error_log(self, df,log,Threshold):
		#name of the table
		file_name  =  "Error_log.csv"
		check = 0
		#check if error_log table exists
		if os.path.exists(file_name):
			pass
		#if error_log table is missing then make error_log table
		else:
			temp = pd.DataFrame(columns = ['PackageName','UserStory','ErrorType','ExecutionTime','ProcessID'])
			temp.to_csv('Error_log.csv')
		#load error_table
		temp = pd.read_csv('Error_log.csv')
		#check if dataframe is empty if yes then that is our error
		#print(temp.loc[temp.shape[0]])
		if self.Empty_Dataset(df):
			temp = temp.append({'PackageName':'Preprocessing','UserStory':'Empty Dataset','ErrorType':'Dataset Missing','ExecutionTime':str(datetime.now().time()),'ProcessID':log.iloc[log.shape[0]]['ProcessID']},ignore_index=True)
			check = 1

		#process id
		p = log.iloc[log.shape[0]-1]['ProcessID']
		#erroneous column
		print(p,'118')
		col = self.Erroneous_Column(df,Threshold)
		if col != False:
			temp = temp.append({'PackageName':'Preprocessing','UserStory':'Error in columns','ErrorType':'Nulls in Column','ExecutionTime':str(datetime.now().time()),'ProcessID':log.iloc[log.shape[0]-1]['ProcessID']},ignore_index=True)
			check = 1
		if check == 1:
			#update the error_log
			temp.to_csv('Error_log.csv')
		return self


#'LogID','Date','PackageName','UserStory','Status','StartingTime','ExecutionTime','ProcessID','Region','SKU','algorithm'
	def Log_Table(self,log):
		#name of the log_table
		file_name = "Log_Table.csv"
		check = 0
		#check if log_table exists
		if os.path.exists(file_name):
			check = 1
			pass
		else:
			temp = pd.DataFrame(columns =['LogID','Date','PackageName','UserStory','Status','StartingTime','ExecutionTime','ProcessID','Region','SKU','algorithm'])
			#if the log_table does't exist assign logid 1 when creating log_table
			logid = 1
		if check == 1:
			#if path exists read the log_table
			temp = pd.read_csv('Log_Table.csv')
			logid = temp.iloc[temp.shape[0]-1]['LogID']
		p = log.iloc[log.shape[0]-1]['ProcessID']#processID
		SKU = log.iloc[log.shape[0]-1]['SKU']#SKU
		Region = log.iloc[log.shape[0]-1]['Region']#Region
		Algo = log.iloc[log.shape[0]-1]['Models']#Algorithm
		dat = str(date.today())#current date
		st_time = str(datetime.now().time())#current time
		exe_time = self.Ex_time #execution time of preprocessing

		#append the log table with the updated log values
		temp = temp.append({'LogID':logid,'Date':dat,'PackageName':'Preprocessing','UserStory':'Complete preprocessing flow','Status':'Done','StartingTime':st_time,'ExecutionTime':exe_time,'ProcessID':p,'Region':Region,'SKU':SKU,'algorithm':Algo},ignore_index=True)

		#write it back
		temp.to_csv('Log_Table.csv')



	def Raw_Data_Cleaning(self, data,Threshold):
		#check for erroneous columns
		column = self.Erroneous_Column(data,Threshold)
		if column != False:
			#remove erroneous columns
			data = self.Remove_Err_column(data,column)
		else:
			data = data.copy()
		#remove nulls once removed erroneous columns
		data = self.Remove_Null(data)

		#return clean data
		return data


	def raw_data_prep(self,data):

		# Per Region, Per Sku Division of raw data
		self.raw_df_list = [ [ None for i in range(self.skucount) ] for j in range(self.rcount) ]

		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:
				temp = data.loc[(data[self.SKU_col] == self.SKU_list[j]) & (data[self.Region_col] == self.region_list[i])]
				self.raw_df_list[i][j] = temp

				j += 1
			i += 1

		#print('raw data prepared')
		return self

	def Data_Prep(self, data, SKU_col, SKU_list,Region_col,region_list,Threshold):
		#clean data

		self.SKU_col = SKU_col
		self.SKU_list = SKU_list
		self.Region_col = Region_col
		self.region_list = region_list

		column = self.Erroneous_Column(data,Threshold)
		if column != False:
			data = self.Remove_Err_column(data,column)
		else:
			data = data.copy()
		data = self.Remove_Null(data)

		self.rcount = len(region_list)
		self.skucount = len(SKU_list)

		# Breaking ads into 74x7 datasets
		self.df_list = [ [ None for i in range(self.skucount) ] for j in range(self.rcount) ]
		# Breaking ads into 74x7 datasets

		#preparing 74x7 Datasets
		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:
				temp = data.loc[(data[SKU_col] == SKU_list[j]) &  (data[Region_col] == region_list[i])]
				self.df_list[i][j] = temp


				j += 1
			i += 1
		return self

	#save prepared datasets using pickle
	def Save_data(self):
		with open("Prepared_DataSets",'wb') as t:
			pickle.dump(self.df_list, t)

	def train_test_split(self,SKU_list,region_list,Year,Training_End,Prediction_Year,Month,Prediction_month):
		#load 74x7 datasets
		dff_list = self.df_list

		#count of region
		self.rcount = len(region_list)

		#count of SKU
		self.skucount = len(SKU_list)

		#training datasets
		self.train_df_list = [ [ None for i in range(self.skucount) ] for j in range(self.rcount) ]

		#testing datasets
		self.test_df_list = [ [ None for i in range(self.skucount) ] for j in range(self.rcount) ]

		#training and testing datsets preparation
		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:
				temp = dff_list[i][j]
				self.train_df_list[i][j] = temp[(temp[Year] <= Training_End)]
				self.test_df_list[i][j] = temp[(temp[Year] == Prediction_Year) & (temp[Month] <= Prediction_month)]
				j += 1
			i += 1

	def save_train_test_data(self):
		with open("Prepared_Train_DataSets",'wb') as t:
			pickle.dump(self.train_df_list, t)
		with open("Prepared_Test_DataSets",'wb') as t:
			pickle.dump(self.test_df_list, t)
		return self


	def raw_data_profile(self, log, reduced_Var):

		# Get the last row number
		last_row_number = log.shape[0]

		# Get the last row
		last_row = log.loc[[last_row_number - 1]]

		# Extract Backlog Id
		ProcessID = last_row.iloc[0]['ProcessID']

		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:

				# Select reduced columns from dataframe
				df = self.raw_df_list[i][j][reduced_Var]

				# save data profiles in variables
				prof_median = df.median(numeric_only=bool)
				prof_unique = df.nunique(dropna=True)
				prof_mean = df.mean(numeric_only=bool)
				prof_mode = df.mode(numeric_only=bool)
				prof_min = df.min(numeric_only=bool)
				prof_max = df.max(numeric_only=bool)
				prof_quant_25 = df.quantile(numeric_only=bool,q=0.25)
				prof_quant_75 = df.quantile(numeric_only=bool,q=0.75)
				prof_std = df.std(numeric_only=bool)
				prof_nan_count = len(df) - df.count()


				# create a dataframe to be exported
				result = pd.concat([prof_median,prof_unique, prof_mean, prof_min, prof_max, prof_quant_25, prof_quant_75,prof_std,prof_nan_count], axis=1)

				# reset index
				result.reset_index(level = 0, inplace =True)

				# rename columns
				result = result.set_axis(['VariableName','Median','Unique_values','Mean','Minimum','Maximum', 'Quantile_25', 'Quantile_75', 'Std_Dev','Null_Count'], axis=1, inplace=False)

				# insert current date into the column
				result['Date'] = str(date.today())

				# insert current time into the column
				result['Time'] = str(datetime.now().time())

				# insert backlog id into column
				result['ProcessID'] = ProcessID

				# insert Data Stage
				result['Data_Stage'] = 'Raw'

				# insert mode column as NULL (For now)
				result['Mode'] = ''

				# insert region column
				test = self.raw_df_list[i][j].Region.unique()

				# insert region column
				result['Region'] = test[0]

				# insert sku column
				test1 = self.raw_df_list[i][j].m_sku_desc.unique()
				result['Sku'] =test1[0]

				# Name of the file we want to save profile to
				file_name = "Profile_Table.csv"

				# Exporting the file
				# Check the availability of the file
				if os.path.exists(file_name):

					# don't include headers if the file already exists
					result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a', header=False)
				else:

					# include headers if the file does not exist
					result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a')


				j += 1
			i += 1

	def processed_data_profile(self, log, reduced_Var):

		# Get the last row number
		last_row_number = log.shape[0]

		# Get the last row
		last_row = log.loc[[last_row_number - 1]]

		# Extract Backlog Id
		ProcessID = last_row.iloc[0]['ProcessID']

		# Loop through all the dataframes
		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:

				# Select reduced variables from the training dataset
				df = self.train_df_list[i][j][reduced_Var]

				# save data profiles in variables
				prof_median = df.median(numeric_only=bool)
				prof_unique = df.nunique(dropna=True)
				prof_mean = df.mean(numeric_only=bool)
				prof_mode = df.mode(numeric_only=bool)
				prof_min = df.min(numeric_only=bool)
				prof_max = df.max(numeric_only=bool)
				prof_quant_25 = df.quantile(numeric_only=bool,q=0.25)
				prof_quant_75 = df.quantile(numeric_only=bool,q=0.75)
				prof_std = df.std(numeric_only=bool)
				prof_nan_count = len(df) - df.count()


				# create a dataframe to be exported
				result = pd.concat([prof_median,prof_unique, prof_mean, prof_min, prof_max, prof_quant_25, prof_quant_75,prof_std,prof_nan_count], axis=1)

				# reset index
				result.reset_index(level = 0, inplace =True)

				# rename columns
				result = result.set_axis(['VariableName','Median','Unique_values','Mean','Minimum','Maximum', 'Quantile_25', 'Quantile_75', 'Std_Dev','Null_Count'], axis=1, inplace=False)

				# insert current date into the column
				result['Date'] = str(date.today())

				# insert current time into the column
				result['Time'] = str(datetime.now().time())

				# insert backlog id into column
				result['ProcessID'] = ProcessID

				# insert Data Stage
				result['Data_Stage'] = 'Processed'

				# insert mode column as NULL (For now)
				result['Mode'] = ''

				# insert region column
				test = self.df_list[i][j].Region.unique()
				#print(test[0])
				result['Region'] = test[0]

				# insert sku column
				test1 = self.df_list[i][j].m_sku_desc.unique()
				result['Sku'] =test1[0]

				# Name of the file we want to save profile to
				file_name = "Profile_Table.csv"

				# Exporting the file
				# Check the availability of the file
				if os.path.exists(file_name):

					# don't include headers if the file already exists
					result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a', header=False)
				else:

					# include headers if the file does not exist
					result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a')


				j += 1
			i += 1


	def error_update(self, func, err):
		# This function takes strings as inputs and generates those strings as error warning

		print('Error found in function ' + func + ':')
		print(err)

	def train_test_val(self, log):

		# Run loop through the data
		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:

				# Check if number of columns in train and test data sets are same
				if self.train_df_list[i][j].shape[1] != self.test_df_list[i][j].shape[1]:

					# #print Error, and pass function name
					self.error_update('train_test_val','number of columns for training and testing set donot match')

					# Stop the program
					#print('Stoping Script')
					sys.exit()


				# Check if Region in train and test data sets are same
				if self.train_df_list[i][j].Region.unique() != self.test_df_list[i][j].Region.unique():

					# #print Error, and pass function name
					self.error_update('train_test_val','Regions in test and training data sets do not match')

					# Stop the program
					#print('Stoping Script')
					sys.exit()

				# Check if Region in train and test data sets are same
				if self.train_df_list[i][j].m_sku_desc.unique() != self.test_df_list[i][j].m_sku_desc.unique():

					# print Error, and pass function name
					self.error_update('train_test_val','SKUs in test and training data sets do not match')

					# Stop the program
					#print('Stoping Script')
					sys.exit()


				j += 1
			i += 1


		# Get the last row number
		last_row_number = log.shape[0]

		# Get the last row
		last_row = log.loc[[last_row_number - 1]]

		# Extract forecast period in months
		fore_period = last_row.iloc[0]['Forecast_Period']

		# Run loop through the data
		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:
				# Check if the number of rows are according to forecast period
				if self.test_df_list[i][j].shape[0] > ((fore_period*4)+2) or self.test_df_list[i][j].shape[0] < ((fore_period*4)-2):

					# Print warnings
					print('WARNING: Number of rows not correct: ')
					print('Number of rows found: ')
					print(self.test_df_list[i][j].shape[0])

				j += 1
			i += 1


###############################################################################################################

	def Preprocessing(self,data,log,SKU_col, SKU_list,Region_col,region_list,Year,Training_End,Prediction_Year,Month,Prediction_Month,Threshold, reduced_Var):
		#self.profile_raw_data(data,log,Threshold)
		start_time = time.time()

		self.Data_Prep(data,SKU_col, SKU_list,Region_col,region_list,Threshold)
		self.Save_data()
		df = self.Raw_Data_Cleaning(data,Threshold)
		self.raw_data_prep(data)
		self.raw_data_profile(log,reduced_Var)

		#self.profile_processed_data(df,log)
		self.train_test_split(SKU_list,region_list,Year,Training_End,Prediction_Year,Month,Prediction_Month)
		self.save_train_test_data()
		self.processed_data_profile(log, reduced_Var)
		self.train_test_val(log)
		self.Ex_time = time.time() - start_time
		self.Error_log(data,log,Threshold)
		self.Log_Table(log)

##############################################################################################################

					####### Modeling Functionalities #######

################# This Function Reads the Parameters from metadata
################# which we will be passing to algorithms for training

	def read_model_parameters(self,hyp_set):

		# stores parameters metadata in df
		df=pd.DataFrame(data=hyp_set)

		# Storing Model parameter in separate list for each algo
		self.Arima_Params=[x for x in df.loc[0, df.columns[2:20]].dropna()]
		self.Lasso_Params=[x for x in df.loc[1, df.columns[2:20]].dropna()]
		self.RF_Params=[x for x in df.loc[2, df.columns[2:20]].dropna()]
		self.GB_Params=[x for x in df.loc[3, df.columns[2:20]].dropna()]
		self.Prophet_Params=[x for x in df.loc[4, df.columns[2:20]].dropna()]

		return self

################# This Function Reads the algorithms for this process
################# from process metadata

	def read_model_backlog(self,log):

		# process metadata is equal to log

		# reading models/algos from log and saving in self.algos

		self.algos=log['Models']
		for x in self.algos:
			self.algos=x.split(',')
		print(self.algos)
		return self

################# This Function defines the Arima Model with parameters
################# read from parameters metadata

	def Arima_models(self, df):

		# df contains training data

		# training model
		df = df[['START_DATE', 'tons']]
		df['START_DATE'] = pd.to_datetime(df['START_DATE'])
		df = df.set_index('START_DATE')
		df=df.fillna(0)
		A=auto_arima(df['tons'],seasonal=self.Arima_Params[0])
		x=A.order
		model=ARIMA(df['tons'],order=x)
		results=model.fit()

		# model returned after training
		return results

################# This Function defines the Lasso Model with parameters
################# read from parameters metadata

	def Lasso_Models(self, X, Y):
		# df contains training data

		lasso = Lasso(max_iter=int(self.Lasso_Params[0]),tol=float(self.Lasso_Params[1]))
		model = lasso.fit(X,Y)
		# model returned after training

		return model

################# This Function defines the Gradient Boosting Model with parameters
################# read from parameters metadata
		# df contains training data

	def GBOOST_Models(self, X, Y):
		# df contains training data

		xgb = GradientBoostingRegressor(n_estimators=int(self.GB_Params[0]))
		model = xgb.fit(X,Y)

		# model returned after training
		return model

################# This Function defines the Random Forest Model with parameters
################# read from parameters metadata

	def RF_Models(self,X,Y):
		# df contains training data

		rfr = RandomForestRegressor(n_estimators = int(self.RF_Params[0]))
		model = rfr.fit(X,Y)

		# model returned after training
		return model

################# This Function defines the Prophet Model with parameters
################# read from parameters metadata

	def Prophet_Models(self,df):
		# df contains training data

		df = df[['START_DATE', 'tons']]
		df['START_DATE'] = pd.to_datetime(df['START_DATE'])
		df = df.set_index('START_DATE')
		df = df.resample('W').mean() #make df daily
		df = df.reset_index()
		df.columns = ['ds', 'y']
		df=df.reset_index()
		df=df.drop('index', axis=1)
		fbp = Prophet(daily_seasonality=bool(self.Prophet_Params[0]),yearly_seasonality=int(self.Prophet_Params[1]),weekly_seasonality=self.Prophet_Params[2])
		model = fbp.fit(df)
		# model returned after training

		return model

################# This Function trains all the models for each algo mentioned in
################# process metadata for each Region and SKU

	def fit_models(self,log,hyp_set):

		# Reading selected features from metada
		variable_Table = pd.read_csv('Variable_Table.csv')
		self.reduced_Var = variable_Table[(variable_Table.Influencer_Cat == "Media") | (variable_Table.Influencer_Cat == "Price") | (variable_Table.Influencer_Cat == "Incentive") ]['Variables'].tolist()

		# calling functions to read algos and their parameters from metadata
		self.read_model_backlog(log)
		self.read_model_parameters(hyp_set)

		# loading training datasets
		with open("Prepared_Train_DataSets", 'rb') as t:
			self.train_df_list = pickle.load(t)


		# calc number of SKU's and regions to iterate throgh each
		self.rcount = len(self.train_df_list)
		self.skucount = len(self.train_df_list[0])


		# loop through algos in metadata
		for x in self.algos:

			# if algo is Lasso
			if (x=='Lasso'):

				# array to store lasso models for each region and sku
				Lasso_M = [ [ None for i in range(self.skucount) ] for j in range(self.rcount) ]

				# loop through each sku and region to train and save lasso model
				i = 0
				while i < 1:##self.rcount:
					j = 0
					while j < 3:#self.skucount:
						df = self.train_df_list[i][j]
						X = df[self.reduced_Var]
						Y = df['tons']

						# calling lasso training func which return a trained model
						Lasso_M[i][j] = self.Lasso_Models( X, Y )
						j += 1
					i+=1

				#Saving Models into Local Machine
				predictions = Lasso_M[0][0].predict(X)
				with open("Lasso",'wb') as t:
					pickle.dump(Lasso_M, t)

			# if algo is Arima
			elif(x=='ARIMA'):

				# array to store ARIMA models for each region and sku
				Arima_M = [ [ None for i in range(self.skucount) ] for j in range(self.rcount) ]

				# loop through each sku and region to train and save ARIMA model
				i = 0
				while i < 1:##self.rcount:
					j = 0
					while j < 3:#self.skucount:
						df = self.train_df_list[i][j]

						# calling lasso training func which return a trained model
						Arima_M[i][j] = self.Arima_models(df)
						j += 1
					i+=1

				#Saving Models into Local Machine
				with open("Arima",'wb') as t:
					pickle.dump(Arima_M, t)

			# if algo is RandomForest
			elif(x=='RandomForest'):

				# array to store RandomForest models for each region and sku
				RandomForest_M =[ [ None for i in range(self.skucount) ] for j in range(self.rcount) ]

				# loop through each sku and region to train and save RandomForest model
				i = 0
				while i < 1:##self.rcount:
					j = 0
					while j < 3:#self.skucount:
						df = self.train_df_list[i][j]
						X = df[self.reduced_Var]
						Y = df['tons']

						# calling RandomForest training func which return a trained model
						RandomForest_M[i][j] = self.RF_Models(X , Y)
						j += 1
					i+=1

				#Saving Models into Local Machine
				with open("RandomForest",'wb') as t:
					pickle.dump(RandomForest_M, t)

			# if algo is GradientBoosting
			elif(x=='GradientBoosting'):

				# array to store GradientBoosting models for each region and sku
				GBOOST_M = [ [ None for i in range(self.skucount) ] for j in range(self.rcount) ]

				# loop through each sku and region to train and save GradientBoosting model
				i = 0
				while i < 1:##self.rcount:
					j = 0
					while j < 3:#self.skucount:
						df = self.train_df_list[i][j]
						X = df[self.reduced_Var]
						Y = df['tons']

						# calling GradientBoosting training func which return a trained model
						GBOOST_M[i][j] = self.GBOOST_Models(X, Y)
						j += 1
					i+=1

				#Saving Models into Local Machine
				with open("GBOOST",'wb') as t:
					pickle.dump(GBOOST_M, t)

			# if algo is Prophet
			elif(x=='Prophet'):

				# array to store Prophet models for each region and sku
				Prophet_M = [ [ None for i in range(self.skucount) ] for j in range(self.rcount) ]

				# loop through each sku and region to train and save Prophet model
				i = 0
				while i < 1:##self.rcount:
					j = 0
					while j < 3:#self.skucount:
						df = self.train_df_list[i][j]

						# calling Prophet training func which return a trained model
						Prophet_M[i][j] = self.Prophet_Models(df)
						j += 1
					i+=1

				#Saving Models into Local Machine
				with open("Prophet",'wb') as t:
					pickle.dump(Prophet_M, t)

################# This Function calculated MAPE
#################

	def mape(self, df):

		#df is dataframe having actual and predictions

		dff = df
		a = dff['Actual_Tons']
		f = dff['Predicted_Tons']
		MAPE = np.mean(abs((a-f)/a))
		return MAPE

################# This Function calculated Weighted MAPE
#################

	def wmape(self, df):

		#df is dataframe having actual and predictions

		dff = df#[(df.YEAR_c == 2018) & (df.month_c >9)]
		a = dff['Actual_Tons']
		f = dff['Predicted_Tons']
		WMAPE = np.sum(abs(a-f))/np.sum(a)
		return WMAPE

################# This Function calculates Error
#################

	def error(self, a, f):
		error = a - f
		return error


################# This Function Reads X Features of Scoring data
################# and make prediction using Arima Models

	def Arima_P(self,df, i , j):

	#Load Models
		with open("Arima", 'rb') as t:
				model = pickle.load(t)
	#Predict
		df=pd.DataFrame(df['START_DATE'])
		df = df[['START_DATE']]
		df['START_DATE'] = pd.to_datetime(df['START_DATE'])
		df = df.set_index('START_DATE')
		df=df.sort_values(by=['START_DATE'],axis=0)
		predictions=model[i][j].predict(start=1,end=df.shape[0]).rename('ARIMA predictions')

		#save predictions in list
		predictions = predictions.tolist()
		return predictions

################# This Function Reads X Features of Scoring data
################# and make prediction using Lasso Models

	def Lasso_P(self,X, i, j):

		#Load Models
		with open("Lasso", 'rb') as t:
			model = pickle.load(t)

		#predict
		predictions = model[i][j].predict(X)
		return predictions

################# This Function Reads X Features of Scoring data
################# and make prediction using Gradient Boosting Models

	def GBOOST_P(self,X, i, j):

		#Load Models
		with open("GBOOST", 'rb') as t:
			model = pickle.load(t)

		#predict
		predictions = model[i][j].predict(X)
		return predictions

################# This Function Reads X Features of Scoring data
################# and make prediction using Random Forest Models

	def RandomForest_P(self,X, i, j):

		#Load Models
		with open("RandomForest", 'rb') as t:
			model = pickle.load(t)

		#predict
		predictions = model[i][j].predict(X)
		return predictions

################# This Function Reads X Features of Scoring data
################# and make prediction using Prophet Models

	def Prophet_P(self,df, i, j):

		# getting X variables ready

		df = pd.DataFrame(df['START_DATE'])
		df = df[['START_DATE']]
		df['START_DATE'] = pd.to_datetime(df['START_DATE'])
		df = df.set_index('START_DATE')
		df=df.sort_values(by=['START_DATE'],axis=0)
		df=df.reset_index()
		df.columns = ['ds']

		# Loading Models
		with open("Prophet", 'rb') as t:
			model = pickle.load(t)

		#Predict
		predictions = model[i][j].predict(df)
		return predictions


################# This Function moves the created models in
################# the models Repository

	def model_repo(self,log):

		# columns required for Model ID
		repo_cols = ['Region','m_sku_desc','Algorithm','Model_Index']

		# self.Resultant is dataframe having info about all models

		# adding info to repo
		repo = self.Resultant
		repo = repo[repo_cols]
		repo = repo.drop_duplicates()
		date = log.Date.unique()
		repo['Creation_Date'] = date[0]
		time = log.StartingTime.unique()
		repo['Creaction_Time'] = time[0]
		tp = log.TrainPeriod.unique()
		repo['Train_Period'] = tp[0]
		fp = log.ForecastPeriod.unique()
		repo['Forecast_Period'] = fp[0]

		#writing models repo in file
		repo.to_csv('Models_Repo.csv',index = False)

################# This Function saves the performance
################# of each model against its model id

	def evaluate_model(self,log):

		# columns required for Model ID
		comp_cols = ['Region','m_sku_desc','Algorithm','Model_Index']

		# self.Resultant is dataframe having info about all models

		# Moving models to completed metadata
		completed = self.Resultant
		completed = completed[comp_cols]
		completed = completed.drop_duplicates()
		pid = log.ProcessID.unique()

		# adding process id as info
		completed['ProcessID'] = pid[0]
		completed.to_csv('Completed_Models.csv',index = False)

		# Moving models to active metadata based on their good performance
		active_cols = ['Region','m_sku_desc','Algorithm','Model_Index','MAPE']
		active = self.Resultant
		active = active[active_cols]
		active = active.drop_duplicates()

		# adding process id as info
		active['ProcessID'] = pid[0]
		active = active[(active['MAPE']<0.4)]
		active.drop('MAPE', axis=1, inplace=True)
		active.to_csv('Active_Models.csv',index = False)


################# This Function do the testing predictions
################# on all the trained models

	def test_prediction(self,log):

		# load test data
		with open("Prepared_Test_DataSets", 'rb') as t:
			self.test_df_list = pickle.load(t)

		output_columns = ['Region','m_sku_desc']

		# reading algos in from process backlog
		self.read_model_backlog(log)

		# looping through each region and sku
		i = 0
		while i < 1:#self.rcount:
			j = 0
			while j < 3:#self.skucount:

				model_id1 = str(i)
				model_id2 = str(j)
				modelid = model_id1+','+model_id2

				# Fetch data of a specific region and sku
				df = self.test_df_list[i][j]
				x = df[self.reduced_Var]
				y = df['tons']

				# looping through algos in metadata
				for alg in self.algos:

					#predicting via ARIMA if its in metadata and saving resultset
					if alg=='ARIMA':

						#predicting VIA Arima
						Arima_p = self.Arima_P(df,i,j)

						# Saving Predictions and accuracies in Dataframe

						#preparing actual and predicted
						Predicted_A = pd.DataFrame({'Actual_Tons': df['tons'], 'Predicted_Tons': Arima_p})
						Arima_r=pd.concat([df[output_columns],Predicted_A],axis=1,sort=False)

						# adding Name column of used algo
						Arima_r ['Algorithm']='Arima'

						#Adding accuracy columns in dataframe
						Arima_r ['MAPE'] = self.mape(Arima_r)
						Arima_r ['WMAPE'] = self.wmape(Arima_r)
						Arima_r ['Error'] = self.error(Arima_r['Actual_Tons'], Arima_r ['Predicted_Tons'] )
						Arima_r['Model_Index'] = modelid

						if i==0 and j==0:
							Arima_R = Arima_r
						else:
							Arima_R=pd.concat([Arima_R, Arima_r],axis=0,sort=False)

						#Finalising ARIMA resultset
						self.Resultant = Arima_R.copy()


					# Same steps as above but for Lasso
					elif alg=='Lasso':

						Lasso_p = self.Lasso_P(x,i,j)

						Predicted_L = pd.DataFrame({'Actual_Tons': df['tons'], 'Predicted_Tons': Lasso_p})
						Lasso_r=pd.concat([df[output_columns],Predicted_L],axis=1,sort=False)
						Lasso_r['Algorithm']='Lasso'
						Lasso_r ['MAPE'] = self.mape(Lasso_r)
						Lasso_r ['WMAPE'] = self.wmape(Lasso_r)
						Lasso_r['Error'] = self.error(Lasso_r['Actual_Tons'], Lasso_r ['Predicted_Tons'] )
						#Lasso_r['r2_score'] = r2_score(Lasso_r['Actual_Tons'], Lasso_r ['Predicted_Tons'])
						Lasso_r['Model_Index'] = modelid

						if i==0 and j==0:
							Lasso_R=Lasso_r
						else:
							Lasso_R=pd.concat([Lasso_R, Lasso_r],axis=0,sort=False)

						self.Resultant = Lasso_R.copy()

					# Same steps as above but for GradientBoosting
					elif alg=='GradientBoosting':
						GBOOST_p = self.GBOOST_P(x,i,j)

						Predicted_GB = pd.DataFrame({'Actual_Tons': df['tons'], 'Predicted_Tons': GBOOST_p})
						Gboost_r=pd.concat([df[output_columns],Predicted_GB],axis=1,sort=False)
						Gboost_r ['Algorithm']='GradientBoosting'
						Gboost_r ['MAPE'] = self.mape(Gboost_r )
						Gboost_r ['WMAPE'] = self.wmape(Gboost_r )
						Gboost_r ['Error'] = self.error(Gboost_r['Actual_Tons'], Gboost_r ['Predicted_Tons'] )
						#Gboost_r ['r2_score'] = r2_score(Gboost_r['Actual_Tons'],Gboost_r['Predicted_Tons'])
						Gboost_r['Model_Index'] = modelid

						if i==0 and j==0:
							Gboost_R = Gboost_r
						else:
							Gboost_R=pd.concat([Gboost_R, Gboost_r],axis=0,sort=False)

						self.Resultant = Gboost_R.copy()

					# Same steps as above but for RandomForest
					elif (alg=='RandomForest'):
						RandomForest_p = self.RandomForest_P(x, i, j)

						Predict_R=pd.DataFrame({'Actual_Tons': df['tons'],'Predicted_Tons': RandomForest_p})
						RF_r=pd.concat([df[output_columns],Predict_R],axis=1,sort=False)
						RF_r ['Algorithm']='RandomForest'
						RF_r ['MAPE'] = self.mape(RF_r)
						RF_r ['WMAPE'] = self.wmape(RF_r )
						RF_r ['Error'] = self.error(RF_r['Actual_Tons'], RF_r ['Predicted_Tons'] )
						#RF_r ['r2_score'] = r2_score(RF_r['Actual_Tons'], RF_r ['Predicted_Tons'])
						RF_r['Model_Index'] = modelid

						if i==0 and j==0:
							RF_R = RF_r
						else:
							RF_R=pd.concat([RF_R, RF_r],axis=0,sort=False)

						self.Resultant = RF_R.copy()

					# Same steps as above but for Prophet
					elif (alg=='Prophet'):

						Prophet_p = self.Prophet_P(df,i,j)

						p = Prophet_p['yhat'].tolist()
						Predict_P=pd.DataFrame({'Actual_Tons': df['tons'],'Predicted_Tons': p })
						Prophet_r=pd.concat([df[output_columns],Predict_P],axis=1,sort=False)
						Prophet_r ['Algorithm'] = 'Prophet'
						Prophet_r['MAPE'] = self.mape(Prophet_r)
						Prophet_r['WMAPE']=self.wmape(Prophet_r)
						Prophet_r ['Error'] = self.error(Prophet_r['Actual_Tons'], Prophet_r ['Predicted_Tons'] )
						#Prophet_r['r2_score'] = r2_score(Prophet_r['Actual_Tons'], Prophet_r ['Predicted_Tons'])
						Prophet_r['Model_Index'] = modelid

						if i==0 and j==0:
							Prophet_R = Prophet_r
						else:
							Prophet_R=pd.concat([Prophet_R, Prophet_r],axis=0,sort=False)

						self.Resultant = Prophet_R.copy()

				j += 1
			i+=1

		self.Resultant.drop(self.Resultant.index, inplace=True)


		#Joining dataframes of all algos
		for x in self.algos:
			if (x == 'ARIMA'):
				self.Resultant=pd.concat([self.Resultant,Arima_R], axis=0, ignore_index=True)

			elif (x == 'Lasso'):
				self.Resultant=pd.concat([self.Resultant,Lasso_R], axis=0, ignore_index=True)

			elif (x == 'GradientBoosting'):
				self.Resultant=pd.concat([self.Resultant,Gboost_R], axis=0, ignore_index=True)

			elif (x == 'RandomForest'):
				self.Resultant=pd.concat([self.Resultant,RF_R], axis=0, ignore_index=True)

			elif (x == 'Prophet'):
				self.Resultant=pd.concat([self.Resultant,Prophet_R], axis=0, ignore_index=True)

		# adding process id
		process = log['ProcessID'].unique()
		self.Resultant['ProcessID'] = process[0]

		# writing final resultset in Model Performance Metadata
		self.Resultant.to_csv("ModelPerformance.csv",index = False)
		return self


##############################################################################################################

					####### Scoring Functionalities #######


################# This Function loads only active models make predictions
################# and reve the resultsets in DB

	def load_activemodels(self,log):

		#load active models
		act_models = pd.read_csv('Active_Models.csv')
		algos = act_models['Algorithm'].tolist()
		algosunique = act_models.Algorithm.unique()

		#extract index of that model and its dataset
		index = act_models['Model_Index'].tolist()
		i=0
		arima_count = 0
		lasso_count=0
		gb_count=0
		RF_count=0
		Pro_count=0

		# set output columns required
		output_columns = ['YEAR_c','WEEK_c','Region','m_sku_desc']

		# iterate through active algos
		for alg in algos:

			#get index to fetch this model and its respective dataset from dataset and model arrays
			modelid = index[i]
			inds = modelid.split(",")

			#extracting dataset
			df = self.test_df_list[int(inds[0])][int(inds[1])]
			x = df[self.reduced_Var]
			y = df['tons']

			# if active algi is ARIMA
			if alg=="Arima":

				# Making predictions
				Arima_p = self.Arima_P(df,int(inds[0]),int(inds[1]))

				#preparing actual and predicted
				Predicted_A = pd.DataFrame({'Actual_Tons': df['tons'], 'Predicted_Tons': Arima_p})
				Arima_r=pd.concat([df[output_columns],Predicted_A],axis=1,sort=False)

				# adding algo name
				Arima_r ['Algorithm']='Arima'

				#adding columns for accuracy measurements
				Arima_r ['MAPE'] = self.mape(Arima_r)
				Arima_r ['WMAPE'] = self.wmape(Arima_r)
				Arima_r ['Error'] = self.error(Arima_r['Actual_Tons'], Arima_r ['Predicted_Tons'] )

				#adding model id in resultset
				Arima_r['Model_Index'] = modelid

				#joinint results for all ARIMA models
				if arima_count==0:
					Arima_R = Arima_r
					arima_count=arima_count+1
				else:
					Arima_R=pd.concat([Arima_R, Arima_r],axis=0,sort=False)

				#saving final ARIMA Resultant
				self.Resultant = Arima_R.copy()

			# Same steps as above but for Lasso
			if alg=='Lasso':
				Lasso_p = self.Lasso_P(x,int(inds[0]),int(inds[1]))

				Predicted_L = pd.DataFrame({'Actual_Tons': df['tons'], 'Predicted_Tons': Lasso_p})
				Lasso_r=pd.concat([df[output_columns],Predicted_L],axis=1,sort=False)
				Lasso_r['Algorithm']='Lasso'
				Lasso_r ['MAPE'] = self.mape(Lasso_r)
				Lasso_r ['WMAPE'] = self.wmape(Lasso_r)
				Lasso_r['Error'] = self.error(Lasso_r['Actual_Tons'], Lasso_r ['Predicted_Tons'] )
				#Lasso_r['r2_score'] = r2_score(Lasso_r['Actual_Tons'], Lasso_r ['Predicted_Tons'])
				Lasso_r['Model_Index'] = modelid

				if lasso_count==0:
					Lasso_R=Lasso_r
					lasso_count=lasso_count+1
				else:
					Lasso_R=pd.concat([Lasso_R, Lasso_r],axis=0,sort=False)

				self.Resultant = Lasso_R.copy()

			# Same steps as above but for Gradient Boosting
			if alg=='GradientBoosting':
				GBOOST_p = self.GBOOST_P(x,int(inds[0]),int(inds[1]))

				Predicted_GB = pd.DataFrame({'Actual_Tons': df['tons'], 'Predicted_Tons': GBOOST_p})
				Gboost_r=pd.concat([df[output_columns],Predicted_GB],axis=1,sort=False)
				Gboost_r ['Algorithm']='GradientBoosting'
				Gboost_r ['MAPE'] = self.mape(Gboost_r )
				Gboost_r ['WMAPE'] = self.wmape(Gboost_r )
				Gboost_r ['Error'] = self.error(Gboost_r['Actual_Tons'], Gboost_r ['Predicted_Tons'] )
				#Gboost_r ['r2_score'] = r2_score(Gboost_r['Actual_Tons'],Gboost_r['Predicted_Tons'])
				Gboost_r['Model_Index'] = modelid

				if gb_count==0:
					Gboost_R = Gboost_r
					gb_count=gb_count+1
				else:
					Gboost_R=pd.concat([Gboost_R, Gboost_r],axis=0,sort=False)

				self.Resultant = Gboost_R.copy()

			# Same steps as above but for Random Forest
			if alg=='RandomForest':
				RandomForest_p = self.RandomForest_P(x, int(inds[0]),int(inds[1]))

				Predict_R=pd.DataFrame({'Actual_Tons': df['tons'],'Predicted_Tons': RandomForest_p})
				RF_r=pd.concat([df[output_columns],Predict_R],axis=1,sort=False)
				RF_r ['Algorithm']='RandomForest'
				RF_r ['MAPE'] = self.mape(RF_r)
				RF_r ['WMAPE'] = self.wmape(RF_r )
				RF_r ['Error'] = self.error(RF_r['Actual_Tons'], RF_r ['Predicted_Tons'] )
				#RF_r ['r2_score'] = r2_score(RF_r['Actual_Tons'], RF_r ['Predicted_Tons'])
				RF_r['Model_Index'] = modelid

				if RF_count==0:
					RF_R = RF_r
					RF_count=RF_count+1
				else:
					RF_R=pd.concat([RF_R, RF_r],axis=0,sort=False)

				self.Resultant = RF_R.copy()

			# Same steps as above but for Prophet
			if alg=='Prophet':
				Prophet_p = self.Prophet_P(df,int(inds[0]),int(inds[1]))

				p = Prophet_p['yhat'].tolist()
				Predict_P=pd.DataFrame({'Actual_Tons': df['tons'],'Predicted_Tons': p })
				Prophet_r=pd.concat([df[output_columns],Predict_P],axis=1,sort=False)
				Prophet_r ['Algorithm'] = 'Prophet'
				Prophet_r['MAPE'] = self.mape(Prophet_r)
				Prophet_r['WMAPE']=self.wmape(Prophet_r)
				Prophet_r ['Error'] = self.error(Prophet_r['Actual_Tons'], Prophet_r ['Predicted_Tons'] )
				#Prophet_r['r2_score'] = r2_score(Prophet_r['Actual_Tons'], Prophet_r ['Predicted_Tons'])
				Prophet_r['Model_Index'] = modelid

				if Pro_count==0:
					Prophet_R = Prophet_r
					Pro_count=Pro_count+1
				else:
					Prophet_R=pd.concat([Prophet_R, Prophet_r],axis=0,sort=False)

				self.Resultant = Prophet_R.copy()
			i=i+1
		self.Resultant.drop(self.Resultant.index, inplace=True)

		#joining resultsets of all algos
		for alg in algosunique:
			if (alg == 'Arima'):
				self.Resultant=pd.concat([self.Resultant,Arima_R], axis=0, ignore_index=True)

			elif (alg == 'Lasso'):
				self.Resultant=pd.concat([self.Resultant,Lasso_R], axis=0, ignore_index=True)

			elif (alg == 'GradientBoosting'):
				self.Resultant=pd.concat([self.Resultant,Gboost_R], axis=0, ignore_index=True)

			elif (alg == 'RandomForest'):
				self.Resultant=pd.concat([self.Resultant,RF_R], axis=0, ignore_index=True)

			elif (alg == 'Prophet'):
				self.Resultant=pd.concat([self.Resultant,Prophet_R], axis=0, ignore_index=True)

		process = log['ProcessID'].unique()

		# Adding process id
		self.Resultant['ProcessID'] = process[0]

		#saving future predictions in DB
		score_performance = self.Resultant.drop(['YEAR_c','WEEK_c'],axis=1)
		score_performance.to_csv("ScoringModelPerformance.csv",index = False)

		return self

################# This Function ranks the models as per their performance
################# in each sku and region and selects champion model in that sku and region



	def rank_performance(self,log):



		# columns needed in output
		comp_cols = ['Region','m_sku_desc','Algorithm','Model_Index','MAPE']

		# self.Resultant has the results of predictions and accuracies
		data = self.Resultant[comp_cols]



		# extracting unique models with their MAPE
		data = data.drop_duplicates()



		# Extracting unique regions and skus
		region_list = data.Region.unique()
		SKU_list = data.m_sku_desc.unique()



		#extracting process ID
		process = log['ProcessID'].unique()




		# creating a list to separate the results of each region,sku
		frames = [None for i in range(len(region_list)*len(SKU_list))]



		#creating an empty dataset to save final result with ranks
		rank = data.copy()
		rank.drop(rank.index, inplace=True)



		i = 0



		# Looping through each region,sku
		for reg in region_list:
			for sku in SKU_list:



				# saving results of models one region,sku in frames[i]
				frames[i] = data[(data['Region'] == reg) & (data['m_sku_desc'] == sku)]
				# sorting all the models according to MAPE
				frames[i] = frames[i].sort_values(by=['MAPE'], ascending=True)
				# Assigning them ranks from top to bottom
				frames[i]['Rank'] = range(1,frames[i].shape[0]+1,1)
				# creating an empty champion column
				frames[i]['Champion'] = None



				#Assigning champion to the model in first row of this resultset
				frames[i].reset_index(inplace=True)
				frames[i]['Champion'][0] = 'Champion'
				#removing useless column
				frames[i].drop('index',axis = 1, inplace = True)
				#joining results of this region,sku with final resultset
				rank = pd.concat([rank,frames[i]], axis=0, ignore_index=True)



				i = i+1
		#assigning processID
		rank['ProcessID'] = process[0]



		# changing string rank to int
		rank = rank.astype({"Rank":int})



		#joining the Rank resultset with predictions resultset
		resultset = pd.merge(self.Resultant, rank, on=['Algorithm','Model_Index','ProcessID','Region','m_sku_desc','MAPE'], how='left')



		# Saving final results in DB
		resultset.to_csv('Resultant.csv',index = False)

################# This Function Reads the log as an argument from metadata
################# Also, it saves the profile of variables in scoring data

	def score_data_profile(self, log):

		# Open test datasets file
		with open("Prepared_Test_DataSets", 'rb') as t:
			self.test_df_list = pickle.load(t)


		# Save region counts
		self.rcount = len(self.test_df_list)

		# Save skus count
		self.skucount = len(self.test_df_list[0])

		# Get the last row number
		last_row_number = log.shape[0]



		# Get the last row
		last_row = log.loc[[last_row_number - 1]]



		# Extract Backlog Id
		ProcessID = last_row.iloc[0]['ProcessID']



		# Importing Data From File
		data = pd.read_csv("Variable_Table.csv")

		# Import variable table in dataframe
		variable_Table = pd.read_csv('Variable_Table.csv')

		# Select reduced variables
		self.reduced_Var = variable_Table[(variable_Table.Influencer_Cat == "Media") | (variable_Table.Influencer_Cat == "Price") | (variable_Table.Influencer_Cat == "Incentive") ]['Variables'].tolist()


		# Iterate through the datframes
		i = 0
		while i < self.rcount:
			j = 0
			while j < self.skucount:


				# Select reduced variables only
				df = self.test_df_list[i][j][self.reduced_Var]



				# save data profiles in variables
				prof_median = df.median(numeric_only=bool)
				prof_unique = df.nunique(dropna=True)
				prof_mean = df.mean(numeric_only=bool)
				prof_mode = df.mode(numeric_only=bool)
				prof_min = df.min(numeric_only=bool)
				prof_max = df.max(numeric_only=bool)
				prof_quant_25 = df.quantile(numeric_only=bool,q=0.25)
				prof_quant_75 = df.quantile(numeric_only=bool,q=0.75)
				prof_std = df.std(numeric_only=bool)
				prof_nan_count = len(df) - df.count()




				# create a dataframe to be exported
				result = pd.concat([prof_median,prof_unique, prof_mean, prof_min, prof_max, prof_quant_25, prof_quant_75,prof_std,prof_nan_count], axis=1)



				# reset index
				result.reset_index(level = 0, inplace =True)



				# rename columns
				result = result.set_axis(['VariableName','Median','Unique_values','Mean','Minimum','Maximum', 'Quantile_25', 'Quantile_75', 'Std_Dev','Null_Count'], axis=1, inplace=False)



				# insert current date into the column
				result['Date'] = str(date.today())



				# insert current time into the column
				result['Time'] = str(datetime.now().time())



				# insert backlog id into column
				result['ProcessID'] = ProcessID



				# insert Data Stage
				result['Data_Stage'] = 'Scoring'



				# insert mode column as NULL (For now)
				result['Mode'] = ''



				# insert region column
				test = self.test_df_list[i][j].Region.unique()
				print(test[0])
				result['Region'] = test[0]



				# insert sku column
				test1 = self.test_df_list[i][j].m_sku_desc.unique()
				result['Sku'] =test1[0]



				# Name of the file we want to write to
				file_name = "Profile_Table.csv"



				# Exporting the file
				# Check the availability of the file
				if os.path.exists(file_name):



					# don't include headers if the file already exists
					result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a', header=False)
				else:



					# include headers if the file does not exist
					result.to_csv(file_name, encoding='utf-8',index=False, mode = 'a')




				j += 1
			i += 1
