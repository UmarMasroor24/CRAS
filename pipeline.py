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
from pathlib import Path

import config
import os

os.system('cmd /c "cls"')

root = Path('.')


class Pipeline:
	def __init__(self):

		self.metadataengine = None
		self.adsengine = None



	#below function does imputation by filling nulls with mean of the column
	def Imputation(self, df,var_table):
		#loop through dataframe and check for nulls in each column
		Threshold = 0
		variables = var_table[(var_table.Variable_Class != 'SKU') & (var_table.Variable_Class != 'Region')]['Variable'].tolist()
		dff = df[[variables]]
		for (columnName, columnData) in dff.iteritems():
			Threshold = var_table.loc[var_table.Variable == columnName]['Null_Threshold'].tolist()[0]
			temp = dff[[columnName]]
			#if nulls is less than the threshold then fill all the nulls of column with mean of the column
			if temp.isnull().sum() < Threshold:
				temp = fillna(temp.mean())
				df[columnName] = temp
		#return the dataframe after imputation
		return df
	
	#remove nulls	
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
	def Erroneous_Column(self,data,VariableInfo):

		variables = VariableInfo['variables']
		variables = list(set(variables))

		df = data.copy()

		#make a list of erroneous columns
		Err_col = []

		Threshold = 0.5
		#Iterate through the dataframe and calculate percentage null points in each columns
		for (columnName, columnData) in df.iteritems():
				if columnName in variables:
						Threshold = VariableInfo['var_table'].loc[VariableInfo['var_table'].Variable == columnName]['Null_Threshold'].tolist()[0]
				else:
						Threshold = 0.5
				temp = df[[columnName]]

				Per_Nulls = temp.isnull().sum()[0]/temp.shape[0]
				#if the percentage of null points is greater than the threshold mark it as erroneous column
				if Per_Nulls > Threshold:
						Err_col.append(columnName)

		#return the erroneous columns if exist
		return Err_col

	#This function removes the erroneous_column and returns dataframe after removal of the erroneous column
	def Remove_Err_column(self,df, col):
		#drop the erroneous column and return dataframe
		df = df.drop(col,axis = 1)
		return df

	#This function checks for empty dataset and returns true if dataset is empty
	def Empty_Dataset(self, df):
		#return true if the dataset is empty
		return len(df.index) == 0

	#'PackageName','UserStory','ErrorType','ExecutionTime','ProcessID'
	
	def Write_Log(self,statusCode,pid):

		os.makedirs('./Logs', exist_ok = True)

		s_code = pd.read_csv('StatusCode.csv')

		s_code = s_code.loc[s_code.StatusCode == statusCode]

		s_code = s_code.reset_index()

		status = s_code['Class'][0]

		print(status)

		if status == "Error":

			file_name  =  "./Logs/Error_log.csv"
			check = 0
			#check if error_log table exists
			if os.path.exists(file_name):
					check = 1
			#if error_log table is missing then make error_log table
			else:
					temp = pd.DataFrame(columns = ['PackageName','UserStory','ErrorType','ExecutionTime','ProcessID'])

			if check == 1:
					#load error_table
					temp = pd.read_csv('./Logs/Error_log.csv')

			#user story from status code metadata
			UserStory = s_code.Userstory[0]

			#package name from metadata
			PackageName = s_code.Module[0]

			ErrorType = s_code.Description[0]


			temp = temp.append({'PackageName':PackageName,'UserStory':UserStory,'ErrorType':ErrorType,'ExecutionTime':self.exe_time,'ProcessID':pid},ignore_index=True)

			temp.to_csv('./Logs/Error_log.csv', index = False)

		else:
			#name of the log_table
			file_name = "./Logs/Log_Table.csv"

			#check if log_table exists
			check = 0
			if os.path.exists(file_name):
				check = 1
				pass
			else:
				temp = pd.DataFrame(columns =['LogID','Date','PackageName','UserStory','StatusCode',
																				'StartingTime','ExecutionTime','ProcessID',
																				'Status','Description'])
				#if the log_table does't exist assign logid 1 when creating log_table
				logid = 1
			if check == 1:
				#if path exists read the log_table
				temp = pd.read_csv('./Logs/Log_Table.csv')
				logid = temp.iloc[temp.shape[0]-1]['LogID']+1

			dat = str(date.today())#current date

			now = datetime.now()

			st_time = now.strftime("%H:%M:%S")#str(datetime.datetime.now().time())#current time
			#read statusCode metadata

			#user story from status code metadata
			UserStory = s_code.Userstory[0]

			#package name from metadata
			PackageName = s_code.Module[0]

			#status from metadata
			status = s_code.Class[0]

			#Description from metadata
			Description = s_code.Description[0]


			#append the log table with the updated log values
			temp = temp.append({'LogID':logid,'Date':dat,'PackageName':PackageName,'UserStory':UserStory,
													'StatusCode':statusCode,'StartingTime':st_time,'ExecutionTime':self.exe_time,
													'ProcessID':pid,
													'Status': status,'Description':Description},ignore_index=True)
			#write it back
			temp.to_csv('./Logs/Log_Table.csv',index = False)







#############################################################################
	#This function prepares the raw data without breaking it into subsets
	def Data_Prep(self, data,pid,VariableInfo):
		start_time = time.time()

		check = 0
		try:
		
			if self.Empty_Dataset(data):

				statusCode = '1.2.1.1'
				check = 1

			if check == 0:
				#check for erroneous columns
				column = self.Erroneous_Column(data,VariableInfo)
				check = 0
				if len(column) != 0:
					#remove erroneous columns
					data = self.Remove_Err_column(data,column)
					check = 0
					statusCode = '1.2.1.2'
			#remove nulls once removed erroneous columns
			data = self.Remove_Null(data)
			statusCode = '1.2.3'
			self.exe_time = time.time() - start_time
			self.Write_Log(statusCode,pid)
			# self.Log_Table(statusCode)

			#return Dataset
			return data

		except:
			statusCode = '1.2.2'
			self.exe_time = time.time() - start_time
			print("dataprep")
			self.Write_Log(statusCode,pid)
			return data

###############################################

	#prepare training and testing datasets
	def train_test_split(self,pid):
			
		try:
			start_time = time.time()

			ProcessInfo = self.read_training_backlog(pid)
			VariableInfo = self.read_variable_list(ProcessInfo)
			Region = self.read_demography(ProcessInfo)

			self.maintain_inprogress_process(ProcessInfo['ProcessID'])

			start_time = time.time()

			df = pd.read_csv('datads.csv')
			#print(self.date)

			df[VariableInfo['date']] = pd.to_datetime(df[VariableInfo['date']],format = "%m/%d/%Y",errors='coerce')

			train_data = df.loc[(df[VariableInfo['SKU_Col']] == str(ProcessInfo['SKU'])) & (df[VariableInfo['Region_Col']] == str(Region))]
			train_data = train_data.loc[(train_data[VariableInfo['date']] >= ProcessInfo['Training_StartDate']) & (train_data[VariableInfo['date']] <= ProcessInfo['Training_EndDate'])]
			raw_train_data = train_data.copy()
			train_data = self.Data_Prep(train_data,ProcessInfo['ProcessID'],VariableInfo)

			test_data = df.loc[(df[VariableInfo['SKU_Col']] == ProcessInfo['SKU']) & (df[VariableInfo['Region_Col']] == Region) & (df[VariableInfo['date']] >= ProcessInfo['Testing_StartDate']) & (df[VariableInfo['date']] <= ProcessInfo['Testing_EndDate'])]
			test_data = self.Data_Prep(test_data,ProcessInfo['ProcessID'],VariableInfo)

			os.makedirs('./Datasets', exist_ok = True)
			# tr_dataset = "PID_"+str(ProcessInfo['ProcessID'])+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_Train"+".csv"

			rw_tr_data = "PID_"+str(ProcessInfo['ProcessID'])+"_raw_Train"+".csv"
			rw_tr_data = "./Datasets/"+rw_tr_data

			tr_dataset = "PID_"+str(ProcessInfo['ProcessID'])+"_Train"+".csv"
			tr_dataset = "./Datasets/"+tr_dataset

			# ts_dataset = "PID_"+str(ProcessInfo['ProcessID'])+"_"+self.SKU+"_"+self.region+"_"+self.algorithm+"_Test"+".csv"
			ts_dataset = "PID_"+str(ProcessInfo['ProcessID'])+"_Test"+".csv"
			ts_dataset = "./Datasets/"+ts_dataset


			raw_train_data.to_csv(rw_tr_data)
			train_data.to_csv(tr_dataset)
			test_data.to_csv(ts_dataset)

			check = 1
			statusCode = '1.1.3'
			self.exe_time = time.time() - start_time
			self.Write_Log(statusCode,ProcessInfo['ProcessID'])
		except:
			statusCode = '1.1.2'
			self.exe_time = time.time() - start_time
			self.Write_Log(statusCode,ProcessInfo['ProcessID'])
			print("There is an error please see the Error log")

		return self

###########################################
	def forecast_split(self,pid):
			
		ProcessInfo = self.read_forecasting_backlog(pid)
		VariableInfo = self.read_variable_list(ProcessInfo)
		Region = self.read_demography(ProcessInfo)

		# try:
		start_time = time.time()

		df = pd.read_csv('datads.csv')

		df[VariableInfo['date']] = pd.to_datetime(df[VariableInfo['date']],format = "%m/%d/%Y",errors='coerce')

		forecast_data = df.loc[(df[VariableInfo['SKU_Col']] == ProcessInfo['SKU']) & (df[VariableInfo['Region_Col']] == Region) & (df[VariableInfo['date']] >= ProcessInfo['Forecasting_StartDate']) & (df[VariableInfo['date']] <= ProcessInfo['Forecasting_EndDate'])]

		forecast_data = self.Data_Prep(forecast_data,ProcessInfo['ProcessID'],VariableInfo)

		forecast_dataset = "PID_"+str(ProcessInfo['ProcessID'])+"_Forecast"+".csv"
		forecast_dataset = "./Datasets/"+forecast_dataset

		forecast_data.to_csv(forecast_dataset)

		# check = 1
		# statusCode = '1.1.3'
		# self.exe_time = time.time() - start_time
		# self.Write_Log(statusCode)
			# self.Log_Table(statusCode)
		# except:
		# 	statusCode = '1.1.2'
		# 	self.exe_time = time.time() - start_time
		# 	self.Write_Log(statusCode)
		# 	print("There is an error please see the error log")

		return forecast_data

	def error_update(self, func, err):
		# This function takes strings as inputs and generates those strings as error warning

		print('Error found in function ' + func + ':')
		print(err)

	def train_test_val(self,pid):

		df_train = self.read_traindata(pid)
		df_test = self.read_testdata(pid)

		# Check if number of columns in train and test data sets are same
		if df_train.shape[1] != df_test.shape[1]:

			# #print Error, and pass function name
			self.error_update('train_test_val','number of columns for training and testing set donot match')

			# Stop the program
			sys.exit()


		# Check if Region in train and test data sets are same
		if df_train.Region.unique() != df_test.Region.unique():

			# #print Error, and pass function name
			self.error_update('train_test_val','Regions in test and training data sets do not match')

			# Stop the program
			sys.exit()

		# Check if SKU in train and test data sets are same
		if df_train.m_sku_desc.unique() != df_test.m_sku_desc.unique():

			# print Error, and pass function name
			self.error_update('train_test_val','SKUs in test and training data sets do not match')

			# Stop the program
			sys.exit()

		print("train_test_val complete")


		# Number of rows in test dataframe

		# # Get the last row number
		# last_row_number = log.shape[0]

		# # Get the last row
		# last_row = log.loc[[last_row_number - 1]]

		# # Extract forecast period in months
		# fore_period = last_row.iloc[0]['Forecast_Period']

		# # Run loop through the data
		# i = 0
		# while i < self.rcount:
		# 	j = 0
		# 	while j < self.skucount:
		# 		# Check if the number of rows are according to forecast period
		# 		if self.test_df_list[i][j].shape[0] > ((fore_period*4)+2) or self.test_df_list[i][j].shape[0] < ((fore_period*4)-2):

		# 			# Print warnings
		# 			print('WARNING: Number of rows not correct: ')
		# 			print('Number of rows found: ')
		# 			print(self.test_df_list[i][j].shape[0])

		# 		j += 1
		# 	i += 1





	def data_profile(self, pid, Data_Stage):

		os.makedirs('./profiles', exist_ok = True)

		if Data_Stage == 'Raw':
			df = self.read_raw_train_data(pid)
			ProcessInfo = self.read_training_backlog(pid)
		elif Data_Stage == 'Processed':
			df = self.read_traindata(pid)
			ProcessInfo = self.read_training_backlog(pid)
		elif Data_Stage == 'Scoring':
			df = self.read_forecastdata(pid)
			ProcessInfo = self.read_forecasting_backlog(pid)

		
		VariableInfo = self.read_variable_list(ProcessInfo)


		# Separate Numerical and Categorical variables
		variables_numerical = [*VariableInfo['X_variables_numerical'],*VariableInfo['targetvariable']]
		variables_categorical = VariableInfo['X_variables_categorical']

		# Select the columns of only the variables we need
		df_numerical = df[variables_numerical]
		df_categorical = df[variables_categorical]


		# save data profiles for numerical variables in variables
		prof_median = df_numerical.median(numeric_only=bool)
		prof_unique = df_numerical.nunique(dropna=True)
		prof_mean = df_numerical.mean(numeric_only=bool)
		prof_min = df_numerical.min(numeric_only=bool)
		prof_max = df_numerical.max(numeric_only=bool)
		prof_quant_25 = df_numerical.quantile(numeric_only=bool,q=0.25)
		prof_quant_75 = df_numerical.quantile(numeric_only=bool,q=0.75)
		prof_std = df_numerical.std(numeric_only=bool)
		prof_nan_count = len(df_numerical) - df_numerical.count()

		# create a resultant dataframe for numerical variables
		result_numerical = pd.concat([prof_median,prof_unique, prof_mean, prof_min, prof_max, prof_quant_25, prof_quant_75,prof_std,prof_nan_count], axis=1)
		
		# reset index
		result_numerical.reset_index(level = 0, inplace =True)
		
		#rename columns
		result_numerical = result_numerical.set_axis(['VariableName','Median','Unique_values','Mean','Minimum','Maximum', 'Quantile_25', 'Quantile_75', 'Std_Dev','Null_Count'], axis=1, inplace=False)

		# set to zero, the profiles which are not needed for numerical variables
		result_numerical.loc[:,'prof_quant_25'] = 0
		result_numerical.loc[:,'prof_quant_75'] = 0
		result_numerical.loc[:,'prof_std'] = 0


		# Check if categorical variables exist
		if VariableInfo['X_variables_categorical'].shape[0] != 0:

			# save data profiles for categorical variables in variables
			prof_median_1 = df_categorical.median(numeric_only=bool)
			prof_unique_1 = df_categorical.nunique(dropna=True)
			prof_mean_1 = df_categorical.mean(numeric_only=bool)
			prof_min_1 = df_categorical.min(numeric_only=bool)
			prof_max_1 = df_categorical.max(numeric_only=bool)
			prof_quant_25_1 = df_categorical.quantile(numeric_only=bool,q=0.25)
			prof_quant_75_1 = df_categorical.quantile(numeric_only=bool,q=0.75)
			prof_std_1 = df_categorical.std(numeric_only=bool)
			prof_nan_count_1 = len(df_categorical) - df_categorical.count()



			# create a resultant dataframe for categorical variables
			result_categorical = pd.concat([prof_median_1,prof_unique_1, prof_mean_1, prof_min_1, prof_max_1, prof_quant_25_1, prof_quant_75_1,prof_std_1,prof_nan_count_1], axis=1)
			

			# reset index
			result_categorical.reset_index(level = 0, inplace =True)
			

			# rename columns
			result_categorical = result_categorical.set_axis(['VariableName','Median','Unique_values','Mean','Minimum','Maximum', 'Quantile_25', 'Quantile_75', 'Std_Dev','Null_Count'], axis=1, inplace=False)


			# set to zero, the profiles which are not needed for numerical variables
			result_categorical.loc[:,'prof_quant_25'] = 0
			result_categorical.loc[:,'prof_quant_75'] = 0
			result_categorical.loc[:,'prof_std'] = 0

		# Check if categoical variables exist
		if VariableInfo['X_variables_categorical'].shape[0] == 0:

			# Profile of numerical variables will be our resultant
			# If there are no categorical vairables
			result = result_numerical
		else:

			# Make a final dataframe to be exported
			result = pd.concat([result_numerical, result_categorical], axis = 0, sort = False)

		# insert current date into the column
		result['Date'] = str(date.today())

		# insert current time into the column
		result['Time'] = str(datetime.now().time())

		# insert backlog id into column
		result['ProcessID'] = ProcessInfo['ProcessID']

		# insert Data Stage
		result['Data_Stage'] = Data_Stage

		# insert region and sku columns
		result['Demography_ID'] = ProcessInfo['Demography_ID']
		result['SKU'] = ProcessInfo['SKU']



		# name the output file
		nameofprofile = "PID_"+str(ProcessInfo['ProcessID'])+"_Profiles" + ".csv"
		nameofprofile = "./profiles/" + nameofprofile

		# Exporting the file
		# Check the availability of the file
		if os.path.exists(nameofprofile):

			# don't include headers if the file already exists
			result.to_csv(nameofprofile, encoding='utf-8',index=False, mode = 'a', header=False)
		else:

			# include headers if the file does not exist
			result.to_csv(nameofprofile, encoding='utf-8',index=False, mode = 'a')


###############################################################################################################


###############################################################################################################
	#Preprocessing wraper
	def Preprocessing_orchestrator(self):

		self.connect_to_MetadataDB()
		Process = pd.read_sql_query('SELECT * FROM dbo.Training_Processes_Table',self.metadataengine)
		# Create a directory 'profiles' if doesnt exist already
		#os.makedirs('./profiles', exist_ok = True)

		

		totalrows = Process.shape[0]

		Pid = 0

		while Pid < totalrows:
			

			self.train_test_split(Pid)
			self.data_profile(Pid,  "Raw")
			self.data_profile(Pid, "Processed")
			self.train_test_val(Pid)


			Pid = Pid + 1

		# print("train_test_val start")

		# train_df = self.read_processed_train_data()
		# test_df = self.read_testdata()

		# Ex_time = time.time() - start_time


##############################################################################################################

					####### Modeling Functionalities #######

################# This is the orchestration function of modeling
################# which call all the modeling functionalities in order

	def modeling_orchestrator(self):
		# process metadata is same as log
		os.makedirs('./Modeling_Output', exist_ok = True)

		# reading models/algos from log and saving in self.algos
		Process = pd.read_csv(config.T_process_Table)
		Process.reset_index(inplace = True)
		totalrows = Process.shape[0]

		pid=0

		while pid<totalrows:


			self.fit_models(pid)

			result = self.modeling_performance(pid,'test')

			self.modeling_performance(pid,'train')

			self.model_repo(result,pid)

			self.evaluate_model(result,pid)

			pid=pid+1

		train_results = pd.read_csv('./Modeling_Output/Train_Performance.csv')
		test_results = pd.read_csv('./Modeling_Output/Test_Performance.csv')

		self.rank_performance(train_results,'./Modeling_Output/Train_Performance')
		self.rank_performance(test_results,'./Modeling_Output/Test_Performance')


################# This Function Reads the Parameters from metadata
################# which we will be passing to algorithms for training

	def read_model_parameters(self,ProcessInfo):

		# stores parameters metadata in parameter_table
		parameter_table = pd.read_sql_query('SELECT * FROM dbo.ParameterTable',self.metadataengine)
		# parameter_table = pd.read_csv(config.parameter_table)

		#read parameters row for Parameter ID of this process
		parameter_table = parameter_table[(parameter_table['ParameterID'] == ProcessInfo['Parameter_id'])]
		parameter_table.reset_index(inplace = True)
		
		# Storing Model parameter in separate list for this process
		model_params=[x for x in parameter_table.loc[0, parameter_table.columns[3:20]].dropna()]
		return model_params

	def read_training_backlog(self,pid):

		#load prosses_table
		Process = pd.read_sql_query('SELECT * FROM dbo.Training_Processes_Table',self.metadataengine)
		# Process = pd.read_csv(config.T_process_Table)

		#Read desired values from each row

		processID = Process['ProcessID'][pid]
		algorithm = Process['Models'][pid]
		SKU = Process['SKU'][pid]
		Demography_ID = Process['Demography_ID'][pid]

		variable_list_id = Process['VariableListID'][pid]
		parameter_id = Process['ParameterID'][pid]
		performancethres_id = Process['ThresID'][pid]

		Training_StartDate = pd.to_datetime(Process['Training_StartDate'][pid],format = "%m/%d/%Y",errors='coerce')
		Training_EndDate = pd.to_datetime(Process['Training_EndDate'][pid],format = "%m/%d/%Y",errors='coerce')
		Testing_StartDate = pd.to_datetime(Process['Testing_StartDate'][pid],format = "%m/%d/%Y",errors='coerce')
		Testing_EndDate = pd.to_datetime(Process['Testing_EndDate'][pid],format = "%m/%d/%Y",errors='coerce')

		ProcessInfo =	{
		  "ProcessID": processID,
		  "Algorithm": algorithm,
		  "SKU": SKU,
		  "Demography_ID": Demography_ID,
		  "variable_list_id": variable_list_id,
		  "Parameter_id": parameter_id,
		  "performancethres_id": performancethres_id,
		  "Training_StartDate": Training_StartDate,
		  "Training_EndDate": Training_EndDate,
		  "Testing_StartDate": Testing_StartDate,
		  "Testing_EndDate": Testing_EndDate
		}
		# Calling functions to read variables and parameters for this process and algorithm

		
		# self.read_model_parameters(ProcessInfo)
		# self.read_performance_thres(ProcessInfo)
		return ProcessInfo


	def read_forecasting_backlog(self,pid):

		active_models = pd.read_csv(config.Active_Models)

		Process = pd.read_sql_query('SELECT * FROM dbo.Forecasting_Processes_Table',self.metadataengine)
		# Process = pd.read_csv(config.F_process_Table)

		Process = pd.merge(active_models, Process, how='inner', on=['ProcessID','Models','VariableListID','ParameterID','ThresID','SKU','Demography_ID'])
		Process.drop(['MAPE'], axis=1,inplace = True)

		#Read desired values from each row

		ProcessID = Process['ProcessID'][pid]
		algorithm = Process['Models'][pid]
		SKU = Process['SKU'][pid]
		Demography_ID = Process['Demography_ID'][pid]
		variable_list_id = Process['VariableListID'][pid]
		parameter_id = Process['ParameterID'][pid]
		performancethres_id = Process['ThresID'][pid]

		Forecasting_StartDate = pd.to_datetime(Process['Forecasting_StartDate'][pid],format = "%m/%d/%Y",errors='coerce')
		Forecasting_EndDate = pd.to_datetime(Process['Forecasting_EndDate'][pid],format = "%m/%d/%Y",errors='coerce')

		ProcessInfo =	{
		  "ProcessID": ProcessID,
		  "Algorithm": algorithm,
		  "SKU": SKU,
		  "Demography_ID": Demography_ID,
		  "variable_list_id": variable_list_id,
		  "Parameter_id": parameter_id,
		  "performancethres_id": performancethres_id,
		  "Forecasting_StartDate": Forecasting_StartDate,
		  "Forecasting_EndDate": Forecasting_EndDate
		}

		# Calling functions to read variables and parameters for this process and algorithm

		return ProcessInfo

#########################################
	def read_demography(self,ProcessInfo):

		demography = pd.read_sql_query('SELECT * FROM dbo.Demography_Table',self.metadataengine)
		# demography = pd.read_csv('Demography_Table.csv')
		demo = demography.loc[demography['Demography_ID'] == ProcessInfo['Demography_ID']]['Region']
		
		demo = demo.reset_index()

		Region = demo['Region'][0]

		return Region

################# This Function reads the variables 
################# for this process

	def read_variable_list (self,ProcessInfo):

		# read variable table for variablelist ID of this process
		variable_table = pd.read_sql_query('SELECT * FROM dbo.Variable_Table',self.metadataengine)
		# variable_table = pd.read_csv(config.variable_table)

		var_table = variable_table

		#retrieve SKU and Region names
		SKU_col = variable_table[(variable_table['Variable_Class'] == 'SKU')]['Variable'].values[0]
		Region_Col = variable_table[variable_table['Variable_Class'] == 'Region']['Variable'].values[0]
		
		#retrive timeseries variable
		date = variable_table[variable_table['Variable_Class'] == 'TimeSeries']['Variable'].values[0]

		variable_table = variable_table[(variable_table['VariableListID'] == ProcessInfo['variable_list_id'])]

		# retrieve target variable name
		targetvariable = variable_table[(variable_table['Variable_Class'] == 'Target')]['Variable'].values

		#All variables
		variables = variable_table[(variable_table['Variable_Class'] != 'SKU') & (variable_table['Variable_Class'] != 'Region')]['Variable'].tolist()

		# retrieve names of X variables to use in the model of this process
		X_variables = variable_table[(variable_table['Variable_Class'] != 'Target') & (variable_table['Variable_Class'] != 'SKU') & (variable_table['Variable_Class'] != 'Region')]['Variable'].values
		X_variables_numerical = variable_table[(variable_table['Variable_Type'] == 'Numerical') & (variable_table['Variable_Class'] != 'Target') & (variable_table['Variable_Class'] != 'SKU') & (variable_table['Variable_Class'] != 'Region')]['Variable'].values
		X_variables_categorical = variable_table[(variable_table['Variable_Type'] == 'Categorical') & (variable_table['Variable_Class'] != 'Target') & (variable_table['Variable_Class'] != 'SKU') & (variable_table['Variable_Class'] != 'Region')]['Variable'].values

		VariableInfo =	{
		  "var_table": var_table,
		  "SKU_Col": SKU_col,
		  "Region_Col": Region_Col,
		  "date": date,
		  "targetvariable": targetvariable,
		  "variables": variables,
		  "X_variables": X_variables,
		  "X_variables_numerical": X_variables_numerical,
		  "X_variables_categorical": X_variables_categorical,
		}

		return VariableInfo

################# This Function reads the Thresholds 
################# of performance for models of this process

	def read_performance_thres(self,ProcessInfo):

		# read Performance Thres table for Thres ID of this process
		threstable = pd.read_sql_query('SELECT * FROM dbo.PerformanceThres',self.metadataengine)

		# threstable = pd.read_csv(config.PerformanceThres_Table)
		threstable = threstable[(threstable['ThresID'] == ProcessInfo['performancethres_id'])]

		# retrieve active models thres value
		active_thres = threstable['Active'].values
		active_thres = active_thres[0]

		# retrieve errored models thres value
		errored_thres = threstable['Errored'].values
		errored_thres = errored_thres[0]

		Performance_ThresInfo =	{
		  "active_thres": active_thres,
		  "errored_thres": errored_thres,
		}

		return Performance_ThresInfo


################# This Function defines the Arima Model with parameters
################# read from parameters metadata

	def Arima_models(self, df,model_params,VariableInfo,pid):

		# df contains training data



		#setting variables required by this algo in a list
		variables = [*VariableInfo['X_variables'],*VariableInfo['targetvariable']]
		
		# training model
		df = df[variables]


		df[VariableInfo['X_variables'][0]] = pd.to_datetime(df[VariableInfo['X_variables'][0]])
		df = df.set_index(VariableInfo['X_variables'][0])
		df=df.fillna(0)
		A=auto_arima(df[VariableInfo['targetvariable'][0]],seasonal=model_params[0])
		x=A.order
		model=ARIMA(df[VariableInfo['targetvariable'][0]],order=x)
		results=model.fit()


		#creating name of model
		nameofmodel = "PID_"+str(pid)+"_Model"
		nameofmodel = "./models/"+nameofmodel


		#saving model
		with open(nameofmodel,'wb') as t:
			pickle.dump(results, t)

		

################# This Function defines the Lasso Model with parameters
################# read from parameters metadata

	def Lasso_Models(self, df,model_params,VariableInfo,pid):
		# df contains training data
		X = df[VariableInfo['X_variables']]
		Y = df[VariableInfo['targetvariable']]

		lasso = Lasso(max_iter=int(model_params[0]),tol=float(model_params[1]))
		model = lasso.fit(X,Y)

		#Saving Models into Local Machine
		nameofmodel = "PID_"+str(pid)+"_Model"
		nameofmodel = "./models/"+nameofmodel

		with open(nameofmodel,'wb') as t:
			pickle.dump(model, t)

################# This Function defines the Gradient Boosting Model with parameters
################# read from parameters metadata
		# df contains training data

	def GBOOST_Models(self, df,model_params,VariableInfo,pid):
		# df contains training data

		X = df[VariableInfo['X_variables']]
		Y = df[VariableInfo['targetvariable']]

		xgb = GradientBoostingRegressor(n_estimators=int(model_params[0]))
		model = xgb.fit(X,Y)

		#Saving Models into Local Machine
		nameofmodel = "PID_"+str(pid)+"_Model"
		nameofmodel = "./models/"+nameofmodel

		with open(nameofmodel,'wb') as t:
			pickle.dump(model, t)
		

################# This Function defines the Random Forest Model with parameters
################# read from parameters metadata

	def RF_Models(self,df,model_params,VariableInfo,pid):
		# df contains training data

		X = df[VariableInfo['X_variables']]
		Y = df[VariableInfo['targetvariable']]

		rfr = RandomForestRegressor(n_estimators = int(model_params[0]))
		model = rfr.fit(X,Y)

		nameofmodel = "PID_"+str(pid)+"_Model"
		nameofmodel = "./models/"+nameofmodel

		with open(nameofmodel,'wb') as t:
			pickle.dump(model, t)
		# model returned after training

################# This Function defines the Prophet Model with parameters
################# read from parameters metadata

	def Prophet_Models(self,df,model_params,VariableInfo,pid):
		# df contains training data

		variables = [*VariableInfo['X_variables'],*VariableInfo['targetvariable']]

		df = df[variables]
		df[VariableInfo['X_variables'][0]] = pd.to_datetime(df[VariableInfo['X_variables'][0]])
		df = df.set_index(VariableInfo['X_variables'][0])
		df = df.resample('W').mean() #make df daily
		df = df.reset_index()
		df.columns = ['ds', 'y']
		df=df.reset_index()
		df=df.drop('index', axis=1)
		fbp = Prophet(daily_seasonality=bool(model_params[0]),yearly_seasonality=int(model_params[1]),weekly_seasonality=model_params[2])
		model = fbp.fit(df)
		# model returned after training

		nameofmodel = "PID_"+str(pid)+"_Model"
		nameofmodel = "./models/"+nameofmodel

		with open(nameofmodel,'wb') as t:
			pickle.dump(model, t)

################# This Function trains all the models for each algo mentioned in
################# process metadata for each Region and SKU

	def fit_models(self,pid):

		ProcessInfo = self.read_training_backlog(pid)
		model_params = self.read_model_parameters(ProcessInfo)
		VariableInfo = self.read_variable_list(ProcessInfo)

		nameofdataset = "PID_"+str(ProcessInfo['ProcessID'])+"_Train.csv"
		nameofdataset = "./Datasets/"+nameofdataset

		# loading training datasets
		# with open("Prepared_Train_DataSets", 'rb') as t:
		# 	self.train_df = pickle.load(t)

		train_df = pd.read_csv(nameofdataset)

		os.makedirs('./models', exist_ok = True)
		# if algo is Lasso
		if (ProcessInfo['Algorithm']=='Lasso'):

			self.Lasso_Models(train_df,model_params,VariableInfo,ProcessInfo['ProcessID'])			

			# if algo is Arima
		elif(ProcessInfo['Algorithm']=='ARIMA'):

			self.Arima_models(train_df,model_params,VariableInfo,ProcessInfo['ProcessID'])

			# if algo is RandomForest
		elif(ProcessInfo['Algorithm']=='RandomForest'):

			self.RF_Models(train_df,model_params,VariableInfo,ProcessInfo['ProcessID'])
			

		# if algo is GradientBoosting
		elif(ProcessInfo['Algorithm']=='GradientBoosting'):

			self.GBOOST_Models(train_df,model_params,VariableInfo,ProcessInfo['ProcessID'])
			

		# if algo is Prophet
		elif(ProcessInfo['Algorithm']=='Prophet'):

			self.Prophet_Models(train_df,model_params,VariableInfo,ProcessInfo['ProcessID'])

		return train_df

################# This Function calculates MAPE
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

	def error(self,a,f):
		error = a - f
		return error



################# This Function moves the created models in
################# the models Repository
	def read_model(self,pid,status):

		if status == 'modeling':
			ProcessInfo = self.read_training_backlog(pid)
		else:
			ProcessInfo = self.read_forecasting_backlog(pid)
		
		nameofmodel = "PID_"+str(ProcessInfo['ProcessID'])+"_Model"
		nameofmodel = "./models/"+nameofmodel


		with open(nameofmodel, 'rb') as t:
			model = pickle.load(t)

		return model


	def read_raw_train_data (self,pid):

		ProcessInfo = self.read_training_backlog(pid)

		nameofdataset = "PID_"+str(ProcessInfo['ProcessID'])+"_raw_Train.csv"
		nameofdataset = "./Datasets/"+nameofdataset

		train_df = pd.read_csv(nameofdataset)

		return train_df

	def read_traindata (self,pid):
		
		ProcessInfo = self.read_training_backlog(pid)

		nameofdataset = "PID_"+str(ProcessInfo['ProcessID'])+"_Train.csv"
		nameofdataset = "./Datasets/"+nameofdataset

		train_df = pd.read_csv(nameofdataset)

		return train_df


	def read_testdata (self,pid):
		
		ProcessInfo = self.read_training_backlog(pid)

		nameofdataset = "PID_"+str(ProcessInfo['ProcessID'])+"_Test.csv"
		nameofdataset = "./Datasets/"+nameofdataset

		test_df = pd.read_csv(nameofdataset)

		return test_df

	def read_forecastdata (self,pid):
		
		ProcessInfo = self.read_forecasting_backlog(pid)

		nameofdataset = "PID_"+str(ProcessInfo['ProcessID'])+"_Forecast.csv"
		nameofdataset = "./Datasets/"+nameofdataset

		forecast_df = pd.read_csv(nameofdataset)

		return forecast_df

	def prediction_func(self, df, pid, status):

		model = self.read_model(pid,status)
		if status == 'modeling':
			ProcessInfo = self.read_training_backlog(pid)
		else:
			ProcessInfo = self.read_forecasting_backlog(pid)

		VariableInfo = self.read_variable_list(ProcessInfo)
		
		#test_df_list = read_testdata()

		if (ProcessInfo['Algorithm'] == 'Prophet' or ProcessInfo['Algorithm'] == 'ARIMA'):
			# df = df[0][0]
			df = pd.DataFrame(df[VariableInfo['X_variables'][0]])
			df[VariableInfo['X_variables'][0]] = pd.to_datetime(df[VariableInfo['X_variables'][0]])
			df = df.set_index(VariableInfo['X_variables'][0])
			df=df.sort_values(by=[VariableInfo['X_variables'][0]],axis=0)
			df=df.reset_index()
			df.columns = ['ds']

			if (ProcessInfo['Algorithm'] == 'ARIMA'):
				predictions=model.predict(start=1,end=df.shape[0]).rename('ARIMA predictions')
				predictions = predictions.tolist()

			else:
				predictions = model.predict(df)
				predictions = predictions['yhat'].tolist()
		else:
			# df = df[0][0]
			x = df[VariableInfo['X_variables']]
			y = df[VariableInfo['targetvariable']]

			predictions = model.predict(x)

		return predictions


	def modeling_performance(self,pid,status):

		if status == 'test':
			df = self.read_testdata(pid)
		else:
			df = self.read_traindata(pid)
		
		ProcessInfo = self.read_training_backlog(pid)
		VariableInfo = self.read_variable_list(ProcessInfo)

		df.reset_index(inplace = True)
		predictions = self.prediction_func(df,pid,'modeling')

		output_columns = ['START_DATE','Region','m_sku_desc']


		targets = pd.DataFrame({'Actual_Tons': df[VariableInfo['targetvariable'][0]], 'Predicted_Tons': predictions})
		result=pd.concat([df[output_columns],targets],axis=1,sort=False)

		# adding process ID in resultset
		result['ProcessID'] = ProcessInfo['ProcessID']

		# adding Name column of used algo
		result ['Algorithm'] = ProcessInfo['Algorithm']
		
		#Adding accuracy columns in dataframe
		result ['MAPE'] = self.mape(result)
		result ['WMAPE'] = self.wmape(result)
		result ['Error'] = self.error( result['Actual_Tons'], result ['Predicted_Tons'] )

		if (status == 'test'):
			with open('./Modeling_Output/Test_Performance.csv', 'a') as f:
				result.to_csv(f, index = False,  mode='a', header=f.tell()==0, line_terminator='\n')

		else:
			with open('./Modeling_Output/Train_Performance.csv', 'a') as f:
				result.to_csv(f, index = False, mode='a', header=f.tell()==0, line_terminator='\n')

		return result
		

################# This Function moves the created models in
################# the models Repository

	def model_repo(self,result,pid):

		# columns required for Model ID
		repo_cols = ['m_sku_desc','Algorithm','ProcessID']

		ProcessInfo = self.read_training_backlog(pid)

		# self.Resultant is dataframe having info about all models

		# adding info to repo
		repo = result
		repo = repo[repo_cols]
		repo = repo.drop_duplicates()

		today = date.today()
		d1 = today.strftime("%d/%m/%Y")
		repo['Creation_Date'] = d1

		t = time.localtime()
		current_time = time.strftime("%H:%M:%S", t)
		repo['Creaction_Time'] = current_time

		repo['VariableListID'] = ProcessInfo['variable_list_id']
		repo['ParameterID'] = ProcessInfo['Parameter_id']
		repo['ThresID'] = ProcessInfo['performancethres_id']
		repo['Demography_ID'] = ProcessInfo['Demography_ID']

		#writing models repo in file
		with open('./Modeling_Output/Models_Repo.csv', 'a') as f:
				repo.to_csv(f, index = False,  mode='a', header=f.tell()==0, line_terminator='\n')

################# This Function saves the performance
################# of each model against its model id

	def evaluate_model(self,result,pid):

		# columns required for Model ID
		comp_cols = ['m_sku_desc','Algorithm','ProcessID','MAPE']

		ProcessInfo = self.read_training_backlog(pid)
		Performance_ThresInfo = self.read_performance_thres(ProcessInfo)

		# self.Resultant is dataframe having info about all models

		# Moving models to completed metadata
		completed = result[comp_cols]
		completed = completed.drop_duplicates()

		completed['VariableListID'] = ProcessInfo['variable_list_id']
		completed['ParameterID'] = ProcessInfo['Parameter_id']
		completed['ThresID'] = ProcessInfo['performancethres_id']
		completed['Demography_ID'] = ProcessInfo['Demography_ID']

		completed.rename(columns={'m_sku_desc': 'SKU', 'Algorithm': 'Models'}, inplace=True)

		with open('./Modeling_Output/Completed_Models.csv', 'a') as f:
				completed.to_csv(f, index = False,  mode='a', header=f.tell()==0, line_terminator='\n')

		errored = completed[(completed['MAPE']>Performance_ThresInfo['errored_thres'])]
		with open('./Modeling_Output/Errored_Models.csv', 'a') as f:
				errored.to_csv(f, index = False,  mode='a', header=f.tell()==0, line_terminator='\n')

		active = completed[(completed['MAPE']<Performance_ThresInfo['active_thres'])]
		with open('./Modeling_Output/Active_Models.csv', 'a') as f:
				active.to_csv(f, index = False,  mode='a', header=f.tell()==0, line_terminator='\n')





##############################################################################################################

					####### Scoring Functionalities #######

################# This is the orchestration function of Scoring
################# which call all the Scoring functionalities in order

	def scoring_orchestrator(self):


		os.makedirs('./Scoring_Output', exist_ok = True)
		# reading models/algos from log and saving in self.algos
		process = pd.read_csv(config.Active_Models)
		process.reset_index(inplace = True)
		totalrows = process.shape[0]


		pid=0

		while pid<totalrows:


			forecastdata = self.forecast_split(pid)

			predictions = self.prediction_func(forecastdata,pid,'scoring')

			self.save_predictions(forecastdata,predictions,pid)
			self.data_profile(pid, 'Scoring')

			self.maintain_completed_process(pid)

			pid=pid+1


################# This Function calculates the performance 
################# for each model

	def evaluation_orchestrator(self):
		process = pd.read_csv(config.Active_Models)
		var_table = pd.read_csv(config.variable_table)
		process.reset_index(inplace = True)
		totalrows = process.shape[0]

		pid=0

		while pid<totalrows:

			df = self.scoring_performance(pid)

			pid=pid+1

		self.rank_performance(df,'./Scoring_Output/Resultant')




################# This Function saves the predictions 
################# for each model

	def save_predictions(self,df,predictions,pid):

		ProcessInfo = self.read_forecasting_backlog(pid)

		df.reset_index(inplace = True)

		output_columns = ['START_DATE','Region','m_sku_desc']

		targets = pd.DataFrame({'Predicted_Tons': predictions})

		result=pd.concat([df[output_columns],targets],axis=1,sort=False)

		# adding Name column of used algo
		result ['Algorithm'] = ProcessInfo['Algorithm']

		# adding process ID in resultset
		result['ProcessID'] = ProcessInfo['ProcessID']


		with open('./Scoring_Output/FurutePredictions.csv', 'a') as f:
				result.to_csv(f, index = False, mode='a', header=f.tell()==0, line_terminator='\n')





################# This Function calculates the accuracy of predictions
################# my the models

	def scoring_performance(self,pid):

		ProcessInfo = self.read_forecasting_backlog(pid)
		VariableInfo = self.read_variable_list(ProcessInfo)

		df = self.read_forecastdata(pid)

		predictions = self.prediction_func(df,pid,'scoring')

		output_columns = ['START_DATE','Region','m_sku_desc']


		targets = pd.DataFrame({'Actual_Tons': df[VariableInfo['targetvariable'][0]], 'Predicted_Tons': predictions})
		result=pd.concat([df[output_columns],targets],axis=1,sort=False)

		# adding process ID in resultset
		result['ProcessID'] = ProcessInfo['ProcessID']

		# adding Name column of used algo
		result ['Algorithm'] = ProcessInfo['Algorithm']
		
		# Adding accuracy columns in dataframe
		result ['MAPE'] = self.mape(result)
		result ['WMAPE'] = self.wmape(result)
		result ['Error'] = self.error( result['Actual_Tons'], result ['Predicted_Tons'] )


		if pid==0:
			self.Resultset = result
		else:
			self.Resultset=pd.concat([self.Resultset, result],axis=0,sort=False)

		return self.Resultset


################# This Function ranks the models as per their performance
################# in each sku and region and selects champion model in that sku and region



	def rank_performance(self,df,nameoffile):

		# columns needed in output
		comp_cols = ['Region','m_sku_desc','Algorithm','ProcessID','MAPE']
		Resultset = df
		# self.Resultant has the results of predictions and accuracies
		data = Resultset[comp_cols]

		# extracting unique models with their MAPE
		data = data.drop_duplicates()

		# Extracting unique regions and skus
		region_list = data.Region.unique()
		SKU_list = data.m_sku_desc.unique()

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

		# changing string rank to int
		rank = rank.astype({"Rank":int})

		rank.to_csv(nameoffile+'_ModelLevel.csv',index = False)

		#joining the Rank resultset with predictions resultset
		resultset = pd.merge(Resultset, rank, on=['Algorithm','ProcessID','Region','m_sku_desc','MAPE'], how='left')

		# Saving final results in DB
		resultset.to_csv(nameoffile+'.csv',index = False)


	def maintain_inprogress_process(self,processid):
		os.makedirs('./Process_Track', exist_ok = True)
		df = pd.read_sql_query('SELECT * FROM dbo.Training_Processes_Table',self.metadataengine)
		# df=pd.read_csv(config.T_process_Table)
		x=df.loc[(df['ProcessID']==processid)]
		with open('./Process_Track/In_progress.csv', 'a') as f:
				x.to_csv(f, index = False, mode='a', header=f.tell()==0, line_terminator='\n')
		


	def maintain_completed_process(self,pid):
		df=pd.read_csv('./Process_Track/In_progress.csv')

		ProcessInfo = self.read_forecasting_backlog(pid)

		index_names = df[df['ProcessID'] == ProcessInfo['ProcessID'] ].index 
		df2=pd.DataFrame(df[df['ProcessID'] == ProcessInfo['ProcessID'] ])

		file_name = "./Process_Track/Completed.csv"

		if os.path.exists(file_name):
			df2.to_csv(file_name, encoding='utf-8',index=False, mode = 'a', header=False)
		else:

			df2.to_csv(file_name, encoding='utf-8',index=False, mode = 'a')

		df.drop(index_names, inplace = True)
		df.to_csv('./Process_Track/In_progress.csv',index=False)

	def connect_to_MetadataDB(self):

		import pyodbc 
		import urllib
		import sqlalchemy as sa

		params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=DESKTOP-L7QE7C8\SQLEXPRESS;"
                                 "DATABASE=CRAS Metadata;"
                                 "Trusted_Connection=yes")

		self.metadataengine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
		return self



	def connect_to_ADSDB(self):

		import pyodbc 
		import urllib
		import sqlalchemy as sa

		params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=DESKTOP-L7QE7C8\SQLEXPRESS;"
                                 "DATABASE=CRAS ADS;"
                                 "Trusted_Connection=yes")

		self.adsengine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
		return self


		

		

