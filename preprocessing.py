import pandas as pd
import numpy as np
import time
import pickle
import config

start_time = time.time()
#variabel table
variable_Table = pd.read_csv(config.variable)
#SKU table
SKUs = pd.read_csv(config.SKU)
#Region table
Regions = pd.read_csv(config.Region)
#model backlog
Model_backlog = pd.read_csv(config.Processes_Table)
print("--- %s seconds ---" % (time.time() - start_time))

#variables
variabletable = variable_Table
# Year
Year = variable_Table[variable_Table.Influencer_Cat == 'Year']['Variables'].tolist()[0]
# Month
Month = variable_Table[variable_Table.Influencer_Cat == 'Month']['Variables'].tolist()[0]
# Column containing all skus
SKU_col = variable_Table[variable_Table.Influencer_Cat == 'SKU']['Variables'].tolist()[0]
# Column containing all regions
Region_col = variable_Table[variable_Table.Influencer_Cat == 'Region']['Variables'].tolist()[0]
#var_encode = variable_Table[variable_Table.Instruction=="encode"]['Column_Name'].tolist()[0]

# variable to be used
reduced_Var = variable_Table[(variable_Table.Influencer_Cat == "Media") | (variable_Table.Influencer_Cat == "Price") | (variable_Table.Influencer_Cat == "Incentive") ]['Variables'].tolist()
# Media variables
Med_var = variable_Table[(variable_Table.Influencer_Cat == "Media")]['Variables'].tolist()
# incentives variables
in_var = variable_Table[(variable_Table.Influencer_Cat == "Incentive")]['Variables'].tolist()
# Price variables
pric_var = variable_Table[(variable_Table.Influencer_Cat == "price")]['Variables'].tolist()
#SKUs from SKU table
SKUs = SKUs
# List of SKUs
SKU_list = SKUs[SKUs.Instruction != 'Remove']['SKU'].tolist()
#Regions from region table
Regions = Regions
#List of region
region_list = Regions['Region'].tolist()
#Threshold for null values
Threshold = variable_Table['Threshold'].unique().tolist()[0]
#Model_backlog
Model_backlog = Model_backlog
#Training_Start = Model_backlog.Training_Start.tolist()[0]
#Training period Start
Training_Start = Model_backlog.Training_Start.tolist()[0]
#Training Period end
Training_End = Model_backlog.Training_End.tolist()[0]
#Prediction Year
Prediction_Year  = Model_backlog.Prediction_Year.tolist()[0]
#Prediction month
Prediction_month = Model_backlog.Forecast_Period.tolist()[0]
# Target variable
Target = variable_Table[(variable_Table.Influencer_Cat == "Target")]['Variables'].tolist()[0]
print(Prediction_month)







from pipeline import Pipeline

pipeline = Pipeline()


if __name__ == '__main__':

	print(len(reduced_Var))
	# load data set
	start_time = time.time()
	log = pd.read_csv(config.Processes_Table)
	#load data
	data = pd.read_csv(config.PATH_TO_DATASET)
	#call Preprocessing wraper function
	pipeline.Preprocessing(data,log,SKU_col, SKU_list,Region_col,region_list,Year,Training_End,Prediction_Year,Month,Prediction_month,Threshold, reduced_Var)
	print("--- %s seconds ---" % (time.time() - start_time))
	# pipeline.train_test_split()
	# pipeline.save_train_test_data()
	print("Success")
