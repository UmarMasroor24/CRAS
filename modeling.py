import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
import config







from pipeline import Pipeline

pipeline = Pipeline( )

if __name__ == '__main__':

	# load process table
	log = pd.read_csv(config.Model_backlog)

	# loading dataset
	data = pd.read_csv(config.PATH_TO_DATASET)

	#loading parameters table
	hyp_set = pd.read_csv('ParameterTable.csv')

	############################################

	# Call Training Functions
	pipeline.fit_models(log,hyp_set)

	# Call Testing and Performance Function
	pipeline.test_prediction(log)

	# Call Model Repo Funtion
	pipeline.model_repo(log)

	#Call Evaluate Model Function to classify them in Completed and Active Models
	pipeline.evaluate_model(log)

	print("Success")
