PATH_TO_DATASET = "datads.csv"
path_to_metadata = "metadata v0.2(Temp).xlsx"
variable = 'Variable_Table_um.csv'
SKU = 'SKU.csv'
Region = 'Region.csv'
Model_backlog = 'Processes_Table.csv'
Processes_Table = "Processes_Table.csv"

#Features to be used for Modeling
reduced_Var = [ 'media_spend_value','min_con_price_per_gram', 'avg_con_price_per_gram',
'max_con_price_per_gram', 'media_lag_1', 'media_lag_2', 
'incentive_applied_pop_count_lagdiff_1_2', 'media_lag_3', 'media_lag_4', 
'media_lagdiff_1_2', 'media_lagdiff_2_3', 'media_lagdiff_3_4', 'price_lag_1', 
'price_lag_2', 'price_lag_3', 'price_lag_4', 'price_lagdiff_1_2', 'price_lagdiff_2_3', 
'price_lagdiff_3_4', 'incentive_applied_pop_count', 'bonus', 'freeitems', 'unit', 'kg',
'bonus_flag', 'unit_flag', 'kg_flag', 'max_bonus', 'max_freeitems', 'max_unit', 
'max_kg', 'min_bonus', 'min_freeitems', 'min_unit', 'min_kg',  'avg_bonus', 
'avg_freeitems', 'avg_unit', 'avg_kg', 'incentive_applied_pop_count_lag_1', 
'incentive_applied_pop_count_lag_2','incentive_applied_pop_count_lag_3', 
'incentive_applied_pop_count_lag_4','incentive_applied_pop_count_lagdiff_2_3', 
'incentive_applied_pop_count_lagdiff_3_4', 'bonus_lag_1', 'bonus_lag_2',
'bonus_lag_3', 'bonus_lag_4', 'bonus_lagdiff_1_2', 'bonus_lagdiff_2_3', 
'bonus_lagdiff_3_4', 'freeitems_lag_1', 'freeitems_lag_2', 'freeitems_lag_3', 
'freeitems_lag_4', 'freeitems_lagdiff_1_2', 'freeitems_lagdiff_2_3', 
'freeitems_lagdiff_3_4', 'unit_lag_1', 'unit_lag_2', 'unit_lag_3', 'unit_lag_4',
'unit_lagdiff_1_2', 'unit_lagdiff_2_3', 'unit_lagdiff_3_4', 'kg_lag_1', 'kg_lag_2', 
'kg_lag_3', 'kg_lag_4', 'kg_lagdiff_1_2', 'kg_lagdiff_2_3', 'kg_lagdiff_3_4' ]

#Media Variables
Med_var=['media_spend_value','media_lag_1','media_lag_2','media_lag_3','media_lag_4',
'media_lagdiff_1_2','media_lagdiff_2_3','media_lagdiff_3_4']

#Incentives Variable
in_var=['incentive_applied_pop_count', 'bonus', 'freeitems', 'unit', 'kg', 
'bonus_flag', 'unit_flag', 'kg_flag', 'max_bonus', 'max_freeitems', 'max_unit', 
'max_kg', 'min_bonus', 'min_freeitems', 'min_unit', 'min_kg',  'avg_bonus', 
'avg_freeitems', 'avg_unit', 'avg_kg', 'incentive_applied_pop_count_lag_1', 
'incentive_applied_pop_count_lag_2', 'incentive_applied_pop_count_lag_3', 
'incentive_applied_pop_count_lag_4', 'incentive_applied_pop_count_lagdiff_1_2', 
'incentive_applied_pop_count_lagdiff_2_3', 'incentive_applied_pop_count_lagdiff_3_4', 
'bonus_lag_1', 'bonus_lag_2',  'bonus_lag_3', 'bonus_lag_4', 'bonus_lagdiff_1_2', 
'bonus_lagdiff_2_3', 'bonus_lagdiff_3_4', 'freeitems_lag_1', 'freeitems_lag_2', 
'freeitems_lag_3', 'freeitems_lag_4', 'freeitems_lagdiff_1_2', 'freeitems_lagdiff_2_3',
'freeitems_lagdiff_3_4', 'unit_lag_1', 'unit_lag_2', 'unit_lag_3', 
'unit_lag_4', 'unit_lagdiff_1_2', 'unit_lagdiff_2_3', 'unit_lagdiff_3_4', 
'kg_lag_1', 'kg_lag_2', 'kg_lag_3', 'kg_lag_4', 'kg_lagdiff_1_2', 'kg_lagdiff_2_3', 
'kg_lagdiff_3_4'] 

# Price Variables
pric_var=[ 'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4',
'price_lagdiff_1_2', 'price_lagdiff_2_3', 'price_lagdiff_3_4','min_con_price_per_gram', 
'avg_con_price_per_gram', 'max_con_price_per_gram']