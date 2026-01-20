#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 20:57:35 2026

@author: Tianheng Zhu
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
#%%MAE
def position_err_calculation_all(df_dict, output_root, exp_name, exp_id):
    columns = ['Vehicle_ID', 'Global_Time', 'x', 'Local_Y','l','Lane_ID']
    rows = []
    mae_list = []
    mape_list = []
    rmse_list = []
    mae_l_list = []
    
    # Calculate the total number of planning horizons
    num_ph = len(df_dict)
    
    for case_index in range(1, num_ph + 1):
        if case_index not in []:  # Skip the planning horizon with erroneous trajectory data
            # Read the trajectory reconstruction result for the current planning horizon
            opt_result = pd.read_csv(output_root+exp_name+'/'+str(case_index)+'/'+exp_id+'.csv')
            
            # Get the corresponding vehicle data for the case (assumes all_veh_list_int is defined globally)
            all_veh = df_dict[case_index].copy()
            all_veh['Lane_ID'] = np.where(
                all_veh['Lane_ID'] == 11,
                1,
                all_veh['Lane_ID'] + 1
            )
            veh_id_list = all_veh['Vehicle_ID'].drop_duplicates().tolist()
            
            # Create a time sequence rounded to one decimal place
            time_sequence_rp = np.arange(min(all_veh.Global_Time), max(all_veh.Global_Time) + 1, 1).tolist()
            time_sequence_rp = [round(num, 1) for num in time_sequence_rp]
            
            # Replace the integer indices in the prediction results with the actual vehicle IDs and times
            for i in range(len(opt_result)):
                veh_id_temp = opt_result.Vehicle.iloc[i]
                time_id_temp = int(opt_result.Time.iloc[i])
                opt_result.loc[opt_result.index[i], 'Vehicle'] = veh_id_list[veh_id_temp]
                opt_result.loc[opt_result.index[i], 'Time'] = time_sequence_rp[time_id_temp]
            
            # Drop rows with missing predictions (assuming column 'v' holds a value you care about)
            opt_result = opt_result.dropna(subset=['v'])
            
            # Rename columns to prepare for merging
            opt_result.rename(columns={'Vehicle': 'Vehicle_ID', 'Time': 'Global_Time'}, inplace=True)
            
            # Merge the prediction results with the actual vehicle data
            merged_df = pd.merge(opt_result, all_veh, on=['Vehicle_ID', 'Global_Time'])
            
            # Calculate MAE for the current case and store it
            mae_val = mean_absolute_error(merged_df['x'], merged_df['Local_Y'])
            mae_list.append(mae_val)
            
            # Calculate MAPE for the current case.
            # Using the formula: MAPE = mean(|(true - pred) / true|) * 100.
            mape_val = np.mean(np.abs((merged_df['x'] - merged_df['Local_Y']) / merged_df['Local_Y'])) * 100
            mape_list.append(mape_val)
            
            # Calculate RMSE for the current case
            rmse_val = math.sqrt(mean_squared_error(merged_df['x'], merged_df['Local_Y']))
            rmse_list.append(rmse_val)
            
            mae_l = mean_absolute_error(merged_df['l'], merged_df['Lane_ID'])
            mae_l_list.append(mae_l)
            
            # Keep only the desired columns and append to the final dataframe.
            merged_df = merged_df[columns]
            if not merged_df.empty:
                rows.append(merged_df)
    
    final_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    # Calculate overall MAE and overall MAPE across all cases
    overall_mae = mean_absolute_error(final_df['x'], final_df['Local_Y'])
    overall_mape = np.mean(np.abs((final_df['x'] - final_df['Local_Y']) / final_df['Local_Y'])) * 100
    overall_rmse = math.sqrt(mean_squared_error(final_df['x'], final_df['Local_Y']))
    overall_mae_l = mean_absolute_error(final_df['l'], final_df['Lane_ID'])
    
    return overall_mae, overall_mape, overall_rmse, overall_mae_l

def position_err_calculation_undetected(df_dict, output_root, exp_name, exp_id):
    
    columns = ['Vehicle_ID', 'Global_Time', 'x', 'Local_Y','l','Lane_ID']
    rows = []
    mae_list = []
    mape_list = []
    rmse_list = []
    mae_l_list = []
    
    # Calculate the total number of planning horizons
    num_ph = len(df_dict)
    
    for case_index in range(1, num_ph+1):
        if case_index not in []:
            
            opt_result = pd.read_csv(output_root+exp_name+'/'+str(case_index)+'/'+exp_id+'.csv')
            all_veh = df_dict[case_index].copy()
            all_veh['Lane_ID'] = np.where(
                all_veh['Lane_ID'] == 11,
                1,
                all_veh['Lane_ID'] + 1
            )
            
            veh_id_list=all_veh['Vehicle_ID'].drop_duplicates().tolist()
            
            time_sequence_rp = np.arange(min(all_veh.Global_Time), max(all_veh.Global_Time)+1, 1).tolist()
            time_sequence_rp = [round(num, 1) for num in time_sequence_rp]
            for i in range(len(opt_result)):
                veh_id_temp = opt_result.Vehicle.iloc[i]
                time_id_temp = int(opt_result.Time.iloc[i])
                opt_result.loc[opt_result.index[i], 'Vehicle'] = veh_id_list[veh_id_temp]
                opt_result.loc[opt_result.index[i], 'Time'] = time_sequence_rp[time_id_temp]
            opt_result=opt_result.dropna(subset=['v'])
            opt_result.rename(columns={'Vehicle': 'Vehicle_ID'}, inplace=True)
            opt_result.rename(columns={'Time': 'Global_Time'}, inplace=True)
            
            # Merge the prediction results with the actual vehicle data
            merged_df = pd.merge(opt_result, all_veh, on=["Vehicle_ID","Global_Time"])
            merged_df = merged_df[merged_df["Detected"] == 0]
            merged_df = merged_df[columns]
            if not merged_df.empty:
                rows.append(merged_df)
            
            # Calculate MAE for the current case and store it
            mae_val = mean_absolute_error(merged_df['x'], merged_df['Local_Y'])
            mae_list.append(mae_val)
            
            # Calculate MAPE for the current case.
            # Using the formula: MAPE = mean(|(true - pred) / true|) * 100.
            mape_val = np.mean(np.abs((merged_df['x'] - merged_df['Local_Y']) / merged_df['Local_Y'])) * 100
            mape_list.append(mape_val)
            
            # Calculate RMSE for the current case
            rmse_val = math.sqrt(mean_squared_error(merged_df['x'], merged_df['Local_Y']))
            rmse_list.append(rmse_val)
            
            mae_l = mean_absolute_error(merged_df['l'], merged_df['Lane_ID'])
            mae_l_list.append(mae_l)
            
            
    
    final_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    # Calculate overall MAE and overall MAPE across all cases
    overall_mae = mean_absolute_error(final_df['x'], final_df['Local_Y'])
    overall_mape = np.mean(np.abs((final_df['x'] - final_df['Local_Y']) / final_df['Local_Y'])) * 100
    overall_rmse = math.sqrt(mean_squared_error(final_df['x'], final_df['Local_Y']))
    overall_mae_l = mean_absolute_error(final_df['l'], final_df['Lane_ID'])
    
    return overall_mae, overall_mape, overall_rmse, overall_mae_l

def lc_timing_err_calculation(df_dict, output_root, exp_name, exp_id):
    
    lc_time_diff = []
    count = 0
    lc_time_diff_undtct = []
    count_undtct = 0
    count_error = 0
    
    # Calculate the total number of planning horizons
    num_ph = len(df_dict)
    
    for case_index in range(1, num_ph+1):
        if case_index not in []:
            
            opt_result=pd.read_csv(output_root+exp_name+'/'+str(case_index)+'/'+exp_id+'.csv')
            all_veh=df_dict[case_index].copy()
            
            veh_id_list=all_veh['Vehicle_ID'].drop_duplicates().tolist()
            
            time_sequence_rp = np.arange(min(all_veh.Global_Time), max(all_veh.Global_Time)+1, 1).tolist()
            time_sequence_rp = [round(num, 1) for num in time_sequence_rp]
            for i in range(len(opt_result)):
                veh_id_temp = opt_result.Vehicle.iloc[i]
                time_id_temp = int(opt_result.Time.iloc[i])
                opt_result.loc[opt_result.index[i], 'Vehicle'] = veh_id_list[veh_id_temp]
                opt_result.loc[opt_result.index[i], 'Time'] = time_sequence_rp[time_id_temp]
            opt_result=opt_result.dropna(subset=['v'])
            opt_result.rename(columns={'Vehicle': 'Vehicle_ID'}, inplace=True)
            opt_result.rename(columns={'Time': 'Global_Time'}, inplace=True)
            
            merged_df = pd.merge(opt_result,  all_veh, on=['Vehicle_ID', 'Global_Time'])
            
            lane_GT={}
            lane_opt={}
            lane_GT_undtct={}
            for veh_id in veh_id_list:
                lane_GT[veh_id]={}
                lane_opt[veh_id]={}
                lane_GT_undtct[veh_id]={}
                veh_temp = merged_df[merged_df.Vehicle_ID == veh_id]
                for j in range(len(veh_temp)-1):
                    if veh_temp.Lane_ID.iloc[j] != veh_temp.Lane_ID.iloc[j+1]:
                        lane_GT[veh_id][(veh_temp.Lane_ID.iloc[j],veh_temp.Lane_ID.iloc[j+1])] = veh_temp.Global_Time.iloc[j+1]
                        if veh_temp.Detected.iloc[j+1] == 0 or veh_temp.Detected.iloc[j] == 0:
                            lane_GT_undtct[veh_id][(veh_temp.Lane_ID.iloc[j],veh_temp.Lane_ID.iloc[j+1])] = veh_temp.Global_Time.iloc[j+1]
                    if veh_temp.l.iloc[j] != veh_temp.l.iloc[j+1]:
                        lane_opt[veh_id][(veh_temp.l.iloc[j],veh_temp.l.iloc[j+1])] = veh_temp.Global_Time.iloc[j+1]
            
            for k, v in lane_GT.items():
                if len(v) == 0:
                    continue
                else:
                    for k_sub, v_sub in v.items():
                        if k_sub in lane_opt[k]:
                            lc_time_diff.append(abs(v_sub-lane_opt[k][k_sub]))
                            count+=1
                        else:
                            count_error+=1
            
            for k, v in lane_GT_undtct.items():
                if len(v) == 0:
                    continue
                else:
                    for k_sub, v_sub in v.items():
                        if k_sub in lane_opt[k]:
                            lc_time_diff_undtct.append(abs(v_sub-lane_opt[k][k_sub]))
                            count_undtct+=1
                            
    mae_lc = sum(lc_time_diff)/count
    mae_lc_undtct = sum(lc_time_diff_undtct)/count_undtct
                        
    return mae_lc, mae_lc_undtct