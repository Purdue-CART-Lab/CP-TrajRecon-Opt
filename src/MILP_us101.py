#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 16:55:58 2025

@author: Tianheng Zhu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyomo.environ import *
import pyomo.environ as pyo
import os
from collections import defaultdict
from typing import Any, Dict

# Silence pyomo warning
import logging
logging.getLogger("pyomo.core").setLevel(logging.CRITICAL)
logging.getLogger("pyomo").setLevel(logging.CRITICAL)
        
def Preparation_for_optimization_indexed(all_veh_input, interval = 1):
    
    def identify_overtake_vehicles(df):
        """
        Identify vehicles that overtake at least one other, under the conditions that:
          1. Each vehicle starts and ends on the same lane.
          2. Overtaking is only checked within that same lane.
        Returns a sorted list of Vehicle_IDs that perform at least one overtake.
        """
    
        # ---------------------------------------------------------------
        # 1) Basic Preprocessing and 'Start/End Same Lane' Filter
        # ---------------------------------------------------------------
        df_sorted = df.sort_values(by=['Vehicle_ID', 'Global_Time'])
        
        # First and last records for each vehicle
        first_records = df_sorted.groupby('Vehicle_ID').first()
        last_records  = df_sorted.groupby('Vehicle_ID').last()
        
        # Vehicles whose first and last Lane_ID match
        same_lane_mask = (first_records['Lane_ID'] == last_records['Lane_ID'])
        vehicles_start_end_same = first_records.index[same_lane_mask]
    
        # Check that those vehicles visited >1 lane overall
        df_detected = df_sorted[df_sorted['Detected'] == 1]
        lane_counts = df_detected.groupby('Vehicle_ID')['Lane_ID'].nunique()
        vehicles_start_end_same_multi = [
            v for v in vehicles_start_end_same
            if lane_counts.get(v, 0) > 1
        ]
        
        # Make a small dataframe for the start/end-same-lane subset
        start_end_same_times = pd.DataFrame({
            'Vehicle_ID': vehicles_start_end_same,
            'First_Time': first_records.loc[vehicles_start_end_same, 'Global_Time'],
            'Last_Time':  last_records.loc[vehicles_start_end_same, 'Global_Time'],
            'Start_Lane': first_records.loc[vehicles_start_end_same, 'Lane_ID'],
            'End_Lane':   last_records.loc[vehicles_start_end_same, 'Lane_ID']
        }).reset_index(drop=True)
        
        # Build a subset of the full dataframe containing ONLY these vehicles
        allowed_vehicles = set(vehicles_start_end_same)
        df_same = df_sorted[df_sorted['Vehicle_ID'].isin(allowed_vehicles)]
        
        # ---------------------------------------------------------------
        # 2) Group vehicles by their start/end lane
        # ---------------------------------------------------------------
        # Build a dict: lane -> list of vehicles whose (start_lane == end_lane == lane)
        lane_entries_exits = defaultdict(list)
        for row in start_end_same_times.itertuples(index=False):
            veh_id = row.Vehicle_ID
            lane   = row.Start_Lane
            
            # Subset for this one vehicle in this lane
            df_sub = df_same[(df_same['Vehicle_ID'] == veh_id) & (df_same['Lane_ID'] == lane)]
            if df_sub.empty:
                continue
            
            # ENTRY EVENT: the row with smallest (Global_Time, Local_Y)
            # nsmallest(1, columns=['Global_Time','Local_Y']) picks the earliest time,
            # breaking ties by the lowest localY.
            entry_row = df_sub.nsmallest(1, ['Global_Time','Local_Y']).iloc[0]
            entry_time    = entry_row['Global_Time']
            entry_local_y = entry_row['Local_Y']
            
            # EXIT EVENT: the row with largest (Global_Time, Local_Y)
            # nlargest(1, columns=['Global_Time','Local_Y']) picks the latest time,
            # breaking ties by the highest localY.
            exit_row = df_sub.nlargest(1, ['Global_Time','Local_Y']).iloc[0]
            exit_time    = exit_row['Global_Time']
            exit_local_y = exit_row['Local_Y']
            
            # Store
            lane_entries_exits[lane].append((
                veh_id,
                entry_time, entry_local_y,
                exit_time,  exit_local_y
            ))
        
        # ---------------------------------------------------------------
        # 3) Lane-by-lane Overtaking Check
        # ---------------------------------------------------------------
        overtakers = set()
    
        for lane, records in lane_entries_exits.items():
            if len(records) < 2:
                continue  # No comparisons if fewer than 2 vehicles in this lane
            
            # Convert to DataFrame for easier sorting
            lane_df = pd.DataFrame(
                records,
                columns=["Vehicle_ID", "entry_time", "entry_localY", "exit_time", "exit_localY"]
            )
            
            # ENTRY ORDER:
            #   Sort ascending by (entry_time, entry_localY)
            lane_df_entry_sorted = lane_df.sort_values(
                by=["entry_time", "entry_localY"],
                ascending=[True, True]
            ).reset_index(drop=True)
            
            # EXIT ORDER:
            #   Sort ascending by exit_time, but descending by exit_localY
            lane_df_exit_sorted = lane_df.sort_values(
                by=["exit_time", "exit_localY"],
                ascending=[True, False]
            ).reset_index(drop=True)
            
            # Build dicts to map each vehicle -> rank index
            exit_positions = {
                row.Vehicle_ID: i for i, row in lane_df_exit_sorted.iterrows()
            }
            
            # Compare pairs in ENTRY order
            vehicle_ids = lane_df_entry_sorted["Vehicle_ID"].tolist()
            n = len(vehicle_ids)
            
            for i in range(n):
                u = vehicle_ids[i]
                for j in range(i+1, n):
                    v = vehicle_ids[j]
                    
                    # By construction, v enters after u
                    # If v exits before u => v overtook u
                    if exit_positions[v] < exit_positions[u]:
                        overtakers.add(v)
        
        # ---------------------------------------------------------------
        # 4) Return or print final results
        # ---------------------------------------------------------------
        overtakers_sorted = sorted(overtakers)
        combined_set = vehicles_start_end_same_multi + overtakers_sorted
        final_vehicle_list = sorted(combined_set)
        
        return final_vehicle_list
    
    all_veh = all_veh_input.copy()
    detected_veh = all_veh.loc[all_veh["Detected"] == 1]
    
    veh_id_list=all_veh['Vehicle_ID'].drop_duplicates().tolist()
    time_sequence = np.arange(min(all_veh.Global_Time), max(all_veh.Global_Time)+1, 1).tolist()
    time_sequence = [round(num, 1) for num in time_sequence]
    
    overtake_list = identify_overtake_vehicles(all_veh)
    overtake_list = [veh_id_list.index(veh_id) for veh_id in overtake_list]
    
    N=len(veh_id_list)#number of vehicles
    T=len(time_sequence)
    v_detected_max = max(detected_veh.v_Vel)
    
    start_dist={}
    start_v={}
    start_a={}
    start_lane={}
    arrival_time={}
    end_dist={}
    end_v={}
    end_lane={}
    departure_time={}
    veh_length={}
    for veh_id in veh_id_list:
        start_dist[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['Local_Y'].iloc[0]
        start_v[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['v_Vel'].iloc[0]
        start_a[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['v_Acc'].iloc[0]
        arrival_time[veh_id_list.index(veh_id)]=time_sequence.index(all_veh[all_veh.Vehicle_ID==veh_id]['Global_Time'].iloc[0])
        end_dist[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['Local_Y'].iloc[-1]
        end_v[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['v_Vel'].iloc[-1]
        departure_time[veh_id_list.index(veh_id)]=time_sequence.index(all_veh[all_veh.Vehicle_ID==veh_id]['Global_Time'].iloc[-1])
        start_lane[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['Lane_ID'].iloc[0]
        end_lane[veh_id_list.index(veh_id)]=all_veh[all_veh.Vehicle_ID==veh_id]['Lane_ID'].iloc[-1]
        veh_len = all_veh[all_veh.Vehicle_ID==veh_id]['v_Length'].iloc[0]
        veh_length[veh_id_list.index(veh_id)] = min(veh_len, 4)
        
    detected_veh_id_list=detected_veh['Vehicle_ID'].drop_duplicates().tolist()
    for i in range(len(detected_veh)):
        detected_veh.loc[detected_veh.index[i], 'Global_Time'] = time_sequence.index(detected_veh.loc[detected_veh.index[i], 'Global_Time'])
    detected_dist={}
    detected_v={}
    detected_a={}
    detected_l={}
    for veh_id in detected_veh_id_list:
        detected_dist[veh_id_list.index(veh_id)] = detected_veh[detected_veh.Vehicle_ID==veh_id].set_index('Global_Time')['Local_Y'].to_dict()
        detected_v[veh_id_list.index(veh_id)] = detected_veh[detected_veh.Vehicle_ID==veh_id].set_index('Global_Time')['v_Vel'].to_dict()
        detected_a[veh_id_list.index(veh_id)] = detected_veh[detected_veh.Vehicle_ID==veh_id].set_index('Global_Time')['v_Acc'].to_dict()
        detected_l[veh_id_list.index(veh_id)] = detected_veh[detected_veh.Vehicle_ID==veh_id].set_index('Global_Time')['Lane_ID'].to_dict()
    
    
    detected_veh_id_list = [veh_id_list.index(item) for item in detected_veh_id_list]
    time_sequence=list(range(T))
    return N, T, start_dist, start_v, start_a, start_lane, arrival_time, end_dist, end_lane, end_v, departure_time, detected_dist, detected_v, detected_a, detected_l, detected_veh_id_list, v_detected_max, veh_length, overtake_list, time_sequence

def replay_results_complete(df_dict, output_root, exp_name, case_index, exp_id):
    
    opt_result=pd.read_csv(output_root+exp_name+'/'+str(case_index)+'/'+exp_id+'.csv')

    all_veh=df_dict[case_index].copy()
    detected_veh = all_veh.loc[all_veh["Detected"] == 1].copy()
    
    veh_id_list=all_veh['Vehicle_ID'].drop_duplicates().tolist()
    
    time_sequence_rp = np.arange(min(all_veh.Global_Time), max(all_veh.Global_Time)+1, 1).tolist()
    time_sequence_rp = [round(num, 1) for num in time_sequence_rp]
    for i in range(len(opt_result)):
        veh_id_temp = opt_result.Vehicle.iloc[i]
        time_id_temp = int(opt_result.Time.iloc[i])
        opt_result.loc[opt_result.index[i], 'Vehicle'] = veh_id_list[veh_id_temp]
        opt_result.loc[opt_result.index[i], 'Time'] = time_sequence_rp[time_id_temp]


    opt_result=opt_result.dropna(subset=['v'])
    
    for l in range(1,7):
        data_veh_l = opt_result[opt_result.l == l]
        all_veh_l = all_veh[all_veh.Lane_ID == l]
        detected_veh_l = detected_veh[detected_veh.Lane_ID == l]
        plt.figure(dpi=200)
        
        first_optimized = True
        first_ground_truth = True
        first_CAV = True
        first_Detected = True
        
        for veh_id in veh_id_list:
            data_veh_temp = data_veh_l[data_veh_l.Vehicle == veh_id]
            
            all_veh_temp = all_veh_l[all_veh_l.Vehicle_ID == veh_id]
            detected_veh_temp = detected_veh_l[detected_veh_l.Vehicle_ID == veh_id]
            
            if len(data_veh_temp) != 0:
                time_veh=list(data_veh_temp.Time)
                time_points=[[time_veh[0]]]
                for i in range(0,len(time_veh)-1):
                    if time_veh[i + 1] - time_veh[i] > 1:
                        time_points[-1].append(time_veh[i])
                        time_points.append([time_veh[i+1]])
                time_points[-1].append(time_veh[-1])
                for j in range(len(time_points)):
                    segment = data_veh_temp[(data_veh_temp.Time >= time_points[j][0]) & (data_veh_temp.Time <= time_points[j][1])]
                    if first_optimized:
                        plt.scatter(segment['Time'], segment['x'], color='green', s=8, label='Reconstructed')
                        plt.plot(segment['Time'], segment['x'], color='green',linewidth=2)
                        first_optimized = False
                    else:
                        plt.scatter(segment['Time'], segment['x'], color='green', s=8)
                        plt.plot(segment['Time'], segment['x'], color='green',linewidth=2)
            
            
            time_veh=list(all_veh_temp.Global_Time)
            if len(time_veh) == 0:
                continue
            time_points=[[time_veh[0]]]
            for i in range(0,len(time_veh)-1):
                if time_veh[i + 1] - time_veh[i] > 1:
                    time_points[-1].append(time_veh[i])
                    time_points.append([time_veh[i+1]])
            time_points[-1].append(time_veh[-1])
            
            for j in range(len(time_points)):
                segment = all_veh_temp[(all_veh_temp.Global_Time >= time_points[j][0]) & (all_veh_temp.Global_Time <= time_points[j][1])]
                if first_ground_truth:
                    plt.plot(segment['Global_Time'], segment['Local_Y'],color='chocolate', label='Undetected')
                    first_ground_truth = False
                else:
                    plt.plot(segment['Global_Time'], segment['Local_Y'],color='chocolate')      
            
            if len(detected_veh_temp) != 0:
                if (detected_veh_temp['CAV'] == 1).any():
                    color = 'red'
                else:
                    color = 'blue'
                time_veh_dtct=list(detected_veh_temp.Global_Time)
                time_points_dtct=[[time_veh_dtct[0]]]
                for i in range(0,len(time_veh_dtct)-1):
                    if time_veh_dtct[i + 1] - time_veh_dtct[i] > 1:
                        time_points_dtct[-1].append(time_veh_dtct[i])
                        time_points_dtct.append([time_veh_dtct[i+1]])
                time_points_dtct[-1].append(time_veh_dtct[-1])
    
                
                for j in range(len(time_points_dtct)):
                    segment = detected_veh_temp[(detected_veh_temp.Global_Time >= time_points_dtct[j][0]) & (detected_veh_temp.Global_Time <= time_points_dtct[j][1])]
                        
                    if color == 'red':
                        if first_CAV:
                            plt.plot(segment['Global_Time'], segment['Local_Y'], color=color, label='CAV')
                            first_CAV = False
                        else:
                            plt.plot(segment['Global_Time'], segment['Local_Y'], color=color)
                    elif color == 'blue':
                        if first_Detected:
                            plt.plot(segment['Global_Time'], segment['Local_Y'], color=color, label='Detected')
                            first_Detected = False
                        else:
                            plt.plot(segment['Global_Time'], segment['Local_Y'], color=color)
    
        plt.title('lane '+str(l))
        plt.xlim(min(opt_result.Time),max(opt_result.Time))
        plt.ylim(40,560) # hardcode
        plt.xlabel('Time (seconds)')
        plt.ylabel('Distance (meters)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(output_root+exp_name+'/'+str(case_index)+'/'+exp_id+'/'+str(l)+'.png')
        plt.close()
    
def optimization_us101(
    df_dict,
    case_index,
    cfg,
):
    
    print('Case '+str(case_index)+' starts!')
    
    io = cfg["io"]
    output_root = io["output_root"]
    exp_name = io["experiment_name"]
    exp_id = io["experiment_id"]
    
    
    w = cfg["weights"]
    b = cfg["bounds"]
    o = cfg["others"]
    
    error_weight = float(w["error_weight"])
    x_weight = float(w["x_weight"])
    a_weight = float(w["a_weight"])
    abslc_weight = float(w["abslc_weight"])
    mobil_weight = float(w["mobil_weight"])
    jerk_weight = float(w["jerk_weight"])
    
    speed_limit = float(b["speed_limit"])
    a_low = float(b["a_low"])
    a_high = float(b["a_high"])
    jerk_low = float(b["jerk_low"])
    jerk_high = float(b["jerk_high"])
    
    rc_time = float(o["rc_time"])
    M = int(o["M"])

    # Create the folder
    os.makedirs(output_root+exp_name+'/'+str(case_index)+'/'+exp_id, exist_ok=True)
    
    N, T, start_dist, start_v, start_a, start_lane, arrival_time, end_dist, end_lane, end_v, departure_time, detected_dist, detected_v, detected_a, detected_l, detected_veh_id_list, v_detected_max, veh_length, overtake_list, time_sequence = Preparation_for_optimization_indexed(df_dict[case_index])

    # Create a Pyomo model
    model = ConcreteModel()

    model.V = RangeSet(0, N-1)  # Vehicles
    model.T = RangeSet(0, T-1)  # Time steps
    
    # Variables: acceleration of vehicle v at time t
    
    model.a = Var(model.V, model.T, domain=Reals)
    # Absolute value of a
    model.abs_a = Var(model.V, model.T, domain=NonNegativeReals)
    # Absolute jerk of vehicle v at time t
    model.jerk_abs = Var(model.V, model.T, domain=NonNegativeReals)
    # Absolute value of a
    model.lc_delta_a = Var(model.V, model.T, domain=Reals)
    # Velocity of vehicle v at time t
    model.v = Var(model.V, model.T, domain=NonNegativeReals)
    # Position of vehicle v at time t
    model.x = Var(model.V, model.T, domain=NonNegativeReals)
    # Lane_id of vehicle v at time t, 0, 1, 2, 3
    model.l = Var(model.V, model.T, domain=NonNegativeIntegers, bounds=(1, 6))
    # Lane changing decision of vehicle v at time t, -1, 0, 1
    model.lc = Var(model.V, model.T, domain=Integers, bounds=(-1, 1))
    # Absolute value of lc
    model.abs_lc = Var(model.V, model.T, domain=Binary)
    # This will be indexed by (v, t).
    model.sign_lc = Var(model.V, model.T, within=Binary)
    # Variables indicating if vehicle v are in lane 0,1,2,3 at time t
    model.l1 = Var(model.V, model.T, within=Binary)
    model.l2 = Var(model.V, model.T, within=Binary)
    model.l3 = Var(model.V, model.T, within=Binary)
    model.l4 = Var(model.V, model.T, within=Binary)
    model.l5 = Var(model.V, model.T, within=Binary)
    model.l6 = Var(model.V, model.T, within=Binary)
    
    # Binary variable to decide if two vehicle are in the same lane
    model.z = Var(model.V, model.V, model.T, domain=Binary)

    # Auxilliary binary variable to decide if v is in front of u
    model.y = Var(model.V, model.V, model.T, domain=Binary)

    # Define auxiliary variables u and w
    model.u = Var(model.V, model.T, domain=NonNegativeReals)
    model.w = Var(model.V, model.T, domain=NonNegativeReals)
    '''
    # Acceleration
    model.a = Var(model.V, model.T, domain=Reals, initialize=0.0)
    
    # Absolute value of acceleration
    model.abs_a = Var(model.V, model.T, domain=NonNegativeReals, initialize=0.0)
    
    # Absolute jerk
    model.jerk_abs = Var(model.V, model.T, domain=NonNegativeReals, initialize=0.0)
    
    # Lane-change delta acceleration
    model.lc_delta_a = Var(model.V, model.T, domain=Reals, initialize=0.0)
    
    # Velocity
    model.v = Var(model.V, model.T, domain=NonNegativeReals, initialize=0.0)
    
    # Position
    model.x = Var(model.V, model.T, domain=NonNegativeReals, initialize=0.0)
    
    # Lane index (1–6)
    model.l = Var(
        model.V, model.T,
        domain=NonNegativeIntegers,
        bounds=(1, 6),
        initialize=1
    )
    
    # Lane-change decision (-1, 0, 1)
    model.lc = Var(
        model.V, model.T,
        domain=Integers,
        bounds=(-1, 1),
        initialize=0
    )
    
    # Absolute lane change indicator
    model.abs_lc = Var(model.V, model.T, domain=Binary, initialize=0)
    
    # Sign of lane change
    model.sign_lc = Var(model.V, model.T, domain=Binary, initialize=0)
    
    # Lane membership binaries
    model.l1 = Var(model.V, model.T, domain=Binary, initialize=0)
    model.l2 = Var(model.V, model.T, domain=Binary, initialize=0)
    model.l3 = Var(model.V, model.T, domain=Binary, initialize=0)
    model.l4 = Var(model.V, model.T, domain=Binary, initialize=0)
    model.l5 = Var(model.V, model.T, domain=Binary, initialize=0)
    model.l6 = Var(model.V, model.T, domain=Binary, initialize=0)
    
    # Same-lane indicator between vehicles
    model.z = Var(model.V, model.V, model.T, domain=Binary, initialize=0)
    
    # Front/back ordering indicator
    model.y = Var(model.V, model.V, model.T, domain=Binary, initialize=0)
    
    # Auxiliary variables
    model.u = Var(model.V, model.T, domain=NonNegativeReals, initialize=0.0)
    model.w = Var(model.V, model.T, domain=NonNegativeReals, initialize=0.0)
    '''
    # Objective function
    def objective_rule(model):
        return (
            error_weight * sum(
                model.u[v, t] + 3*model.w[v, t]
                for v in model.V 
                if v in detected_dist 
                for t in detected_dist[v]
            )
            + abslc_weight*sum(
                model.abs_lc[v, t]
                for t in time_sequence
                for v in model.V 
                if t >= arrival_time[v] and t <= departure_time[v]
            )
            + a_weight * sum(
                model.abs_a[v, t]
                for t in time_sequence
                for v in model.V 
                if t >= arrival_time[v] and t <= departure_time[v]
            )
            + jerk_weight * sum(
                model.jerk_abs[v, t]
                for t in time_sequence
                for v in model.V
                if t > arrival_time[v] and t <= departure_time[v]
            )
            - x_weight * sum(
                model.x[v, t]
                for t in time_sequence
                for v in model.V 
                if t >= arrival_time[v] and t <= departure_time[v]
            )- mobil_weight * sum(
                model.lc_delta_a[v,t] 
                for t in time_sequence
                for v in model.V 
                if t > arrival_time[v] and t <= departure_time[v]
            )
        )
    
    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Constraints
    # Constarints for objectoive to linearize the absolute values
    def abs_constraints_u1(model, v, t):
        if v in detected_dist:
            if t in detected_dist[v]:
                return model.u[v, t] >= model.x[v, t] - detected_dist[v][t]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.abs_constraints_u1 = Constraint(model.V, model.T, rule=abs_constraints_u1)

    def abs_constraints_u2(model, v, t):
        if v in detected_dist:
            if t in detected_dist[v]:
                return model.u[v, t] >= -(model.x[v, t] - detected_dist[v][t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.abs_constraints_u2 = Constraint(model.V, model.T, rule=abs_constraints_u2)

    def abs_constraints_w1(model, v, t):
        if v in detected_dist:
            if t in detected_dist[v]:
                return model.w[v, t] >= model.l[v, t] - detected_l[v][t]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.abs_constraints_w1 = Constraint(model.V, model.T, rule=abs_constraints_w1)

    def abs_constraints_w2(model, v, t):
        if v in detected_dist:
            if t in detected_dist[v]:
                return model.w[v, t] >= -(model.l[v, t] - detected_l[v][t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.abs_constraints_w2 = Constraint(model.V, model.T, rule=abs_constraints_w2)
    
    # Distance, Speed, and Acceleration
    def start_dist_rule(model, v):
        return model.x[v, arrival_time[v]] == start_dist[v]
    model.start_dist = Constraint(model.V, rule=start_dist_rule)
    
    def start_v_rule(model, v):
        return model.v[v, arrival_time[v]] == start_v[v]
    model.start_v = Constraint(model.V, rule=start_v_rule)
    
    def distance_end_rule(model, v, t):
        if t == departure_time[v]:
            return model.x[v, t] == end_dist[v]
        else:
            return Constraint.Skip
    model.distance_end = Constraint(model.V, model.T, rule=distance_end_rule)
    
    def distance_update_rule(model, v, t):
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.x[v, t] == model.x[v, t-1] + model.v[v, t-1] + 0.5 * model.a[v, t-1]
        else:
            return Constraint.Skip
    model.distance_update = Constraint(model.V, model.T, rule=distance_update_rule)
    
    def velocity_update_rule(model, v, t):
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.v[v, t] == model.v[v, t-1] + model.a[v, t-1]
        else:
            return Constraint.Skip
    model.velocity_update = Constraint(model.V, model.T, rule=velocity_update_rule)
    
    def velocity_limit_rule(model, v, t):
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.v[v, t] <= speed_limit  # Use '<=' instead of '<'
        else:
            return Constraint.Skip
    model.velocity_limit = Constraint(model.V, model.T, rule=velocity_limit_rule)
    
    def acc_limit_rule(model,v,t):
        if t >= arrival_time[v] and t <= departure_time[v]:
            return (a_low, model.a[v, t], a_high)
        else:
            return Constraint.Skip
    model.acc_limit = Constraint(model.V, model.T, rule=acc_limit_rule)
    
    def abs_a_constraints_1(model, v, t):
        return model.abs_a[v, t] >= model.a[v, t]
    model.abs_a_constraints_1 = Constraint(model.V, model.T, rule=abs_a_constraints_1)

    def abs_a_constraints_2(model, v, t):
        return model.abs_a[v, t] >= -model.a[v, t]
    model.abs_a_constraints_2 = Constraint(model.V, model.T, rule=abs_a_constraints_2)
    
    def jerk_rule(model,v,t):
        if t >= arrival_time[v] + 1 and t <= departure_time[v]:
            return (jerk_low, model.a[v, t] - model.a[v, t-1], jerk_high) #-3 3
        else:
            return Constraint.Skip
    model.jerk = Constraint(model.V, model.T, rule=jerk_rule)
      
    def jerk_abs_1_rule(model, v, t):
        # Only relevant if t> arrival_time[v] (so that t-1 is valid)
        if t >= arrival_time[v] + 1 and t <= departure_time[v]:
            return model.jerk_abs[v, t] >= model.a[v, t] - model.a[v, t-1]
        else:
            return Constraint.Skip
    model.jerk_abs_1 = Constraint(model.V, model.T, rule=jerk_abs_1_rule)
    
    def jerk_abs_2_rule(model, v, t):
        if t >= arrival_time[v] + 1 and t <= departure_time[v]:
            return model.jerk_abs[v, t] >= -(model.a[v, t] - model.a[v, t-1])
        else:
            return Constraint.Skip
    model.jerk_abs_2 = Constraint(model.V, model.T, rule=jerk_abs_2_rule)
    
    def start_lane_rule(model, v):
        return model.l[v, arrival_time[v]] == start_lane[v]
    model.start_lane = Constraint(model.V, rule=start_lane_rule)

    def end_lane_rule(model, v):
        return model.l[v, departure_time[v]] == end_lane[v]
    model.end_lane = Constraint(model.V, rule=end_lane_rule)

    def l_update_rule(model, v, t):
        if t>arrival_time[v] and t <= departure_time[v]:
            return model.l[v,t] == model.l[v,t-1] + model.lc[v,t]
        else:
            return Constraint.Skip
    model.l_update = Constraint(model.V, model.T, rule=l_update_rule)
    
    def no_lane_changing_l_rule(model,v,t):
        if start_lane[v]==end_lane[v] and v not in overtake_list:
            if t >= arrival_time[v] and t <= departure_time[v]:
                return model.l[v, t] == model.l[v, arrival_time[v]]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.no_lane_changing_l = Constraint(model.V, model.T, rule=no_lane_changing_l_rule)

    def no_lane_changing_lc_rule(model,v,t):
        if start_lane[v]==end_lane[v] and v not in overtake_list:
            if t > arrival_time[v] and t <= departure_time[v]:
                return model.lc[v, t] == 0
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.no_lane_changing_lc = Constraint(model.V, model.T, rule=no_lane_changing_lc_rule)
    
    def lc_direction_rule(model, v, t):
        return sum(model.lc[v, t] for t in time_sequence) == end_lane[v] - start_lane[v]
    model.lc_direction = Constraint(model.V, model.T, rule=lc_direction_rule)

    # (A) Enforce sign logic for lc[v,t].
    #     If sign_lc[v,t] = 1 ==> lc[v,t] >= 0
    #     If sign_lc[v,t] = 0 ==> lc[v,t] <= 0
    def sign_lc_1_rule(model, v, t):
        # lc[v,t] <= M * sign_lc[v,t]
        return model.lc[v, t] <= M * model.sign_lc[v, t]
    model.sign_lc_1 = Constraint(model.V, model.T, rule=sign_lc_1_rule)
    
    def sign_lc_2_rule(model, v, t):
        # lc[v,t] >= -M * (1 - sign_lc[v,t])
        return model.lc[v, t] >= -M * (1 - model.sign_lc[v, t])
    model.sign_lc_2 = Constraint(model.V, model.T, rule=sign_lc_2_rule)
    
    # (B) Enforce abs_lc[v,t] = |lc[v,t]|
    def abs_lc_1_rule(model, v, t):
        return model.abs_lc[v, t] >= model.lc[v, t]
    model.abs_lc_1 = Constraint(model.V, model.T, rule=abs_lc_1_rule)
    
    def abs_lc_2_rule(model, v, t):
        return model.abs_lc[v, t] >= -model.lc[v, t]
    model.abs_lc_2 = Constraint(model.V, model.T, rule=abs_lc_2_rule)
    
    def abs_lc_3_rule(model, v, t):
        # abs_lc[v,t] <= lc[v,t] + M*(1 - sign_lc[v,t])
        return model.abs_lc[v, t] <= model.lc[v, t] + M * (1 - model.sign_lc[v, t])
    model.abs_lc_3 = Constraint(model.V, model.T, rule=abs_lc_3_rule)
    
    def abs_lc_4_rule(model, v, t):
        # abs_lc[v,t] <= -lc[v,t] + M*sign_lc[v,t]
        return model.abs_lc[v, t] <= -model.lc[v, t] + M * model.sign_lc[v, t]
    model.abs_lc_4 = Constraint(model.V, model.T, rule=abs_lc_4_rule)
    
    def lc_delta_a_upper_rule(model, v, t):
        # lc_delta_a[v,t] ≤ (a[v,t] - a[v,t-1]) + M*(1 - abs_lc[v,t])
        # meaning if abs_lc[v,t] = 1 => lc_delta_a[v,t] ≤ a[v,t] - a[v,t-1]
        # if abs_lc[v,t] = 0 => no binding from this constraint
        if arrival_time[v] < t <= departure_time[v]:
            return (model.lc_delta_a[v,t]
                    <= (model.a[v,t] - model.a[v,t-1])
                    + M*(1 - model.abs_lc[v,t]))
        else:
            return Constraint.Skip 
    model.lc_delta_a_upper = Constraint(model.V, model.T, rule=lc_delta_a_upper_rule)
    
    def lc_delta_a_lower_rule(model, v, t):
        # lc_delta_a[v,t] ≥ (a[v,t] - a[v,t-1]) - M*(1 - abs_lc[v,t])
        if arrival_time[v] < t <= departure_time[v]:
            return (model.lc_delta_a[v,t]
                    >= (model.a[v,t] - model.a[v,t-1])
                    - M*(1 - model.abs_lc[v,t]))
        else:
            return Constraint.Skip
    model.lc_delta_a_lower = Constraint(model.V, model.T, rule=lc_delta_a_lower_rule)
    
    def lc_delta_a_zero_upper_rule(model, v, t):
        # If no lane change => lc_delta_a[v,t] ≤ +M * abs_lc[v,t]
        # i.e. if abs_lc[v,t] = 0 => lc_delta_a[v,t] ≤ 0
        # combined with the next constraint => 0
        if arrival_time[v] < t <= departure_time[v]:
            return (model.lc_delta_a[v,t] 
                    <= M * model.abs_lc[v,t])
        else:
            return Constraint.Skip
    model.lc_delta_a_zero_upper = Constraint(model.V, model.T, rule=lc_delta_a_zero_upper_rule)
    
    def lc_delta_a_zero_lower_rule(model, v, t):
        # If no lane change => lc_delta_a[v,t] ≥ -M * abs_lc[v,t]
        # i.e. if abs_lc[v,t] = 0 => lc_delta_a[v,t] ≥ 0
        if arrival_time[v] < t <= departure_time[v]:
            return (model.lc_delta_a[v,t]
                    >= -M * model.abs_lc[v,t])
        else:
            return Constraint.Skip
    model.lc_delta_a_zero_lower = Constraint(model.V, model.T, rule=lc_delta_a_zero_lower_rule)
    
    #Create model.l0/l1/l2/l3 variable, model.l0=1 means vehicle v is in lane 0 at time t
    def l_binary_1_rule(model, v, t):
        if arrival_time[v] <= t <= departure_time[v]:
            return (
                model.l1[v, t]
              + model.l2[v, t]
              + model.l3[v, t]
              + model.l4[v, t]
              + model.l5[v, t]
              + model.l6[v, t]
            ) == 1
        else:
            return Constraint.Skip
    model.l_binary_1 = Constraint(model.V, model.T, rule=l_binary_1_rule)
    
    def l_binary_2_rule(model, v, t):
        if arrival_time[v] <= t <= departure_time[v]:
            return model.l[v, t] == (
                1*model.l1[v, t] +
                2*model.l2[v, t] +
                3*model.l3[v, t] +
                4*model.l4[v, t] +
                5*model.l5[v, t] +
                6*model.l6[v, t]
            )
        else:
            return Constraint.Skip
    model.l_binary_2 = Constraint(model.V, model.T, rule=l_binary_2_rule)
    
    def y_constraint_rule_1(model, v, u, t):
        if v != u:
            if t >= arrival_time[v] and t <= departure_time[v] and t >= arrival_time[u] and t <= departure_time[u]:
                return model.x[v,t] - model.x[u,t] >= 1e-20- M * (1 - model.y[v, u, t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.y_constraint_1 = Constraint(model.V, model.V, model.T, rule=y_constraint_rule_1)
    
    def y_constraint_rule_2(model, v, u, t):
        if v != u:
            if t >= arrival_time[v] and t <= departure_time[v] and t >= arrival_time[u] and t <= departure_time[u]:
                return model.x[v,t] - model.x[u,t] <=  M * model.y[v, u, t]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.y_constraint_2 = Constraint(model.V, model.V, model.T, rule=y_constraint_rule_2)
    
    #if two vehicles are in the same lane and don't make LC at the current lane, then their lanes at the next time stamp should be the same
    def y_constraint_rule_3(model, v, u, t):
        if v != u:
            if t > arrival_time[v] and t <= departure_time[v] and t > arrival_time[u] and t <= departure_time[u]:
                return model.y[v, u, t] - model.y[v, u, t-1] >= -M * (model.abs_lc[v,t] + model.abs_lc[u,t] + model.z[v,u,t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip 
    model.y_constraint_3 = Constraint(model.V, model.V, model.T, rule=y_constraint_rule_3)
    
    def y_constraint_rule_4(model, v, u, t):
        if v != u:
            if t > arrival_time[v] and t <= departure_time[v] and t > arrival_time[u] and t <= departure_time[u]:
                return model.y[v, u, t] - model.y[v, u, t-1] <= M * (model.abs_lc[v,t] + model.abs_lc[u,t] + model.z[v,u,t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip   
    model.y_constraint_4 = Constraint(model.V, model.V, model.T, rule=y_constraint_rule_4)
    
    # --- Lane 1 (l1[v,t]) ---
    def z_constraint_rule_1(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l1[u,t] - model.l1[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_2(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l1[v,t] - model.l1[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_3(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            # "2 - l1[v,t] - l1[u,t]" ensures z[v,u,t] can be zero if v and u are in the same lane
            return model.z[v,u,t] <= 2 - model.l1[v,t] - model.l1[u,t]
        return Constraint.Skip
    
    # --- Lane 2 (l2[v,t]) ---
    def z_constraint_rule_4(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l2[u,t] - model.l2[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_5(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l2[v,t] - model.l2[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_6(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] <= 2 - model.l2[v,t] - model.l2[u,t]
        return Constraint.Skip
    
    # --- Lane 3 (l3[v,t]) ---
    def z_constraint_rule_7(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l3[u,t] - model.l3[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_8(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l3[v,t] - model.l3[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_9(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] <= 2 - model.l3[v,t] - model.l3[u,t]
        return Constraint.Skip
    
    # --- Lane 4 (l4[v,t]) ---
    def z_constraint_rule_10(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l4[u,t] - model.l4[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_11(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l4[v,t] - model.l4[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_12(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] <= 2 - model.l4[v,t] - model.l4[u,t]
        return Constraint.Skip
    
    # --- Lane 5 (l5[v,t]) ---
    def z_constraint_rule_13(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l5[u,t] - model.l5[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_14(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l5[v,t] - model.l5[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_15(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] <= 2 - model.l5[v,t] - model.l5[u,t]
        return Constraint.Skip
    
    # --- Lane 6 (l6[v,t]) ---
    def z_constraint_rule_16(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l6[u,t] - model.l6[v,t]
        return Constraint.Skip
    
    def z_constraint_rule_17(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] >= model.l6[v,t] - model.l6[u,t]
        return Constraint.Skip
    
    def z_constraint_rule_18(model, v, u, t):
        if (arrival_time[v] <= t <= departure_time[v]) and (arrival_time[u] <= t <= departure_time[u]):
            return model.z[v,u,t] <= 2 - model.l6[v,t] - model.l6[u,t]
        return Constraint.Skip

    # ADD z-CONSTRAINTS TO THE MODEL
    model.z_constraint_1  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_1)
    model.z_constraint_2  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_2)
    model.z_constraint_3  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_3)
    model.z_constraint_4  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_4)
    model.z_constraint_5  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_5)
    model.z_constraint_6  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_6)
    model.z_constraint_7  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_7)
    model.z_constraint_8  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_8)
    model.z_constraint_9  = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_9)
    model.z_constraint_10 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_10)
    model.z_constraint_11 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_11)
    model.z_constraint_12 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_12)
    model.z_constraint_13 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_13)
    model.z_constraint_14 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_14)
    model.z_constraint_15 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_15)
    model.z_constraint_16 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_16)
    model.z_constraint_17 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_17)
    model.z_constraint_18 = Constraint(model.V, model.V, model.T, rule=z_constraint_rule_18)
    
    # Force x[v,t] in [200,400] if abs_lc[v,t]==1 AND l6[v,t]==1
    def lane6_change_now_x_lower_rule(model, v, t):
        # If abs_lc[v,t]=1 AND l6[v,t]=1 => x[v,t]>=200
        # otherwise relaxed by big-M
        if t >= arrival_time[v] and t <= departure_time[v]:
            return model.x[v,t] >= 187 - M * (2 - model.abs_lc[v,t] - model.l6[v,t])
        else:
            return Constraint.Skip
    model.lane6_change_now_x_lower = Constraint(model.V, model.T, rule=lane6_change_now_x_lower_rule)
    
    def lane6_change_now_x_upper_rule(model, v, t):
        # If abs_lc[v,t]=1 AND l6[v,t]=1 => x[v,t]<=400
        if t >= arrival_time[v] and t <= departure_time[v]:
            return model.x[v,t] <= 400 + M * (2 - model.abs_lc[v,t] - model.l6[v,t])
        else:
            return Constraint.Skip
    model.lane6_change_now_x_upper = Constraint(model.V, model.T, rule=lane6_change_now_x_upper_rule)

    # Force x[v,t] in [200,400] if abs_lc[v,t]==1 AND l6[v,t-1]==1  
    def lane6_change_prev_x_lower_rule(model, v, t):
        # If abs_lc[v,t]=1 AND l6[v,t-1]=1 => x[v,t]>=200
        # Make sure t>arrival_time[v] so t-1 is valid
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.x[v,t] >= 187 - M * (2 - model.abs_lc[v,t] - model.l6[v,t-1])
        else:
            return Constraint.Skip
    model.lane6_change_prev_x_lower = Constraint(model.V, model.T, rule=lane6_change_prev_x_lower_rule)
    
    def lane6_change_prev_x_upper_rule(model, v, t):
        # If abs_lc[v,t]=1 AND l6[v,t-1]=1 => x[v,t]<=400
        if t > arrival_time[v] and t <= departure_time[v]:
            return model.x[v,t] <= 400 + M * (2 - model.abs_lc[v,t] - model.l6[v,t-1])
        else:
            return Constraint.Skip
    model.lane6_change_prev_x_upper = Constraint(model.V, model.T, rule=lane6_change_prev_x_upper_rule)
    
    #if model.y[v, u, t]=1, model.x[v,t]>model.x[u,t], v is in front of u
    #if model.z[v, u, t]=0, two vehicles are on the same lane
    def car_following_1_rule(model, v, u, t):
        if v != u:
            if t >= arrival_time[v] and t <= departure_time[v] and t >= arrival_time[u] and t <= departure_time[u]:
                return model.x[v,t] - model.x[u,t] >= veh_length[v] + rc_time*model.v[u,t] - M * model.z[v, u, t] - M * (1 - model.y[v, u, t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.car_following_1 = Constraint(model.V, model.V, model.T, rule=car_following_1_rule)
    
    def car_following_2_rule(model, v, u, t):
        if v != u:
            if t >= arrival_time[v] and t <= departure_time[v] and t >= arrival_time[u] and t <= departure_time[u]:
                return model.x[u,t] - model.x[v,t] >= veh_length[u] + rc_time*model.v[v,t] - M * model.z[v, u, t] - M * model.y[v, u, t]
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.car_following_2 = Constraint(model.V, model.V, model.T, rule=car_following_2_rule)
    
    # Solve the model
    solver = SolverFactory("gurobi_direct")#SolverFactory('cbc')
    
    start_time = time.time()  # Start timer
    results = solver.solve(model)
    end_time = time.time()  # End timer

    optimization_time = end_time - start_time

    # Open the file in write mode and save the text
    with open(output_root+exp_name+'/'+str(case_index)+'/'+exp_id+'/time.txt', 'w') as file:
        file.write(str(optimization_time))
    
    def extract_variables_to_dataframe(model, variable_name):
        var = getattr(model, variable_name)
        data = []
        for v in model.V:
            for t in model.T:
                try:
                    value = pyo.value(var[v, t])
                    data.append((v, t, value))
                except:
                    data.append((v, t, None))
        df = pd.DataFrame(data, columns=['Vehicle', 'Time', variable_name])
        return df

    def extract_variables_to_dataframe_3(model, variable_name):
        var = getattr(model, variable_name)
        data = []
        for v in model.V:
            for u in model.V:
                for t in model.T:
                    try:
                        value = pyo.value(var[v, u, t])
                        data.append((v, u, t, value))
                    except:
                        data.append((v, u, t, None))
        df = pd.DataFrame(data, columns=['Vehicle', 'LeadVehicle', 'Time', variable_name])
        return df
    
    a_df = extract_variables_to_dataframe(model, 'a')
    v_df = extract_variables_to_dataframe(model, 'v')
    x_df = extract_variables_to_dataframe(model, 'x')
    l_df = extract_variables_to_dataframe(model, 'l')
    lc_df = extract_variables_to_dataframe(model, 'lc')
    abs_lc_df = extract_variables_to_dataframe(model, 'abs_lc')
    l1_df = extract_variables_to_dataframe(model, 'l1')
    l2_df = extract_variables_to_dataframe(model, 'l2')
    l3_df = extract_variables_to_dataframe(model, 'l3')
    l4_df = extract_variables_to_dataframe(model, 'l4')
    l5_df = extract_variables_to_dataframe(model, 'l5')
    l6_df = extract_variables_to_dataframe(model, 'l6')
    
    dfs = [a_df, v_df, x_df, l_df, lc_df, abs_lc_df, l1_df, l2_df, l3_df,l4_df, l5_df, l6_df]

    # Merging dataframes on 'vehicle' and 'time'
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['Vehicle', 'Time'])

    csv_file_path = output_root+exp_name+'/'+str(case_index)+'/'+exp_id+'.csv'
    merged_df.to_csv(csv_file_path, index=False)
    
    replay_results_complete(df_dict, output_root, exp_name, case_index, exp_id)



