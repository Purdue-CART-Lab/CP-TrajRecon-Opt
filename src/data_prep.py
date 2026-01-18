#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:11:51 2026

@author: Tianheng Zhu
"""
import pandas as pd
import random
import numpy as np
from scipy.signal import savgol_filter

def traj_preprocessing_Lankershim(df, Section_ID, Direction):
    df_approach = df[(df.Section_ID == Section_ID) & (df.Direction == Direction)] # 1 = Westbound, 2 = Eastbound, 3 = Northbound, 4 = Southbound
    diffs = df_approach.groupby('Vehicle_ID')['Local_Y'].max() - df_approach.groupby('Vehicle_ID')['Local_Y'].min()
    mean_val = diffs.mean()
    std_val = diffs.std()
    three_sigma_val = mean_val - 3 * std_val

    valid_vehicle_ids = diffs[diffs >= three_sigma_val].index
    filtered_df = df_approach[df_approach["Vehicle_ID"].isin(valid_vehicle_ids)]
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def traj_preprocessing_US101(df):
    
    def detect_overtakes(df):

        df_sorted = df.sort_values(by=["Lane_ID", "Frame_ID", "Local_Y"])
        lane_groups = df_sorted.groupby(["Lane_ID", "Frame_ID"])
        
        positions_dict = {}
        for (lane, frame), group_data in lane_groups:
            # Sort again by Local_Y just to be safe
            group_data = group_data.sort_values("Local_Y")
            positions_dict[(lane, frame)] = list(group_data["Vehicle_ID"])
        
        overtake_counts = {}
        
        # For each lane, we iterate through frames in ascending order
        unique_lanes = df["Lane_ID"].unique()
        for lane in unique_lanes:
            # frames for this lane, sorted
            frames_lane = sorted(
                df[df["Lane_ID"] == lane]["Frame_ID"].unique()
            )
            
            for i in range(len(frames_lane) - 1):
                f_curr = frames_lane[i]
                f_next = frames_lane[i+1]
                
                # Current and next positions (lists of Vehicle_IDs)
                curr_positions = positions_dict.get((lane, f_curr), [])
                next_positions = positions_dict.get((lane, f_next), [])
                
                # We only check vehicles that appear in both frames
                common_vehicles = set(curr_positions).intersection(next_positions)
                
                # Convert to list so we can check ordering
                curr_order = [v for v in curr_positions if v in common_vehicles]
                next_order = [v for v in next_positions if v in common_vehicles]
                
                # Check pairs in curr_order to see if they swapped in next_order
                # A simple approach: compare each pair (v,u) in curr_order
                for v in common_vehicles:
                    for u in common_vehicles:
                        if v == u:
                            continue
                        # Find index in current order
                        v_curr_idx = curr_order.index(v)
                        u_curr_idx = curr_order.index(u)
                        # Find index in next order
                        v_next_idx = next_order.index(v)
                        u_next_idx = next_order.index(u)
                        
                        # "v overtook u" if v was behind u before but in front after
                        # i.e. v_curr_idx > u_curr_idx and v_next_idx < u_next_idx
                        if (v_curr_idx > u_curr_idx) and (v_next_idx <= u_next_idx):
                            overtake_counts[v] = overtake_counts.get(v, 0) + 1       
        return overtake_counts
    
    def iterative_overtake_removal(df):
        removed_vehicles_history = []
        
        while True:
            overtake_counts = detect_overtakes(df)
            
            if not overtake_counts:
                # No overtakes found at all
                break
            
            max_overtakes = max(overtake_counts.values())
            if max_overtakes == 0:
                # No actual overtakes
                break
            
            # Find all vehicles that share the max overtake count
            worst_offenders = [v for v, cnt in overtake_counts.items() if cnt == max_overtakes]
            
            # Record who we are removing and how many overtakes they had
            for veh in worst_offenders:
                removed_vehicles_history.append((veh, max_overtakes))
            
            # Remove those vehicles from df
            df = df[~df["Vehicle_ID"].isin(worst_offenders)].copy()
            
        return df, removed_vehicles_history
    
    # Define a function that computes corrected velocity for the first row in each vehicle
    def correct_first_speed(group):
        if len(group) < 2:
            # If there's only one record for this vehicle, can't correct based on next record
            return group
        
        # Identify the first and second rows for this vehicle
        first_index = group.index[0]
        second_index = group.index[1]
        
        y1 = group.loc[first_index, 'Local_Y']
        y2 = group.loc[second_index, 'Local_Y']
        t1 = group.loc[first_index, 'Global_Time']
        t2 = group.loc[second_index, 'Global_Time']
        
        # Calculate corrected speed in (ft/s) if Local_Y is in feet and time is now in seconds
        corrected_speed = (y2 - y1) / (t2 - t1)
        
        # Assign to the first record's v_Vel
        group.at[first_index, 'v_Vel'] = corrected_speed
        
        return group
    
    diffs = df.groupby('Vehicle_ID')['Global_Time'].max() - df.groupby('Vehicle_ID')['Global_Time'].min()
    mean_val = diffs.mean()
    std_val = diffs.std()
    three_sigma_val = mean_val - 3 * std_val

    valid_vehicle_ids = diffs[diffs >= three_sigma_val].index
    df = df[df["Vehicle_ID"].isin(valid_vehicle_ids)] 
    df.reset_index(drop=True, inplace=True)
    
    # Run the procedure
    final_df, removal_list = iterative_overtake_removal(df)

    # Sort by Vehicle_ID and time (important to ensure the "first" record is truly the first in time)
    df = final_df.sort_values(['Vehicle_ID', 'Global_Time']).reset_index(drop=True)
    
    # Group by Vehicle_ID
    grouped = df.groupby('Vehicle_ID', group_keys=False)
    
    # Apply the correction function to each group
    df_corrected = grouped.apply(correct_first_speed)
    
    return df_corrected

def traj_preprocessing_MAGIC(df):
    
    def detect_overtakes(df):

        df_sorted = df.sort_values(by=["Lane_ID", "Global_Time", "Local_Y"])
        lane_groups = df_sorted.groupby(["Lane_ID", "Global_Time"])
        
        # Collect sorted lists of (Vehicle_ID) for each (Lane_ID, Frame_ID)
        # Example: positions_dict[(lane, frame)] = [v1, v2, v3 ...] in ascending order by Local_Y
        positions_dict = {}
        for (lane, frame), group_data in lane_groups:
            # Sort again by Local_Y just to be safe
            group_data = group_data.sort_values("Local_Y")
            positions_dict[(lane, frame)] = list(group_data["Vehicle_ID"])
        
        overtake_counts = {}
        
        # For each lane, we iterate through frames in ascending order
        unique_lanes = df["Lane_ID"].unique()
        for lane in unique_lanes:
            # frames for this lane, sorted
            frames_lane = sorted(
                df[df["Lane_ID"] == lane]["Global_Time"].unique()
            )
            
            for i in range(len(frames_lane) - 1):
                f_curr = frames_lane[i]
                f_next = frames_lane[i+1]
                
                # Current and next positions (lists of Vehicle_IDs)
                curr_positions = positions_dict.get((lane, f_curr), [])
                next_positions = positions_dict.get((lane, f_next), [])
                
                # We only check vehicles that appear in both frames
                common_vehicles = set(curr_positions).intersection(next_positions)
                
                # Convert to list so we can check ordering
                curr_order = [v for v in curr_positions if v in common_vehicles]
                next_order = [v for v in next_positions if v in common_vehicles]
                
                # Check pairs in curr_order to see if they swapped in next_order
                # A simple approach: compare each pair (v,u) in curr_order
                for v in common_vehicles:
                    for u in common_vehicles:
                        if v == u:
                            continue
                        # Find index in current order
                        v_curr_idx = curr_order.index(v)
                        u_curr_idx = curr_order.index(u)
                        # Find index in next order
                        v_next_idx = next_order.index(v)
                        u_next_idx = next_order.index(u)
                        
                        # "v overtook u" if v was behind u before but in front after
                        # i.e. v_curr_idx > u_curr_idx and v_next_idx < u_next_idx
                        if (v_curr_idx > u_curr_idx) and (v_next_idx <= u_next_idx):
                            overtake_counts[v] = overtake_counts.get(v, 0) + 1
        return overtake_counts
    
    def iterative_overtake_removal(df):
        removed_vehicles_history = []
        
        while True:
            overtake_counts = detect_overtakes(df)
            
            if not overtake_counts:
                # No overtakes found at all
                break
            
            max_overtakes = max(overtake_counts.values())
            if max_overtakes == 0:
                # No actual overtakes
                break
            
            # Find all vehicles that share the max overtake count
            worst_offenders = [v for v, cnt in overtake_counts.items() if cnt == max_overtakes]
            
            # Record who we are removing and how many overtakes they had
            for veh in worst_offenders:
                removed_vehicles_history.append((veh, max_overtakes))
            
            # Remove those vehicles from df
            df = df[~df["Vehicle_ID"].isin(worst_offenders)].copy()
            
        return df, removed_vehicles_history
    
    diffs = df.groupby('Vehicle_ID')['Global_Time'].max() - df.groupby('Vehicle_ID')['Global_Time'].min()
    mean_val = diffs.mean()
    std_val = diffs.std()
    three_sigma_val = mean_val - 3 * std_val

    valid_vehicle_ids = diffs[diffs >= three_sigma_val].index
    df = df[df["Vehicle_ID"].isin(valid_vehicle_ids)] 
    df.reset_index(drop=True, inplace=True)
    
    # Run the procedure
    final_df, removal_list = iterative_overtake_removal(df)

    # Sort by Vehicle_ID and time (important to ensure the "first" record is truly the first in time)
    df = final_df.sort_values(['Vehicle_ID', 'Global_Time']).reset_index(drop=True)
    
    return df

def CP_data_generation(df, PR, valid_range, Range=80, PLR=0, seed=42, time_based = True):
    
    random.seed(seed)
    np.random.seed(seed)
    
    df['CAV'] = 0
    df['Detected'] = 0
    Veh_list = list(set(df['Vehicle_ID']))
    travel_df = df.groupby('Vehicle_ID')['Local_Y'].agg(['min', 'max'])
    travel_df['range_local_y'] = travel_df['max'] - travel_df['min']
    valid_vehicles = travel_df[travel_df['range_local_y'] >= valid_range].index.tolist()

    # Select which vehicles will be CAV
    num_cav = int(len(Veh_list) * PR)

    if num_cav <= 0:
        # If no vehicles qualify or PR=0, skip the selection
        CAV_list = []
    else:
        if time_based:
            # --- Time-based selection ---
            # (a) Find earliest appearance for each valid vehicle
            first_appear = (
                df[df['Vehicle_ID'].isin(valid_vehicles)]
                .groupby('Vehicle_ID')['Global_Time']
                .min()
                .reset_index(name='Min_Time')
            )
            # (b) Sort those vehicles by earliest time
            first_appear_sorted = first_appear.sort_values('Min_Time')
            # (c) Pick vehicles at regular intervals in the sorted list
            step = len(valid_vehicles) / num_cav
            chosen_indices = [int(i * step) for i in range(num_cav)]
            CAV_list = first_appear_sorted.iloc[chosen_indices]['Vehicle_ID'].values
        else:
            # --- Random selection ---
            CAV_list = random.sample(valid_vehicles, num_cav)

    # Mark these vehicles as CAV
    df.loc[df['Vehicle_ID'].isin(CAV_list), 'CAV'] = 1

    for idx, cav in df[df['CAV'] == 1].iterrows():
        
        if random.random() <= PLR:
            df.at[idx, 'CAV'] = 0
            continue

        global_time = cav['Global_Time']
        df_temp = df[df['Global_Time'] == global_time]
        distances = np.sqrt(
            (df_temp['Global_X'] - cav['Global_X'])**2 
            + (df_temp['Global_Y'] - cav['Global_Y'])**2
        )
        
        # We only consider vehicles within the max detection Range
        in_range_mask = distances <= Range

        detection_probs = []
        for dist_val in distances:
            if dist_val < 10:
                detection_probs.append(0.95)
            elif 10 <= dist_val < 20:
                detection_probs.append(0.928)
            elif 20 <= dist_val < 30:
                detection_probs.append(0.816)
            elif 30 <= dist_val < 40:
                detection_probs.append(0.665)
            elif 40 <= dist_val < 50:
                detection_probs.append(0.489)
            elif 50 <= dist_val < Range:
                detection_probs.append(0.299)
            else:
                detection_probs.append(0)

        detection_probs = np.array(detection_probs)
        detect_success = (np.random.rand(len(detection_probs)) < detection_probs) & in_range_mask
        detect_success_int = detect_success.astype(int)

        # Update the original df with detection results at the corresponding indices
        # Note: df_temp and distances have the same index as df_temp
        df.loc[df_temp.index, 'Detected'] = (
            df.loc[df_temp.index, 'Detected'] | detect_success_int
        )

        # Ensure the CAV itself is detected = 1 for sure
        df.at[idx, 'Detected'] = 1
    return df

def CP_data_generation_occlusion_us101(df: pd.DataFrame,
                                PR: float,
                                valid_range: float,
                                Range: float = 80,
                                PLR: float = 0.0,
                                seed: int = 42,
                                time_based: bool = True) -> pd.DataFrame:

    random.seed(seed)
    np.random.seed(seed)

    df = df.copy()
    df['CAV'] = 0
    df['Detected'] = 0
    vehs = df['Vehicle_ID'].unique()
    travel = df.groupby('Vehicle_ID')['Local_Y'].agg(['min', 'max'])
    travel['range_local_y'] = travel['max'] - travel['min']
    valid_vehicles = travel[travel['range_local_y'] >= valid_range].index.tolist()

    # select CAVs
    num_cav = int(len(vehs) * PR)
    if num_cav > 0:
        if time_based:
            first_app = (
                df[df['Vehicle_ID'].isin(valid_vehicles)]
                  .groupby('Vehicle_ID')['Global_Time']
                  .min().reset_index(name='Min_Time')
            )
            first_app.sort_values('Min_Time', inplace=True)
            step = len(valid_vehicles) / num_cav
            picks = [int(i * step) for i in range(num_cav)]
            CAV_list = first_app.iloc[picks]['Vehicle_ID'].tolist()
        else:
            CAV_list = random.sample(valid_vehicles, num_cav)
    else:
        CAV_list = []

    # apply packet loss
    surviving = [vid for vid in CAV_list if random.random() > PLR]
    df.loc[df['Vehicle_ID'].isin(surviving), 'CAV'] = 1

    # helper for interval subtraction
    def subtract_intervals(intervals, rem):
        a, b = rem
        out = []
        for c, d in intervals:
            if b <= c or a >= d:
                out.append((c, d))
            else:
                if a > c: out.append((c, a))
                if b < d: out.append((b, d))
        return out

    # helper to test intersection
    def has_intersection(intervals, rem):
        a, b = rem
        return any(not (d <= a or c >= b) for c, d in intervals)

    # occlusion-based detection per CAV
    for idx, cav in df[df['CAV'] == 1].iterrows():
        t = cav['Global_Time']
        # neighbors excluding self
        frame = df[(df['Global_Time'] == t) & (df['Vehicle_ID'] != cav['Vehicle_ID'])].copy()

        # compute offsets and distances
        frame['dx'] = frame['Local_X'] - cav['Local_X']
        frame['dy'] = frame['Local_Y'] - cav['Local_Y']
        frame['distance'] = np.hypot(frame['dx'], frame['dy'])
        frame = frame[frame['distance'] <= Range]

        # quadrant labeling
        conds = [
            (frame['dx'] < 0) & (frame['dy'] > 0),  # Q1
            (frame['dx'] < 0) & (frame['dy'] < 0),  # Q2
            (frame['dx'] > 0) & (frame['dy'] < 0),  # Q3
            (frame['dx'] > 0) & (frame['dy'] > 0),  # Q4
        ]
        frame['quadrant'] = np.select(conds, [1, 2, 3, 4], default=np.nan)

        # vehicle half sizes
        w2 = frame['v_Width'] / 2.0
        l2 = frame['v_Length'] / 2.0 #'v_Length' for us101

        # corner coordinates based on quadrant
        xA1, yA1 = frame['dx'] + w2, frame['dy'] + l2
        xA2, yA2 = frame['dx'] - w2, frame['dy'] - l2
        xB1, yB1 = frame['dx'] - w2, frame['dy'] + l2
        xB2, yB2 = frame['dx'] + w2, frame['dy'] - l2
        is_A = frame['quadrant'].isin([1, 3])
        px1 = np.where(is_A, xA1, xB1)
        py1 = np.where(is_A, yA1, yB1)
        px2 = np.where(is_A, xA2, xB2)
        py2 = np.where(is_A, yA2, yB2)

        # compute occlusion angles
        ang1 = np.degrees(np.arctan2(px1, py1))
        ang2 = np.degrees(np.arctan2(px2, py2))
        frame['occl_min'] = (np.minimum(ang1, ang2) + 360) % 360
        frame['occl_max'] = (np.maximum(ang1, ang2) + 360) % 360

        # sort by distance
        frame.sort_values('distance', inplace=True)
        detectable_intervals = [(0.0, 360.0)]

        # CAV always detected
        df.at[idx, 'Detected'] = 1

        # iterate neighbors
        for _, row in frame.iterrows():
            vid = row['Vehicle_ID']
            lo, hi = row['occl_min'], row['occl_max']
            occ_ints = [(lo, hi)] if lo <= hi else [(lo, 360.0), (0.0, hi)]

            # check if any part is visible
            visible = any(has_intersection(detectable_intervals, oc) for oc in occ_ints)

            # set to detected if visible; never unset an already-detected vehicle
            if visible:
                df.loc[(df['Global_Time'] == t) & (df['Vehicle_ID'] == vid), 'Detected'] = 1
                # subtract occluded spans
                for oc in occ_ints:
                    detectable_intervals = subtract_intervals(detectable_intervals, oc)
                if not detectable_intervals:
                    break

    return df

def CP_data_generation_occlusion_lankershim(df: pd.DataFrame,
                                PR: float,
                                valid_range: float,
                                Range: float = 80,
                                PLR: float = 0.0,
                                seed: int = 42,
                                time_based: bool = True) -> pd.DataFrame:

    random.seed(seed)
    np.random.seed(seed)

    df = df.copy()
    df['CAV'] = 0
    df['Detected'] = 0
    vehs = df['Vehicle_ID'].unique()
    travel = df.groupby('Vehicle_ID')['Local_Y'].agg(['min', 'max'])
    travel['range_local_y'] = travel['max'] - travel['min']
    valid_vehicles = travel[travel['range_local_y'] >= valid_range].index.tolist()

    # select CAVs
    num_cav = int(len(vehs) * PR)
    if num_cav > 0:
        if time_based:
            first_app = (
                df[df['Vehicle_ID'].isin(valid_vehicles)]
                  .groupby('Vehicle_ID')['Global_Time']
                  .min().reset_index(name='Min_Time')
            )
            first_app.sort_values('Min_Time', inplace=True)
            step = len(valid_vehicles) / num_cav
            picks = [int(i * step) for i in range(num_cav)]
            CAV_list = first_app.iloc[picks]['Vehicle_ID'].tolist()
        else:
            CAV_list = random.sample(valid_vehicles, num_cav)
    else:
        CAV_list = []

    # apply packet loss
    surviving = [vid for vid in CAV_list if random.random() > PLR]
    df.loc[df['Vehicle_ID'].isin(surviving), 'CAV'] = 1

    # helper for interval subtraction
    def subtract_intervals(intervals, rem):
        a, b = rem
        out = []
        for c, d in intervals:
            if b <= c or a >= d:
                out.append((c, d))
            else:
                if a > c: out.append((c, a))
                if b < d: out.append((b, d))
        return out

    # helper to test intersection
    def has_intersection(intervals, rem):
        a, b = rem
        return any(not (d <= a or c >= b) for c, d in intervals)

    # occlusion-based detection per CAV
    for idx, cav in df[df['CAV'] == 1].iterrows():
        t = cav['Global_Time']
        # neighbors excluding self
        frame = df[(df['Global_Time'] == t) & (df['Vehicle_ID'] != cav['Vehicle_ID'])].copy()

        # compute offsets and distances
        frame['dx'] = frame['Local_X'] - cav['Local_X']
        frame['dy'] = frame['Local_Y'] - cav['Local_Y']
        frame['distance'] = np.hypot(frame['dx'], frame['dy'])
        frame = frame[frame['distance'] <= Range]

        # quadrant labeling
        conds = [
            (frame['dx'] < 0) & (frame['dy'] > 0),  # Q1
            (frame['dx'] < 0) & (frame['dy'] < 0),  # Q2
            (frame['dx'] > 0) & (frame['dy'] < 0),  # Q3
            (frame['dx'] > 0) & (frame['dy'] > 0),  # Q4
        ]
        frame['quadrant'] = np.select(conds, [1, 2, 3, 4], default=np.nan)

        # vehicle half sizes
        w2 = frame['v_Width'] / 2.0
        l2 = frame['v_length'] / 2.0 #'v_Length' for us101

        # corner coordinates based on quadrant
        xA1, yA1 = frame['dx'] + w2, frame['dy'] + l2
        xA2, yA2 = frame['dx'] - w2, frame['dy'] - l2
        xB1, yB1 = frame['dx'] - w2, frame['dy'] + l2
        xB2, yB2 = frame['dx'] + w2, frame['dy'] - l2
        is_A = frame['quadrant'].isin([1, 3])
        px1 = np.where(is_A, xA1, xB1)
        py1 = np.where(is_A, yA1, yB1)
        px2 = np.where(is_A, xA2, xB2)
        py2 = np.where(is_A, yA2, yB2)

        # compute occlusion angles
        ang1 = np.degrees(np.arctan2(px1, py1))
        ang2 = np.degrees(np.arctan2(px2, py2))
        frame['occl_min'] = (np.minimum(ang1, ang2) + 360) % 360
        frame['occl_max'] = (np.maximum(ang1, ang2) + 360) % 360

        # sort by distance
        frame.sort_values('distance', inplace=True)
        detectable_intervals = [(0.0, 360.0)]

        # CAV always detected
        df.at[idx, 'Detected'] = 1

        # iterate neighbors
        for _, row in frame.iterrows():
            vid = row['Vehicle_ID']
            lo, hi = row['occl_min'], row['occl_max']
            occ_ints = [(lo, hi)] if lo <= hi else [(lo, 360.0), (0.0, hi)]

            # check if any part is visible
            visible = any(has_intersection(detectable_intervals, oc) for oc in occ_ints)

            # set to detected if visible; never unset an already-detected vehicle
            if visible:
                df.loc[(df['Global_Time'] == t) & (df['Vehicle_ID'] == vid), 'Detected'] = 1
                # subtract occluded spans
                for oc in occ_ints:
                    detectable_intervals = subtract_intervals(detectable_intervals, oc)
                if not detectable_intervals:
                    break

    return df

def build_complete_trajectories_dict_highway(df):
    """
    Build a dict where each key is the same as time_dict's key (e.g. 1, 2, 3...),
    and each value is a dictionary of {vehicle_id: subset_of_df_for_that_vehicle}
    for all vehicles whose entire trajectory is contained within the time range
    specified by time_dict[key] = [start_time, end_time].
    """
    
    def build_cav_time_dict(df):
        """
        Build a dictionary where:
          key: An integer (starting from 1)
          value: [enter_time_of_current_CAV, exit_time_of_next_CAV]
        
        Assumptions:
          - 'CAV' is 1 for those vehicles designated as CAV.
          - 'Vehicle_ID' identifies vehicles.
          - 'Global_Time' is the time column used for 'enter' and 'exit'.
        """

        df_cav = df[df['CAV'] == 1].copy()
        cav_ids = df_cav['Vehicle_ID'].unique()
        df_cav_whole = df[df['Vehicle_ID'].isin(cav_ids)].copy()
        # Store tuples of the form: (cav_id, enter_time, exit_time)
        cav_enter_exit = []
        for cid in cav_ids:
            df_cid = df_cav_whole[df_cav_whole['Vehicle_ID'] == cid]
            enter_time = df_cid['Global_Time'].min()
            exit_time  = df_cid['Global_Time'].max()
            cav_enter_exit.append((cid, enter_time, exit_time))
        cav_enter_exit.sort(key=lambda x: x[1])  # sort by enter_time

        # my_dict[i] = [enter_time_of_i, exit_time_of_(i+1)]
        # Skip the last CAV because it has no "next" one
        time_dict = {}
        for i in range(len(cav_enter_exit) - 1):
            # i-th CAV is cav_enter_exit[i]
            # next CAV is cav_enter_exit[i+1]
            curr_enter = cav_enter_exit[i][1]
            next_enter  = cav_enter_exit[i+1][1]
            time_dict[i + 1] = [curr_enter, next_enter]

        return time_dict
    
    time_dict = build_cav_time_dict(df)

    # Compute the earliest and latest time that vehicle appears)
    grouped = df.groupby('Vehicle_ID')['Global_Time']
    min_times = grouped.min()
    # Initialize the output dictionary
    complete_trajectories_dict = {}
    # Iterate over time_dict
    for k, (start_time, end_time) in time_dict.items():
        # Find vehicles whose entire min->max time is fully in [start_time, end_time]
        # min_time >= start_time AND max_time <= end_time
        valid_vehicles = min_times[
            (min_times >= start_time) & (min_times <= end_time)
        ].index
        # Store in the output dictionary under key k
        complete_trajectories_dict[k] = df[df['Vehicle_ID'].isin(valid_vehicles)].copy()

    return complete_trajectories_dict

def build_complete_trajectories_dict_intersection(df):
    """
    Build a dict where each key is the same as time_dict's key (e.g. 1, 2, 3...),
    and each value is a dictionary of {vehicle_id: subset_of_df_for_that_vehicle}
    for all vehicles whose entire trajectory is contained within the time range
    specified by time_dict[key] = [start_time, end_time].
    """
    
    def build_cav_time_dict(df):
        min_T = min(df.Global_Time)

        allowed_lanes = {1, 2, 3}
        lane_check = df.groupby('Vehicle_ID')['Lane_ID'].apply(lambda x: x.isin(allowed_lanes).all())
        
        # Filter df to keep only vehicles that pass the lane check
        valid_vehicle_ids = lane_check[lane_check].index  # vehicles that stay in lanes 1, 2, 3
        df_lane_filtered = df[df['Vehicle_ID'].isin(valid_vehicle_ids)]
        
        min_speeds = df_lane_filtered.groupby('Vehicle_ID')['v_Vel'].min()
        fast_vehicle_ids = min_speeds[min_speeds > 3].index
        
        df_speed_filtered = df_lane_filtered[df_lane_filtered['Vehicle_ID'].isin(fast_vehicle_ids)]
        earliest_times = df_speed_filtered.groupby('Vehicle_ID')['Global_Time'].min().reset_index()
        earliest_times.rename(columns={'Global_Time': 'min_Global_Time'}, inplace=True)
        earliest_times = earliest_times.sort_values(by='min_Global_Time', ascending=True)
        
        earliest_times['time_diff'] = earliest_times['min_Global_Time'].diff()

        large_gaps = earliest_times[earliest_times['time_diff'] > 40]['min_Global_Time']
        
        time_dict = {}
        for i in range(len(large_gaps)):
            if i == 0:
                time_dict[i + 1] = [min_T, large_gaps.iloc[i]]
            else:
                time_dict[i + 1] = [large_gaps.iloc[i-1], large_gaps.iloc[i]]
                
        return time_dict
    
    time_dict = build_cav_time_dict(df)

    grouped = df.groupby('Vehicle_ID')['Global_Time']
    min_times = grouped.min()
    complete_trajectories_dict = {}

    # Iterate over time_dict
    for k, (start_time, end_time) in time_dict.items():
        
        # Find vehicles whose entire min->max time is fully in [start_time, end_time]
        valid_vehicles = min_times[
            (min_times >= start_time) & (min_times < end_time)
        ].index
        
        # Store in the output dictionary under key k
        complete_trajectories_dict[k] = df[df['Vehicle_ID'].isin(valid_vehicles)].copy()

    return complete_trajectories_dict
    
def smooth_trajectories(
    df,
    vehicle_id_col='Vehicle_ID',    # input DataFrame containing trajectory data.
    time_col='Global_Time',         # column name for Vehicle ID.
    position_col='Local_Y',         # column name for the timestamp or frame.
    speed_col='v_Vel',              # column name for the longitudinal position (Local_Y).
    acc_col='v_Acc',                # column name for the speed (v_Vel).
    max_acc=3.5,                    # maximum allowable acceleration (units consistent with your data) before clipping.
    max_jerk=3,                     # maximum allowable jerk (units consistent with your data) before clipping.
    window_length=7,                # window length for Savitzky-Golay filter (must be odd).
    polyorder=3,                    # polynomial order for Savitzky-Golay filter, balances smoothness with sharper changes
    dt=1                            # time step between consecutive frames (seconds).
):
    """
    Smooth vehicle trajectories using a Savitzky-Golay filter to reduce noise,
    then compute and clip extreme acceleration and jerk.
    """

    # ------------------------------------------------------------------
    # 1) Fix backward movement by overwriting backward positions.
    #    Also sets speed and acceleration to zero where the fix happens.
    # ------------------------------------------------------------------
    def fix_backward_movement(group, position_col='Local_Y', speed_col='v_Vel', acc_col='v_Acc'):
        """
        Walk backwards through the group's position data.
        If position[i] > position[i+1], overwrite position[i] = position[i+1].
        Also set speed and acceleration to 0 at that index.
        """
        positions = group[position_col].values.copy()
        speeds = group[speed_col].values.copy()
        accs = group[acc_col].values.copy()
        
        # Traverse from second-to-last down to the first index
        for i in range(len(positions) - 2, -1, -1):
            if positions[i] > positions[i + 1]:
                positions[i] = positions[i + 1]  # fix position
                speeds[i] = 0
                accs[i]   = 0
        
        group[position_col] = positions
        group[speed_col]    = speeds
        group[acc_col]      = accs
        return group
    
    # Ensure DataFrame is sorted
    df = df.sort_values(by=[vehicle_id_col, time_col]).reset_index(drop=True)

    # Fix backward movement group-wise
    df = df.groupby(vehicle_id_col, group_keys=False).apply(
        fix_backward_movement
    )
    
    # ------------------------------------------------------------------
    # 2) Smooth each vehicle's trajectory with Savitzky-Golay
    #    and clip acceleration + jerk.
    # ------------------------------------------------------------------
    smoothed_dfs = []
    
    for vid, group in df.groupby(vehicle_id_col):
        group = group.sort_values(by=time_col).reset_index(drop=True)
        
        # 2a) Smooth the position
        y_raw = group[position_col].values
        effective_window_length = min(len(group), window_length)
        # Make window_length odd and at least 3
        if effective_window_length < 3:
            y_smooth = y_raw
        else:
            if effective_window_length % 2 == 0:
                effective_window_length -= 1
            if polyorder >= effective_window_length:
                effective_window_length = polyorder + 1
                if len(y_raw) < effective_window_length:
                    y_smooth = y_raw
                else:
                    y_smooth = savgol_filter(y_raw, window_length=effective_window_length, polyorder=polyorder)
            else:
                y_smooth = savgol_filter(y_raw, window_length=effective_window_length, polyorder=polyorder)
        
        # 2b) Compute velocity and acceleration from the smoothed position
        #     v(t) ≈ dy/dt, a(t) ≈ dv/dt
        v_smooth = np.gradient(y_smooth, dt)
        a_smooth = np.gradient(v_smooth, dt)

        # 2c) Clip acceleration to ±max_acc
        a_clipped = np.clip(a_smooth, -max_acc, max_acc)

        # 2d) Clip jerk to ±max_jerk
        for i in range(1, len(a_clipped)):
            local_jerk = (a_clipped[i] - a_clipped[i-1]) / dt
            
            if local_jerk > max_jerk:
                # Reduce acceleration so jerk = max_jerk
                a_clipped[i] = a_clipped[i-1] + max_jerk * dt
            elif local_jerk < -max_jerk:
                # Increase acceleration so jerk = -max_jerk
                a_clipped[i] = a_clipped[i-1] - max_jerk * dt
        
        # 2e) (Optional) Reintegrate velocity from the clipped acceleration 
        #     to ensure consistency. Below is a simple forward Euler approach.
        #     That said, it can introduce drift or steps, so use with caution.
        v_final = np.zeros_like(v_smooth)
        v_final[0] = v_smooth[0]  # keep the initial velocity
        for i in range(1, len(v_final)):
            v_final[i] = v_final[i-1] + a_clipped[i-1] * dt
        
        # Store results in the group
        group['Local_Y'] = y_smooth
        group['v_Vel']   = v_final
        group['v_Acc']   = a_clipped
        
        smoothed_dfs.append(group)

    # Combine results
    smoothed_df = pd.concat(smoothed_dfs, axis=0)
    smoothed_df = smoothed_df.sort_values(by=[vehicle_id_col, time_col]).reset_index(drop=True)
    smoothed_df = smoothed_df.groupby(vehicle_id_col, group_keys=False).apply(
        fix_backward_movement
    )

    return smoothed_df