'''
  9.  Calculate a distance matrix based on the dataframe, df.
    Args:
        df (pandas.DataFrame)
    Returns:
        pandas.DataFrame: Distance matrix
'''
import pandas as pd
import numpy as np

def calculate_distance_matrix(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Extract unique toll IDs
    toll_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    distance_matrix = pd.DataFrame(index=toll_ids, columns=toll_ids, data=np.inf)
    
    # Set diagonal to 0 (distance from a location to itself is 0)
    np.fill_diagonal(distance_matrix.values, 0)
    for _, row in df.iterrows():
        loc_a = row['id_start']
        loc_b = row['id_end']
        distance = row['distance']
        
        distance_matrix.loc[loc_a, loc_b] = distance
        distance_matrix.loc[loc_b, loc_a] = distance
    
    for k in toll_ids:
        for i in toll_ids:
            for j in toll_ids:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]
    
    return distance_matrix
distance_matrix = calculate_distance_matrix('datasets\dataset-2.csv')
print(distance_matrix)

'''
#Output
1001400  1001402  1001404  1001406  1001408  1001410  1001412  1001414  ...  1001461  1001462  1001464  1001466  1001468  1001470  1001437  1001472
1001400      0.0      9.7     29.9     45.9     67.6     78.7     94.3    112.5  ...    366.7    371.8    398.5    407.0    417.7    428.3    242.1    444.3
1001402      9.7      0.0     20.2     36.2     57.9     69.0     84.6    102.8  ...    357.0    362.1    388.8    397.3    408.0    418.6    232.4    434.6
1001404     29.9     20.2      0.0     16.0     37.7     48.8     64.4     82.6  ...    336.8    341.9    368.6    377.1    387.8    398.4    212.2    414.4
1001406     45.9     36.2     16.0      0.0     21.7     32.8     48.4     66.6  ...    320.8    325.9    352.6    361.1    371.8    382.4    196.2    398.4
1001408     67.6     57.9     37.7     21.7      0.0     11.1     26.7     44.9  ...    299.1    304.2    330.9    339.4    350.1    360.7    174.5    376.7   
'''

'''
  10.  Unroll a distance matrix to a DataFrame in the style of the initial dataset.
    Args:
        df (pandas.DataFrame)
    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
        '''
import pandas as pd
def unroll_distance_matrix(distance_matrix):
    # Create an empty list to store the rows of the unrolled DataFrame
    unrolled_data = []
    # Iterate over the DataFrame to unroll it into id_start, id_end, and distance
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude same id_start to id_end
                distance = distance_matrix.loc[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    unrolled_df = pd.DataFrame(unrolled_data)
    return unrolled_df

unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)
'''
#Output
[43 rows x 43 columns]
      id_start   id_end  distance
0      1001400  1001402       9.7
1      1001400  1001404      29.9
2      1001400  1001406      45.9
3      1001400  1001408      67.6
4      1001400  1001410      78.7
...        ...      ...       ...
1801   1001472  1001464      45.8
1802   1001472  1001466      37.3
1803   1001472  1001468      26.6
1804   1001472  1001470      16.0
1805   1001472  1001437     202.2

[1806 rows x 3 columns]
'''

'''
  11. Find all IDs whose average distance lies within 10% of the average distance of the reference ID.
    Args:
        df (pandas.DataFrame)
        reference_id (int)
    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold of the reference ID's average distance. 
'''
import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Step 1: Calculate the average distance for the reference id_start
    reference_distances = df[df['id_start'] == reference_id]['distance']
    reference_avg = reference_distances.mean()
    # Step 2: Calculate 10% threshold (both floor and ceiling)
    threshold_floor = reference_avg * 0.9
    threshold_ceiling = reference_avg * 1.1
    # Step 3: Find id_start values whose average distance falls within the 10% threshold
    ids_within_threshold = []
    for id_start in df['id_start'].unique():
        avg_distance = df[df['id_start'] == id_start]['distance'].mean()
        if threshold_floor <= avg_distance <= threshold_ceiling:
            ids_within_threshold.append(id_start)
    # Step 4: Return the sorted list of id_start values
    return sorted(ids_within_threshold)

sorted_ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, 1001400)
print(sorted_ids_within_threshold)

'''Output '''
[1001400, 1001402]


'''
  12.  Calculate toll rates for each vehicle type based on the unrolled DataFrame.
    Args:
        df (pandas.DataFrame)
    Returns:
        pandas.DataFrame'''

import pandas as pd

def calculate_toll_rate(df):
    # Define the rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    # Calculate toll rates for each vehicle type and add new columns to the DataFrame
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate
    
    return df

toll_rate_df = calculate_toll_rate(unrolled_df)
print(toll_rate_df)
'''
#Output
   id_start   id_end  distance    moto     car      rv     bus   truck
0      1001400  1001402       9.7    7.76   11.64   14.55   21.34   34.92
1      1001400  1001404      29.9   23.92   35.88   44.85   65.78  107.64
2      1001400  1001406      45.9   36.72   55.08   68.85  100.98  165.24
3      1001400  1001408      67.6   54.08   81.12  101.40  148.72  243.36
4      1001400  1001410      78.7   62.96   94.44  118.05  173.14  283.32
...        ...      ...       ...     ...     ...     ...     ...     ...
1801   1001472  1001464      45.8   36.64   54.96   68.70  100.76  164.88
1802   1001472  1001466      37.3   29.84   44.76   55.95   82.06  134.28
1803   1001472  1001468      26.6   21.28   31.92   39.90   58.52   95.76
1804   1001472  1001470      16.0   12.80   19.20   24.00   35.20   57.60
1805   1001472  1001437     202.2  161.76  242.64  303.30  444.84  727.92
'''


'''
  13.  Calculate time-based toll rates for different time intervals within a day.
    Args:
        df (pandas.DataFrame)
    Returns:
        pandas.DataFrame
'''
import pandas as pd
import numpy as np
from datetime import time, timedelta

def calculate_time_based_toll_rates(df):
    # Create lists for days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    #time intervals and their discount factors for weekdays
    weekday_intervals = [
        (time(0, 0), time(10, 0), 0.8),      # 00:00 to 10:00
        (time(10, 0), time(18, 0), 1.2),     # 10:00 to 18:00
        (time(18, 0), time(23, 59, 59), 0.8) # 18:00 to 23:59
    ]
    # Define a discount factor for weekends
    weekend_discount = 0.7
    new_rows = []
    # Iterate through unique (id_start, id_end) pairs
    for (id_start, id_end), group in df.groupby(['id_start', 'id_end']):
        for day in days_of_week:
            for hour in range(24):  # For each hour in a day
                # Start and end time for the current hour
                start_time = time(hour, 0)
                end_time = time(hour, 59, 59)
                # Initialize the toll rates for the current row
                toll_rates = {vehicle: 0 for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
                # Determine the discount factor based on the day of the week
                if day in days_of_week[:5]:  # Weekdays
                    for start, end, factor in weekday_intervals:
                        if start_time >= start and end_time <= end:
                            discount_factor = factor
                            break
                    else:
                        discount_factor = 1  # Default if no interval matches
                else:  # Weekends
                    discount_factor = weekend_discount
                # Calculate the adjusted toll rates
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    original_rate = group[vehicle].mean()  # Use the average rate from the group
                    toll_rates[vehicle] = original_rate * discount_factor
                
                # Create a new row with calculated values
                new_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **toll_rates  # Unpack the toll rates into the row
                }
                new_rows.append(new_row)
    # Convert the new rows into a DataFrame
    time_based_toll_df = pd.DataFrame(new_rows)

    return time_based_toll_df
time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)
print(time_based_toll_df)
'''
# Output
     id_start   id_end start_day start_time end_day  end_time   moto    car     rv     bus   truck
0        1001400  1001402    Monday   00:00:00  Monday  00:59:59  6.208  9.312  11.64  17.072  27.936
1        1001400  1001402    Monday   01:00:00  Monday  01:59:59  6.208  9.312  11.64  17.072  27.936
2        1001400  1001402    Monday   02:00:00  Monday  02:59:59  6.208  9.312  11.64  17.072  27.936
3        1001400  1001402    Monday   03:00:00  Monday  03:59:59  6.208  9.312  11.64  17.072  27.936
4        1001400  1001402    Monday   04:00:00  Monday  04:59:59  6.208  9.312  11.64  17.072  27.936
...          ...      ...       ...        ...     ...       ...    ...    ...    ...     ...     ...
303403   1004356  1004355    Sunday   19:00:00  Sunday  19:59:59  2.240  3.360   4.20   6.160  10.080
303404   1004356  1004355    Sunday   20:00:00  Sunday  20:59:59  2.240  3.360   4.20   6.160  10.080
303405   1004356  1004355    Sunday   21:00:00  Sunday  21:59:59  2.240  3.360   4.20   6.160  10.080
303406   1004356  1004355    Sunday   22:00:00  Sunday  22:59:59  2.240  3.360   4.20   6.160  10.080
303407   1004356  1004355    Sunday   23:00:00  Sunday  23:59:59  2.240  3.360   4.20   6.160  10.080
'''