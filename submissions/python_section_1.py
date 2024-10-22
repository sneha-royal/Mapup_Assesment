'''1.Reverses the input list by group of n elements.'''
def reverse_by_n_elements(lst, n):

    for i in range(0, len(lst), n):
        left = i
        right = min(i + n - 1, len(lst) - 1)
        
        # Reverse elements in the current group of n elements
        while left < right:
            # Swap the elements
            lst[left], lst[right] = lst[right], lst[left]
            left += 1
            right -= 1
    
    return lst
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))            # Output: [2, 1, 4, 3, 5]
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]


'''2.Groups the strings by their length and returns a dictionary.'''
def group_strings_by_length(lst):
    # an empty dictionary to hold the groups
    length_dict = {}
    for string in lst:
        length = len(string)  # Get the length of the string
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))


print(group_strings_by_length(["apple", "bat", "car", "elephant", "dog", "bear"])) 
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}
print(group_strings_by_length(["one", "two", "three", "four"]))  
# Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}

"""
  3.  Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    
    for k, v in d.items():
        # Create new key by appending current key to parent key
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Recursively flatten dictionaries
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Iterate through list and flatten each element with index
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    # Recursively flatten if list item is a dictionary
                    items.extend(flatten_dict(item, list_key, sep=sep).items())
                else:
                    # Add non-dictionary items in the list directly
                    items.append((list_key, item))
        else:
            # Add non-dictionary, non-list item to the result
            items.append((new_key, v))
    
    return dict(items)
data = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}
print(flatten_dict(data))
# Output
{'road.name': 'Highway 1', 'road.length': 350, 'road.sections[0].id': 1, 'road.sections[0].condition.pavement': 'good', 'road.sections[0].condition.traffic': 'moderate'}

"""
4.    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """

from itertools import permutations

def unique_permutations(nums):
    # Generate all permutations using itertools
    all_perms = permutations(nums)
    # Use a set to filter out duplicates and convert back to a list
    unique_perms = set(all_perms)
    return [list(p) for p in unique_perms]
print(unique_permutations([1, 1, 2])) 
# Output: [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
print(unique_permutations([1, 2, 3]))
# Output: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
print(unique_permutations([1, 1, 1]))
# Output: [[1, 1, 1]]

"""
  5.  This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
import re
def find_all_dates(text):
    # Define the regex patterns for the date formats
    patterns = (
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    )  
    combined_pattern = '|'.join(patterns)
    valid_dates = re.findall(combined_pattern, text)
    return valid_dates

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)


"""
  6.  Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    Args:
        polyline_str (str): The encoded polyline string.
    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
import polyline
import pandas as pd
import numpy as np

def haversine(coord1, coord2):
    # Haversine formula to calculate the distance between two points on the Earth
    R = 6371000  # Radius of the Earth in meters
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c  # Distance in meters
    return distance

def decode_polyline_to_dataframe(polyline_str):
    # Decode the polyline string into a list of (latitude, longitude) coordinates
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Calculate distances between successive points
    distances = [0]  # First distance is 0
    for i in range(1, len(coordinates)):
        distance = haversine(coordinates[i - 1], coordinates[i])
        distances.append(distance)

    # Add distances to the DataFrame
    df['distance'] = distances

    return df

# Example usage
polyline_str = "u{~vFf|y@`@Fj@b@|@b@~@F|@t@d@b@a@d@b@d@"
output_df = decode_polyline_to_dataframe(polyline_str)
print(output_df)

# Output

#   latitude  longitude   distance
# 0  40.63179   -0.30164   0.000000
# 1  40.63162   -0.30168  19.202148
# 2  40.63140   -0.30186  28.795141
# 3  40.63109   -0.30204  37.668815
# 4  40.63077   -0.30208  35.742127
# 5  40.63046   -0.30235  41.320226
# 6  40.63027   -0.30253  26.020896
# 7  40.63044   -0.30272  24.787365
# 8  40.63026   -0.30291  25.645440

"""
 7.    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
def rotate_and_multiply_matrix(matrix):
    n = len(matrix)  # Size of the n x n matrix

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]  # Create an empty n x n matrix
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Calculate row and column sums
    row_sums = [sum(row) for row in rotated_matrix]
    col_sums = [sum(rotated_matrix[i][j] for i in range(n)) for j in range(n)]

    # Step 3: Create the final transformed matrix
    final_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # Replace each element with the sum of its row and column, excluding itself
            final_matrix[i][j] = row_sums[i] + col_sums[j] - rotated_matrix[i][j]

    return final_matrix

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
output_matrix = rotate_and_multiply_matrix(matrix)
print(output_matrix)


"""
 8.  Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period
    Args:
        df (pandas.DataFrame)
    Returns:
        pd.Series: return a boolean series
    """

import pandas as pd

def verify_timestamp_completeness(df):
    # Step 1: Combine start and end times into a single timestamp
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Step 2: Create a multi-index DataFrame based on (id, id_2)
    grouped = df.groupby(['id', 'id_2'])
    
    # Initialize a dictionary to store results
    results = {}
    
    # Step 3: Check conditions for each (id, id_2) pair
    for (id_value, id_2_value), group in grouped:
        # Check if timestamps cover all 7 days of the week
        days_covered = group['start_timestamp'].dt.day_name().unique()
        all_days = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Check if timestamps cover a full 24-hour period
        min_time = group['start_timestamp'].min().time()
        max_time = group['end_timestamp'].max().time()
        full_day_covered = (min_time == pd.Timestamp('00:00:00').time() and 
                            max_time == pd.Timestamp('23:59:59').time())
        
        # Check if all days are covered
        has_full_days = len(days_covered) == 7
        
        # Store the result (False if correct, True if incorrect)
        results[(id_value, id_2_value)] = not (has_full_days and full_day_covered)

    # Step 4: Convert results to a boolean series with multi-index
    result_series = pd.Series(results, dtype=bool)
    return result_series

# Example usage
# Load the dataset
df = pd.read_csv('datasets\dataset-1.csv')
# Check for timestamp completeness
boolean_series = verify_timestamp_completeness(df)
print(boolean_series)
