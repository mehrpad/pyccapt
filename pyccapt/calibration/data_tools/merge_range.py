import numpy as np


def merge_by_range(data_df, range_df, full=False):
    """
    Optimized merging function based on the 'mc' column value falling within the 'mc_low' and 'mc_up' range.
    Uses vectorized operations for performance.

    Parameters:
        data_df (pd.DataFrame): The dataframe containing the data to be merged.
        range_df (pd.DataFrame): The dataframe containing the range values 'mc_low' and 'mc_up'.
        full (bool): If True, the merged dataframe will contain all columns from the range_df. Default is False.

    Returns:
        pd.DataFrame: The merged dataframe with the range data attached.
    """
    # Prepare the necessary columns for merging
    data_mc = data_df['mc (Da)']

    # Use broadcasting to create masks for matching conditions
    mask = (range_df['mc_low'].values[:, None] <= data_mc.values) & (range_df['mc_up'].values[:, None] >= data_mc.values)

    # Find the matching range index for each data row (max mask index per row)
    matched_idx = mask.argmax(axis=0)  # For each data point, find the index of the matching range in range_df

    # Check if a valid match exists (mask is not empty for the given row)
    valid_matches = mask[matched_idx, np.arange(len(data_mc))]

    # Create merged dataframe
    merged_df = data_df.copy()

    # Default values for no matches
    default_values = {
        'name': np.nan,
        'ion': np.nan,
        'mass': np.nan,
        'mc': np.nan,
        'mc_low': np.nan,
        'mc_up': np.nan,
        'color': 'black',
        'element': ['noise'],
        'complex': [np.nan],
        'isotope': [np.nan],
        'charge': np.nan,
    }
    if full:
        # For valid matches, update with the corresponding values
        for col in ['name', 'ion', 'mass', 'mc', 'mc_low', 'mc_up', 'color', 'element', 'complex', 'isotope', 'charge']:
            merged_df[col] = np.where(valid_matches, range_df[col].values[matched_idx], default_values[col])
    else:
        # only add the columns name and ion
        for col in ['name', 'ion']:
            merged_df[col] = np.where(valid_matches, range_df[col].values[matched_idx], default_values[col])

    return merged_df
