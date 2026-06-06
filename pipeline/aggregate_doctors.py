"""
==============================================================================
DOCTOR DATA AGGREGATION SCRIPT
==============================================================================

PURPOSE:
--------
This script converts weekly time-series data (86 weeks per doctor) into 
a single comprehensive profile per doctor.

INPUT:  data/raw/data_processed.csv (1,800,066 rows)
OUTPUT: data/processed/doctors_aggregated.csv (20,931 rows)

AUTHOR: Pfizer Data Science Team
DATE: March 2026

==============================================================================
"""

import pandas as pd
import numpy as np
import os

# ==============================================================================
# PROJECT PATHS
# ==============================================================================

# Base directories
# repo root = parent of this pipeline/ folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# File paths
INPUT_FILE = os.path.join(RAW_DATA_DIR, 'data_processed.csv')
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'doctors_aggregated.csv')


# ==============================================================================
# STEP 1: LOAD THE DATA
# ==============================================================================

def load_data(filepath):
    """
    Load the weekly HCP data
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file with weekly data
        
    Returns:
    --------
    pd.DataFrame : Loaded data
    """
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(filepath)
    
    print(f"✓ Loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"✓ Unique doctors: {df['NUEVO_ID'].nunique():,}")
    print(f"✓ Average rows per doctor: {len(df) / df['NUEVO_ID'].nunique():.1f}")
    
    return df


# ==============================================================================
# STEP 2: DEFINE AGGREGATION RULES
# ==============================================================================

def create_aggregation_rules(df):
    """
    Define how each column should be aggregated
    
    This is the CORE of the aggregation process.
    Different types of metrics need different aggregation methods.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe
        
    Returns:
    --------
    dict : Dictionary mapping column names to aggregation functions
    """
    
    print("\n" + "="*80)
    print("DEFINING AGGREGATION RULES")
    print("="*80)
    
    agg_rules = {}
    
    # -------------------------------------------------------------------------
    # RULE 1: PRESCRIPTION VOLUMES
    # -------------------------------------------------------------------------
    # Columns: UC_TRX, ORAL_TRX, IL23_TRX, BRAND1_TRX, etc.
    # Aggregation: SUM (total prescriptions) + MEAN (average) + MAX (peak)
    # -------------------------------------------------------------------------
    
    prescription_cols = [col for col in df.columns 
                        if any(x in col for x in ['_TRX', '_NRX', '_NBRX', 'N_CLM'])]
    
    print(f"\n1. PRESCRIPTION COLUMNS ({len(prescription_cols)})")
    print(f"   Aggregation: Sum + Mean + Max")
    print(f"   Examples: {prescription_cols[:3]}")
    print(f"   Logic: We want TOTAL prescriptions over all weeks")
    
    for col in prescription_cols:
        agg_rules[col] = ['sum', 'mean', 'max']
    
    
    # -------------------------------------------------------------------------
    # RULE 2: ENGAGEMENT METRICS
    # -------------------------------------------------------------------------
    # Columns: RTE, SAMPLES, COPAY, DIRECTMAIL, SPK, DETAILS
    # Aggregation: SUM (total activities) + MEAN (average per week)
    # -------------------------------------------------------------------------
    
    engagement_cols = ['RTE', 'SAMPLES', 'COPAY', 'DIRECTMAIL', 'SPK', 
                      'DETAILS', 'ENGAGEMENT_SCORE']
    engagement_cols = [c for c in engagement_cols if c in df.columns]
    
    print(f"\n2. ENGAGEMENT COLUMNS ({len(engagement_cols)})")
    print(f"   Aggregation: Sum + Mean")
    print(f"   Columns: {engagement_cols}")
    print(f"   Logic: Total engagement activities over time")
    
    for col in engagement_cols:
        agg_rules[col] = ['sum', 'mean']
    
    
    # -------------------------------------------------------------------------
    # RULE 3: INDICES AND RATIOS
    # -------------------------------------------------------------------------
    # Columns: BRAND1_NTB_GIDX, NBRx_RATIO, BRAND1_MARKET_SHARE
    # Aggregation: MEAN (typical performance) + MAX (peak performance)
    # -------------------------------------------------------------------------
    
    index_cols = [col for col in df.columns 
                 if any(x in col for x in ['GIDX', 'RATIO', 'SHARE'])]
    
    print(f"\n3. INDEX/RATIO COLUMNS ({len(index_cols)})")
    print(f"   Aggregation: Mean + Max")
    print(f"   Columns: {index_cols}")
    print(f"   Logic: These are already percentages, so we average them")
    
    for col in index_cols:
        agg_rules[col] = ['mean', 'max']
    
    
    # -------------------------------------------------------------------------
    # RULE 4: ROLLING SUMS
    # -------------------------------------------------------------------------
    # Columns: UC_TRX_R4_16SUM, ORAL_NBRX_R4_29SUM
    # Aggregation: MAX (peak period) + MEAN (typical period)
    # -------------------------------------------------------------------------
    
    rolling_cols = [col for col in df.columns 
                   if 'SUM' in col and col not in prescription_cols]
    
    print(f"\n4. ROLLING SUM COLUMNS ({len(rolling_cols)})")
    print(f"   Aggregation: Max + Mean")
    print(f"   Columns: {rolling_cols}")
    print(f"   Logic: Peak performance in rolling windows")
    
    for col in rolling_cols:
        agg_rules[col] = ['max', 'mean']
    
    
    # -------------------------------------------------------------------------
    # RULE 5: DEMOGRAPHIC/CATEGORICAL COLUMNS
    # -------------------------------------------------------------------------
    # Columns: SPEC_GE, STATE_1, (1960, 1980], ATSEG
    # Aggregation: FIRST (they're constant over time)
    # -------------------------------------------------------------------------
    
    demo_cols = [col for col in df.columns 
                if any(x in col for x in ['SPEC_', 'STATE_', 'STS_', 'ATSEG', 
                                          '(', ']'])]
    
    print(f"\n5. DEMOGRAPHIC COLUMNS ({len(demo_cols)})")
    print(f"   Aggregation: First (constant value)")
    print(f"   Examples: {demo_cols[:5]}")
    print(f"   Logic: Doctor characteristics don't change over time")
    
    for col in demo_cols:
        agg_rules[col] = 'first'
    
    
    # -------------------------------------------------------------------------
    # RULE 6: TIME TRACKING
    # -------------------------------------------------------------------------
    # Column: WEEK_ID
    # Aggregation: FIRST (start) + LAST (end) + COUNT (# of weeks)
    # -------------------------------------------------------------------------
    
    if 'WEEK_ID' in df.columns:
        agg_rules['WEEK_ID'] = ['first', 'last', 'count']
        print(f"\n6. TIME COLUMN")
        print(f"   Aggregation: First + Last + Count")
        print(f"   Logic: Track data coverage period")
    
    
    print(f"\n{'='*80}")
    print(f"TOTAL COLUMNS TO AGGREGATE: {len(agg_rules)}")
    print(f"{'='*80}")
    
    return agg_rules


# ==============================================================================
# STEP 3: PERFORM AGGREGATION
# ==============================================================================

def aggregate_by_doctor(df, agg_rules):
    """
    Group data by doctor and apply aggregation rules
    
    This is where the magic happens!
    We go from 86 rows per doctor to 1 row per doctor.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with multiple weeks per doctor
    agg_rules : dict
        Aggregation rules from create_aggregation_rules()
        
    Returns:
    --------
    pd.DataFrame : Aggregated data with one row per doctor
    """
    
    print("\n" + "="*80)
    print("PERFORMING AGGREGATION")
    print("="*80)
    
    print("\nGrouping by NUEVO_ID (doctor) and applying aggregation rules...")
    
    # This is the key operation: groupby + aggregate
    aggregated = df.groupby('NUEVO_ID').agg(agg_rules)
    
    print("✓ Aggregation complete")
    
    # Flatten multi-level column names
    # Example: ('UC_TRX', 'sum') becomes 'UC_TRX_sum'
    print("✓ Flattening column names...")
    
    if isinstance(aggregated.columns, pd.MultiIndex):
        aggregated.columns = ['_'.join(col).strip() if isinstance(col, tuple) 
                             else col 
                             for col in aggregated.columns.values]
    
    # Reset index to make NUEVO_ID a regular column
    aggregated.reset_index(inplace=True)
    
    # Handle missing values
    print("✓ Handling missing values (filling with 0)...")
    aggregated = aggregated.fillna(0)
    
    print("\n" + "="*80)
    print("AGGREGATION RESULTS")
    print("="*80)
    print(f"Original:   {len(df):,} rows × {len(df.columns)} columns")
    print(f"Aggregated: {len(aggregated):,} rows × {len(aggregated.columns)} columns")
    print(f"\nReduction:  {len(df)/len(aggregated):.1f}x fewer rows")
    print(f"Result:     Each doctor now has exactly 1 row")
    
    return aggregated


# ==============================================================================
# STEP 4: SAVE RESULTS
# ==============================================================================

def save_aggregated_data(aggregated_df, output_filepath):
    """
    Save the aggregated data to CSV
    
    Parameters:
    -----------
    aggregated_df : pd.DataFrame
        Aggregated data
    output_filepath : str
        Where to save the file
    """
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    aggregated_df.to_csv(output_filepath, index=False)
    
    print(f"✓ Saved to: {output_filepath}")
    print(f"✓ File size: {len(aggregated_df):,} rows × {len(aggregated_df.columns)} columns")


# ==============================================================================
# STEP 5: DISPLAY SAMPLE AND STATISTICS
# ==============================================================================

def display_results(aggregated_df):
    """
    Show sample of aggregated data and key statistics
    
    Parameters:
    -----------
    aggregated_df : pd.DataFrame
        Aggregated data
    """
    
    print("\n" + "="*80)
    print("SAMPLE DATA (First 3 Doctors)")
    print("="*80)
    
    # Show key columns only for readability
    key_cols = ['NUEVO_ID', 'TOTAL_TRX_sum', 'TOTAL_TRX_mean', 
                'ENGAGEMENT_SCORE_sum', 'ATSEG_first']
    
    available_cols = [col for col in key_cols if col in aggregated_df.columns]
    
    if available_cols:
        print(aggregated_df[available_cols].head(3))
    else:
        # If key columns don't exist, show first 5 columns
        print(aggregated_df.iloc[:3, :5])
    
    print("\n" + "="*80)
    print("KEY STATISTICS")
    print("="*80)
    
    # Find columns with prescription totals
    total_cols = [col for col in aggregated_df.columns if 'TOTAL_TRX_sum' in col]
    
    if total_cols:
        print(f"\nTotal Prescriptions Distribution:")
        print(aggregated_df[total_cols[0]].describe())
    
    # Find engagement columns
    engagement_cols = [col for col in aggregated_df.columns 
                      if 'ENGAGEMENT_SCORE_sum' in col]
    
    if engagement_cols:
        print(f"\nTotal Engagement Score Distribution:")
        print(aggregated_df[engagement_cols[0]].describe())


# ==============================================================================
# MAIN FUNCTION - RUN THE ENTIRE PROCESS
# ==============================================================================

def main(input_filepath, output_filepath):
    """
    Main function to run the entire aggregation process
    
    Parameters:
    -----------
    input_filepath : str
        Path to input CSV with weekly data
    output_filepath : str
        Path where to save aggregated data
    """
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "DOCTOR DATA AGGREGATION SCRIPT" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Step 1: Load data
    df = load_data(input_filepath)
    
    # Step 2: Create aggregation rules
    agg_rules = create_aggregation_rules(df)
    
    # Step 3: Perform aggregation
    aggregated_df = aggregate_by_doctor(df, agg_rules)
    
    # Step 4: Save results
    save_aggregated_data(aggregated_df, output_filepath)
    
    # Step 5: Display sample and statistics
    display_results(aggregated_df)
    
    print("\n" + "="*80)
    print("✅ AGGREGATION COMPLETE!")
    print("="*80)
    print("\nYour data is now ready for classification.")
    print("Each doctor appears exactly once in the dataset.")
    print(f"\nOutput saved to: {output_filepath}")
    print("\n")
    
    return aggregated_df


# ==============================================================================
# RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("PROJECT STRUCTURE")
    print("="*80)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Raw data: {RAW_DATA_DIR}")
    print(f"Processed data: {PROCESSED_DATA_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"\nInput file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Run the aggregation
    aggregated_data = main(INPUT_FILE, OUTPUT_FILE)
    
    print(f"\n{'='*80}")
    print(f"Next step: Use 'data/processed/doctors_aggregated.csv' for classification")
    print(f"{'='*80}\n")
