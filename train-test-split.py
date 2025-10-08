#Input: csv file of buckets (each bucket has all records of users in that category)
#Splits into train-test based on defined thresholds 
#output: train and test files of buckets

import pandas as pd
import json
import ast
from collections import defaultdict

def load_availability_data(progress_file, doc_mapping_file):
    """Load and prepare document availability mapping"""
    # Load progress data
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
    
    # Load document mapping
    doc_df = pd.read_csv(doc_mapping_file, delimiter='\t')
    
    # Create DocIndex to URL mapping
    docid_to_url = dict(zip(doc_df['DocIndex'], doc_df['Url']))
    
    # Create DocIndex to availability mapping
    docid_availability = {}
    for docid, url in docid_to_url.items():
        docid_availability[docid] = progress_data.get(url, False)
    
    return docid_availability

def count_available_docs(candi_list, docid_availability):
    """Count how many documents in the candidate list are available"""
    try:
        docs = ast.literal_eval(candi_list)
        available_count = sum(1 for doc in docs if docid_availability.get(doc, False))
        return available_count
    except:
        return 0

def get_repeat_threshold(threshold):
    """Calculate maximum allowed repeated queries (1/3 of threshold)"""
    return threshold // 3

def process_user_records(user_df, train_threshold, docid_availability):
    """Process records for a single user"""
    # Sort by QueryTime
    user_df = user_df.sort_values('QueryTime')
    
    # Process train records
    train_records = []
    seen_queries = set()
    repeat_count = 0
    max_train_repeats = get_repeat_threshold(train_threshold)
    
    for _, row in user_df.iterrows():
        if len(train_records) < train_threshold:
            if row['QueryIndex'] in seen_queries:
                if repeat_count < max_train_repeats:
                    train_records.append(row)
                    repeat_count += 1
            else:
                train_records.append(row)
                seen_queries.add(row['QueryIndex'])
    
    # If we don't have enough train records, return empty frames
    if len(train_records) < train_threshold:
        return pd.DataFrame(), pd.DataFrame()
    
    # Process test records
    train_df = pd.DataFrame(train_records)
    last_train_time = train_df['QueryTime'].max()
    
    test_records = []
    seen_queries = set()
    repeat_count = 0
    max_test_repeats = get_repeat_threshold(5)  # Using 5 as minimum test threshold
    
    remaining_df = user_df[user_df['QueryTime'] > last_train_time]
    
    for _, row in remaining_df.iterrows():
        available_docs = count_available_docs(row['CandiList'], docid_availability)
        
        if available_docs >= 5:
            if row['QueryIndex'] in seen_queries:
                if repeat_count < max_test_repeats:
                    test_records.append(row)
                    repeat_count += 1
            else:
                test_records.append(row)
                seen_queries.add(row['QueryIndex'])
    
    # If we don't have minimum 5 test records, return empty frames
    if len(test_records) < 5:
        return pd.DataFrame(), pd.DataFrame()
    
    test_df = pd.DataFrame(test_records)
    return train_df, test_df

def process_bucket(file_path, train_threshold, docid_availability):
    """Process a single bucket file"""
    # Read bucket file
    df = pd.read_csv(file_path)
    
    valid_user_data = {'train': [], 'test': []}
    
    # Process each user
    for user_id, user_df in df.groupby('AnonID'):
        train_df, test_df = process_user_records(user_df, train_threshold, docid_availability)
        
        # Only keep users who have both valid train and test records
        if not train_df.empty and not test_df.empty:
            valid_user_data['train'].append(train_df)
            valid_user_data['test'].append(test_df)
    
    # Combine results
    final_train = pd.concat(valid_user_data['train']) if valid_user_data['train'] else pd.DataFrame()
    final_test = pd.concat(valid_user_data['test']) if valid_user_data['test'] else pd.DataFrame()
    
    return final_train, final_test

def main():
    # Define bucket-specific train thresholds
    bucket_train_thresholds = {
        "10-15": 5,
        "16-25": 10,
        "26-35": 20,
        "36-45": 30,
        "46-above": 40,
    }
    
    # Load document availability data
    docid_availability = load_availability_data(r'outputs\progress.json', r'data\doc.csv')
    
    #print(f"docid_availability: {docid_availability}")
    
    # Process each bucket
    import glob
    bucket_files = glob.glob(r"buckets\user_bucket_*.csv")
    
    # Store statistics
    stats = defaultdict(dict)
    print(f"bucket_files: {bucket_files}")
    for file in bucket_files:
        print(f"file: {file}")
        bucket_label = file.split("user_bucket_")[1].replace(".csv", "")
        print(f"bucket_label: {bucket_label}")
        if bucket_label not in bucket_train_thresholds:
            continue
            
        train_threshold = bucket_train_thresholds[bucket_label]
        
        # Process bucket
        train_df, test_df = process_bucket(file, train_threshold, docid_availability)
        print(f"bucket: {bucket_label}, train_df: {train_df} \n")
        print(f"bucket: {bucket_label}, train_df: {test_df} \n")
        if not train_df.empty and not test_df.empty:
            # Verify same users in both sets
            train_users = set(train_df['AnonID'].unique())
            test_users = set(test_df['AnonID'].unique())
            assert train_users == test_users, f"Mismatch in users for bucket {bucket_label}"
            
            # Save results
            train_df.to_csv(f"merged_data\train_{bucket_label}.csv", index=False)
            test_df.to_csv(f"merged_data\test_{bucket_label}.csv", index=False)
            
            # Collect statistics
            stats[bucket_label].update({
                'users': len(train_users),
                'train_records': len(train_df),
                'test_records': len(test_df),
                'avg_train_per_user': len(train_df) / len(train_users),
                'avg_test_per_user': len(test_df) / len(train_users)
            })
    
    # Print statistics
    print("\nProcessing Statistics:")
    for bucket, bucket_stats in stats.items():
        print(f"\nBucket {bucket}:")
        print(f"Valid Users: {bucket_stats['users']}")
        print(f"Train Records: {bucket_stats['train_records']} (avg {bucket_stats['avg_train_per_user']:.2f} per user)")
        print(f"Test Records: {bucket_stats['test_records']} (avg {bucket_stats['avg_test_per_user']:.2f} per user)")

if __name__ == "__main__":
    main()
    
    #df = pd.read_csv("user_bucket_16-25.csv")
    #print(df)
