#Input file: filtered data from filter-data.py
#Divides users into buckets based on the number of records available
#Output: user_bucket_{bucket_num}.csv

import pandas as pd
import os

# Load CSV file
file_path = "data/filtered_data.csv"  # Update with your actual file path
df = pd.read_csv(file_path, delimiter='\t')

# Create directories if they do not exist
os.makedirs("buckets", exist_ok=True)

# Count records per user
user_counts = df['AnonID'].value_counts()

# Filter out users with fewer than 15 records
valid_users = user_counts[user_counts >= 10].index
df_filtered = df[df['AnonID'].isin(valid_users)]
print("Number of Records after Filtering < 10: ", len(df_filtered))

# Define bucket ranges
bucket_ranges = [(10, 15), (16, 25), (26, 35), (36, 45),(46, float('inf'))]  # 56+ includes all users above 55
#bucket_ranges = [(46, float('inf'))]
# Function to assign buckets
def assign_bucket(record_count):
    for start, end in bucket_ranges:
        if start <= record_count <= end:
            return f"{start}-{int(end) if end != float('inf') else 'above'}"
    return None

# Assign buckets based on user record counts
user_counts_filtered = user_counts.loc[valid_users]
user_buckets = user_counts_filtered.apply(assign_bucket)

# Merge bucket info into the main dataframe
df_filtered['Bucket'] = df_filtered['AnonID'].map(user_buckets)

# Save each bucket as a separate CSV and gather stats
bucket_stats = {}

for start, end in bucket_ranges:
    bucket_label = f"{start}-{int(end) if end != float('inf') else 'above'}"
    bucket_df = df_filtered[df_filtered['Bucket'] == bucket_label]
    
    if not bucket_df.empty:
        #bucket_df.to_csv(rf"buckets\user_bucket_{bucket_label}.csv", index=False)
        bucket_df.to_csv(rf"user_bucket_{bucket_label}.csv", index=False)
        num_users = bucket_df['AnonID'].nunique()
        bucket_stats[bucket_label] = num_users

# Print statistics
print("Bucket Statistics (Number of Users per Bucket):")
for bucket, count in bucket_stats.items():
    print(f"{bucket}: {count} users")
