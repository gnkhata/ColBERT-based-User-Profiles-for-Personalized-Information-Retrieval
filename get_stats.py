#input file: filtered user records from filter_data.py 
#gives plots of number of queries per user and number of unique queries per user 
#this distribution is used to create buckets 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_query_logs(file_path):
    """
    Analyze query logs and return various statistics
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    tuple: Statistics and DataFrame with user metrics
    """
    # Read the CSV file
    df = pd.read_csv(file_path, delimiter='\t')
    
    # Calculate basic statistics
    user_count = df['AnonID'].nunique()
    
    # Calculate queries per user
    queries_per_user = df.groupby('AnonID').agg({
        'QueryIndex': 'nunique',  # Unique queries
        'AnonID': 'count'        # Total records
    }).rename(columns={
        'QueryIndex': 'unique_queries',
        'AnonID': 'total_records'
    })
    
    # Calculate summary statistics
    stats = {
        'total_users': user_count,
        'avg_queries_per_user': queries_per_user['unique_queries'].mean(),
        'median_queries_per_user': queries_per_user['unique_queries'].median(),
        'avg_records_per_user': queries_per_user['total_records'].mean(),
        'median_records_per_user': queries_per_user['total_records'].median()
    }
    
    return stats, queries_per_user

def plot_query_distribution(queries_per_user):
    """
    Create a distribution plot of queries per user
    
    Parameters:
    queries_per_user (pd.DataFrame): DataFrame containing user metrics
    """
    plt.figure(figsize=(12, 6))
    
    # Create distribution plot
    sns.histplot(data=queries_per_user, x='unique_queries', bins=50)
    plt.title('Distribution of Unique Queries per User')
    plt.xlabel('Number of Unique Queries')
    plt.ylabel('Number of Users')
    
    # Add a box plot to show quartiles
    plt.figure(figsize=(12, 4))
    sns.boxplot(data=queries_per_user, x='unique_queries')
    plt.title('Box Plot of Unique Queries per User')
    plt.xlabel('Number of Unique Queries')

# Example usage
if __name__ == "__main__":
    file_path = r"data\data.csv"  # Replace with your file path
    stats, user_metrics = analyze_query_logs(file_path)
    
    # Print statistics
    print("\nQuery Log Statistics:")
    print(f"Total number of users: {stats['total_users']:,}")
    print(f"\nQueries per user:")
    print(f"- Average: {stats['avg_queries_per_user']:.2f}")
    print(f"- Median: {stats['median_queries_per_user']:.2f}")
    print(f"\nRecords per user:")
    print(f"- Average: {stats['avg_records_per_user']:.2f}")
    print(f"- Median: {stats['median_records_per_user']:.2f}")
    
    # Create visualization
    plot_query_distribution(user_metrics)
    plt.show()