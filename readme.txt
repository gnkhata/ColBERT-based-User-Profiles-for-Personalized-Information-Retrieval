Creating Buckets:
- outputs/progress.json: maps each URL extracted from users to T/F based on availability

- filter-data.py:
  - Gets data from data.csv (the ORIGINAL data file in AOL4PS)
  - Filters records based on validity of clicked doc
  - Returns count of valid docs from CandiList
  - Records with clicked doc or 0 CandiList valid are removed

- get-stats.py:
  - From filtered_data.csv
  - Visualizes the distribution of records and unique records across users
  - Used to determine the bucket sizes


- generate_buckets:
  - Takes filtered_data
  - Divides users and their records into bucket files based on the number of records

- train-test-split:
  - From each bucket file
  - Divides records for each user into train and test split based on defined threshold
-rerank.py
  -the main retrieval code for generating user profiles, indexing with bert, and reraranking the retrieved documents 