This work is based on the AOL4PS dataset (https://www.scidb.cn/en/detail?dataSetId=5246eba9ec8d4519aa4f0d8f9f092d4b), which should be downloaded first and for our paper, and our paper "ColBERT-Based User Profiles for Personalized Information Retrieval" (https://www.thinkmind.org/library/eKNOW/eKNOW_2025/eknow_2025_1_80_60025.html) The code can be run in two ways: (1 )Treat the entire dataset as a single bucket, (2) Sort the dataset into buckets based on the length of user records, evaluate each bucket separately, and aggregate the results. For option (1), we have included a sample of records for 100 users, unzip sample100.zip and paste its contents in data directory. You can run rerank.py directly, ensuring that the code points to the correct data file paths. To run evaluate the code on the entire dataset and to try out option (2), you will need to filter the original dataset, generate buckets, split each bucket into training and test sets, and update the code to reflect the new data categorization. We have included the code to preprocess the original dataset according to our project needs.

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
If you use any partion of this code please cite our paper.
