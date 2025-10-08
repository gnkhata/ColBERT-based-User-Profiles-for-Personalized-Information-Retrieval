#Input file: data.csv from original dataset, progress.json
#Checks each record, returns 0 if clicked doc not valid or ALL candiList not valid
#if clicked doc valid, returns the number of candilist docs that are valid 
#output file: updated_buckets\filtered_data.csv
import pandas as pd
import json 

doc_df = pd.read_csv(r'data\doc.csv', delimiter='\t')
doc_dict = dict(zip(doc_df['DocIndex'], doc_df['Url']))  # Convert to dictionary

# Optimized get_url function
def get_url(doc_index):
    return doc_dict.get(doc_index, None)

def load_valid_urls(filename=r'outputs\backup.json'):
    try:
        with open(filename, 'r') as file:
            url_data = json.load(file)
        return {url for url, valid in url_data.items() if valid}  # Store only True URLs
    except FileNotFoundError:
        return set()
    
# Ensure CandiList is split properly
def process_candi_list(candi):
    return candi.split('\t') if isinstance(candi, str) else []  # Handle NaN cases

def check_page(url):
    return url in valid_urls  # O(1) lookup

def check_record(row):
    candi_list = row['CandiList']
    click_pos = int(row['ClickPos'])

    clicked_doc_url = get_url(candi_list[click_pos])  # Get clicked URL
    
    if clicked_doc_url in valid_urls:  # If clicked doc is valid
        return sum(1 for doc in candi_list if get_url(doc) in valid_urls)
    return 0

chunk_size = 10000
output_file = r'data\filtered_data.csv'
first_chunk = True
valid_urls = load_valid_urls(r'outputs\progress.json')  # Use a set instead of dict

count = 0
for chunk in pd.read_csv(r'data\data.csv', delimiter='\t', chunksize=chunk_size):
    chunk['CandiList'] = chunk['CandiList'].apply(process_candi_list)
    print(f"\nProcessing chunk number: {count}")
    chunk['DocsCount'] = chunk.apply(check_record, axis=1)
    chunk = chunk[chunk['DocsCount'] > 0]
    chunk.to_csv(output_file, mode='w' if first_chunk else 'a', header=first_chunk, index=False, sep='\t')
    first_chunk = False
    count+=1

print("Num valid urls: ", len(valid_urls))