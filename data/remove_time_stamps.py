import gzip
import json

lang = 'zho'
input_file_path = f'/home/dju/datasets/neuclir1/{lang}.mt.eng-00000-of-00001.jsonl.gz'
output_file_path = f'{lang}.processed_output.jsonl.gz'
processed_count = 0

# Open the input file for reading (gzipped) and the output file for writing (gzipped)
with gzip.open(input_file_path, 'rt', encoding='utf-8') as infile, \
     gzip.open(output_file_path, 'wt', encoding='utf-8') as outfile:
    
    # Iterate through each line (JSON record) in the input file
    for line in infile:
        try:
            # 1. Load the line as a Python dictionary
            record = json.loads(line)
            
            # 2. Process the record (Your logic here)
            # Example: Remove the problematic timestamp key
            if 'timestamp' in record:
                del record['timestamp']
            
            # Example: Add a new key
            record['processed'] = True
            
            # 3. Write the modified record to the output file
            # json.dumps() converts the dictionary back to a string
            # The newline character ('\n') is CRUCIAL for the JSON Lines format
            outfile.write(json.dumps(record) + '\n')
            
            processed_count += 1
            if processed_count % 10000 == 0:
                print(f"Processed {processed_count} records...")
                
        except json.JSONDecodeError as e:
            # Handle cases where a line might not be valid JSON
            print(f"Skipping bad line: {line.strip()}. Error: {e}")

print(f"Done! Total records processed and written: {processed_count}")
