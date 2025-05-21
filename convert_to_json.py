import pandas as pd
import json

csv_path = '/Users/dantemacstudio/Desktop/langgraph/mayo_database_table.csv'
json_path = '/Users/dantemacstudio/Desktop/langgraph/mayo_database_table.json'

def convert_csv_to_json(csv_path, json_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Convert 'Use_it' column to boolean if it contains boolean-like values
    if 'Use_it' in df.columns:
        df['Use_it'] = df['Use_it'].map({'True': True, 'False': False, 
                                          'true': True, 'false': False,
                                          '1': True, '0': False}, 
                                          na_action='ignore')
    
    # Create a two-level nested structure (database -> tables -> columns)
    result = {}
    
    for _, row in df.iterrows():
        db_name = row['database_name']
        schema = row['table_schema']
        table = row['table_name']
        
        # Create composite key for table (schema.table)
        table_key = f"{schema}.{table}"
        
        if db_name not in result:
            result[db_name] = {"tables": {}}
            
        if table_key not in result[db_name]["tables"]:
            result[db_name]["tables"][table_key] = {"columns": []}
            
        # Add column information
        column_info = {
            "name": row['column_name'],
            "data_type": row['data_type'],
            "ordinal_position": row['ordinal_position'],
            "description": row['Description'],
            "use_it": row['Use_it'],
            "unique_count": row['unique_count']
        }
        
        result[db_name]["tables"][table_key]["columns"].append(column_info)
    
    # Write to JSON file
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

# Add these lines to actually execute the function:
print(f"Starting conversion from {csv_path} to {json_path}...")
result = convert_csv_to_json(csv_path, json_path)
print(f"Conversion complete. JSON file created at: {json_path}")