import os
import json
import re

def process_notebooks():
    # Get current directory
    current_dir = os.getcwd()
    
    # Counter for modified files
    modified_count = 0
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(current_dir):
        # Filter for .ipynb files
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                
                try:
                    # Load the notebook
                    with open(notebook_path, 'r', encoding='utf-8') as f:
                        notebook_content = json.load(f)
                    
                    # Flag to track if we need to save changes
                    needs_update = False
                    
                    # Check if notebook contains 'create_react_agent'
                    contains_react_agent = False
                    
                    # Iterate through cells to check for 'create_react_agent'
                    for cell in notebook_content.get('cells', []):
                        if cell.get('cell_type') == 'code':
                            source = ''.join(cell.get('source', []))
                            if 'create_react_agent' in source:
                                contains_react_agent = True
                                break
                    
                    # If it contains 'create_react_agent', then replace the pip install command
                    if contains_react_agent:
                        # Now iterate through cells again to do the replacement
                        for cell in notebook_content.get('cells', []):
                            if cell.get('cell_type') == 'code':
                                new_source = []
                                for line in cell.get('source', []):
                                    if re.search(r'pip\s+install\s+langgraph(?!\s+langgraph-prebuilt)', line):
                                        new_line = line.replace('pip install langgraph', 'pip install langgraph langgraph-prebuilt')
                                        new_source.append(new_line)
                                        needs_update = True
                                    else:
                                        new_source.append(line)
                                cell['source'] = new_source
                        
                        # Save the notebook if changes were made
                        if needs_update:
                            with open(notebook_path, 'w', encoding='utf-8') as f:
                                json.dump(notebook_content, f, indent=1)
                            modified_count += 1
                            print(f"Updated: {notebook_path}")
                
                except Exception as e:
                    print(f"Error processing {notebook_path}: {e}")
    
    print(f"Total notebooks modified: {modified_count}")

if __name__ == "__main__":
    process_notebooks()
