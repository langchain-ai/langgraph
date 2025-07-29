#!/usr/bin/env python3
import json
import os
import glob

def load_replacement_map():
    with open('replacement_map.json', 'r') as f:
        return json.load(f)

def replace_in_file(file_path, replacement_map):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for key, value in replacement_map.items():
        content = content.replace(key, value)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {file_path}")
        return True
    return False

def main():
    replacement_map = load_replacement_map()
    
    md_files = glob.glob('**/*.md', recursive=True)
    updated_count = 0
    
    for file_path in md_files:
        if replace_in_file(file_path, replacement_map):
            updated_count += 1
    
    print(f"Processed {len(md_files)} markdown files")
    print(f"Updated {updated_count} files")

if __name__ == "__main__":
    main()