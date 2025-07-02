#!/usr/bin/env python3
"""
Script to extract images from the graph-api.ipynb notebook and save them to assets folder.
"""

import json
import base64
import os
from pathlib import Path

def extract_images_from_notebook(notebook_path, assets_dir):
    """Extract images from notebook and save them to assets directory."""
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Create assets directory if it doesn't exist
    os.makedirs(assets_dir, exist_ok=True)
    
    image_count = 0
    
    # Process each cell
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            # Check if this cell contains draw_mermaid_png
            source = ''.join(cell.get('source', []))
            if 'draw_mermaid_png' in source:
                print(f"Found draw_mermaid_png in cell {cell_idx}")
                
                # Check for outputs with images
                if 'outputs' in cell:
                    for output_idx, output in enumerate(cell['outputs']):
                        if output.get('output_type') == 'display_data':
                            data = output.get('data', {})
                            
                            # Check for PNG data
                            if 'image/png' in data:
                                png_data = data['image/png']
                                
                                # Decode base64 data
                                try:
                                    image_bytes = base64.b64decode(png_data)
                                    
                                    # Generate filename
                                    image_count += 1
                                    filename = f"graph_api_image_{image_count}.png"
                                    filepath = os.path.join(assets_dir, filename)
                                    
                                    # Save the image
                                    with open(filepath, 'wb') as img_file:
                                        img_file.write(image_bytes)
                                    
                                    print(f"Saved image: {filepath}")
                                    
                                except Exception as e:
                                    print(f"Error decoding image {image_count}: {e}")
    
    print(f"Extracted {image_count} images to {assets_dir}")
    return image_count

if __name__ == "__main__":
    notebook_path = "docs/docs/how-tos/graph-api.ipynb"
    assets_dir = "docs/docs/how-tos/assets"
    
    if os.path.exists(notebook_path):
        count = extract_images_from_notebook(notebook_path, assets_dir)
        print(f"Successfully extracted {count} images")
    else:
        print(f"Notebook not found: {notebook_path}") 