import json
import sys

def clean_notebook(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Clean metadata
    if 'metadata' in notebook:
        if 'widgets' in notebook['metadata']:
            del notebook['metadata']['widgets']
    
    # Clean cell outputs that might contain widget states
    if 'cells' in notebook:
        for cell in notebook['cells']:
            if 'outputs' in cell:
                cell['outputs'] = [
                    output for output in cell['outputs']
                    if output.get('output_type') != 'display_data' or
                    not any(mime_type.startswith('@jupyter-widgets')
                           for mime_type in output.get('data', {}).keys())
                ]
    
    # Write back the cleaned notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

if __name__ == '__main__':
    notebook_path = 'NER (NLP).ipynb'
    clean_notebook(notebook_path) 