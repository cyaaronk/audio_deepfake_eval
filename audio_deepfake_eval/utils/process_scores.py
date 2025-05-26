#!/usr/bin/env python3

import os
import argparse
from pathlib import Path

def process_score_file(file_path: str):
    """Process a score file to modify its format."""
    # Read the original file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    processed_lines = []
    for line in lines:
        if line.strip():  # Skip empty lines
            parts = line.strip().split()
            if len(parts) >= 3:  # Ensure we have at least 3 columns
                # Get the file path and score
                file_path_ = parts[0]
                score = parts[-1]  # Take the last column as score
                
                # Extract base name without extension
                base_name = os.path.splitext(os.path.basename(file_path_))[0]
                
                # Create new line
                processed_lines.append(f"{base_name} {score}\n")
    
    # Write back to the same file
    with open(file_path, 'w') as f:
        f.writelines(processed_lines)

def main():
    parser = argparse.ArgumentParser(description='Process score files to modify their format')
    parser.add_argument('--eer-files-dir', type=str, default='./eer_files',
                      help='Directory containing EER files')
    args = parser.parse_args()
    
    # Find all model directories
    eer_files_dir = Path(args.eer_files_dir)
    model_dirs = ["SCL-Deepfake-audio-detection"]
    
    for model_dir in model_dirs:
        scores_dir = Path(eer_files_dir, model_dir, 'scores')
        if not scores_dir.exists():
            print(f"No scores directory found for model: {model_dir}")
            continue
        
        print(f"\nProcessing scores for model: [{model_dir}]")
        
        # Process each score file
        score_files = [f for f in scores_dir.iterdir() if f.is_file()]
        for score_file in score_files:
            print(f"   Processing: {score_file}")
            try:
                process_score_file(str(score_file))
            except Exception as e:
                print(f"   ERROR processing {score_file}: {str(e)}")
        
        print(f"Completed processing {len(score_files)} files for {model_dir}")

if __name__ == "__main__":
    main() 