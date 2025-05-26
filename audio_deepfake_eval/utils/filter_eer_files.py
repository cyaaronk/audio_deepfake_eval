#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def read_trial_file(trial_file_path):
    """Read audio IDs from a trial file."""
    audio_ids = set()
    try:
        with open(trial_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Extract the audio ID (first column)
                    audio_id = line.split()[0]
                    # Remove file extension if present
                    audio_id = os.path.splitext(audio_id)[0]
                    audio_ids.add(audio_id)
        return audio_ids
    except Exception as e:
        logger.error(f"Error reading trial file {trial_file_path}: {e}")
        return set()

def filter_score_file(score_file_path, audio_ids, output_dir):
    """Filter a score file to only include lines with audio IDs in the dataset."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the score file and filter lines
        filtered_lines = []
        with open(score_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    filtered_lines.append(line)
                    continue
                
                # Extract the audio ID (first column)
                parts = line.split()
                if len(parts) < 1:
                    continue
                
                audio_id = parts[0]
                # Remove file extension if present
                audio_id = os.path.splitext(audio_id)[0]
                
                # Check if this audio ID is in our dataset
                if audio_id in audio_ids:
                    filtered_lines.append(line)
        
        # Write the filtered lines to the output file
        output_file = os.path.join(output_dir, os.path.basename(score_file_path))
        with open(output_file, 'w') as f:
            f.write('\n'.join(filtered_lines))
        
        return len(filtered_lines)
    except Exception as e:
        logger.error(f"Error filtering score file {score_file_path}: {e}")
        return 0

def filter_key_file(key_file_path, audio_ids, output_dir):
    """Filter a key file to only include lines with audio IDs in the dataset."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the key file and filter lines
        filtered_lines = []
        with open(key_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    filtered_lines.append(line)
                    continue
                
                # Extract the audio ID (first column)
                parts = line.split()
                if len(parts) < 1:
                    continue
                
                audio_id = parts[0]
                # Remove file extension if present
                audio_id = os.path.splitext(audio_id)[0]
                
                # Check if this audio ID is in our dataset
                if audio_id in audio_ids:
                    filtered_lines.append(line)
        
        # Write the filtered lines to the output file
        output_file = os.path.join(output_dir, os.path.basename(key_file_path))
        with open(output_file, 'w') as f:
            f.write('\n'.join(filtered_lines))
        
        return len(filtered_lines)
    except Exception as e:
        logger.error(f"Error filtering key file {key_file_path}: {e}")
        return 0

def process_dataset(dataset_dir, eer_files_dir, output_dir):
    """Process a single dataset."""
    dataset_name = os.path.basename(dataset_dir)
    trial_file = os.path.join(dataset_dir, f"{dataset_name}.cm.eval.trl.txt")
    
    if not os.path.exists(trial_file):
        logger.warning(f"Trial file not found: {trial_file}")
        return
    
    logger.info(f"Processing dataset: {dataset_name}")
    audio_ids = read_trial_file(trial_file)
    logger.info(f"Found {len(audio_ids)} audio IDs in {dataset_name}")
    
    # Process each model in eer_files
    for model_dir in os.listdir(eer_files_dir):
        model_path = os.path.join(eer_files_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        logger.info(f"Processing model: {model_dir}")
        
        # Process scores
        scores_dir = os.path.join(model_path, "scores")
        if os.path.exists(scores_dir):
            output_scores_dir = os.path.join(output_dir, model_dir, "scores")
            # Create the output directory for scores
            os.makedirs(output_scores_dir, exist_ok=True)
            
            for score_file in os.listdir(scores_dir):
                if score_file.endswith(".txt"):
                    score_path = os.path.join(scores_dir, score_file)
                    filtered_count = filter_score_file(score_path, audio_ids, output_scores_dir)
                    logger.info(f"Filtered {filtered_count} lines in {score_file}")
        
        # Process keys
        keys_dir = os.path.join(model_path, "keys")
        if os.path.exists(keys_dir):
            output_keys_dir = os.path.join(output_dir, model_dir, "keys")
            # Create the output directory for keys
            os.makedirs(output_keys_dir, exist_ok=True)
            
            for key_file in os.listdir(keys_dir):
                if key_file.endswith(".txt"):
                    key_path = os.path.join(keys_dir, key_file)
                    filtered_count = filter_key_file(key_path, audio_ids, output_keys_dir)
                    logger.info(f"Filtered {filtered_count} lines in {key_file}")

def main():
    parser = argparse.ArgumentParser(description="Filter eer_files based on dataset trial files")
    parser.add_argument("--datasets-dir", type=str, default="./datasets", help="Directory containing datasets")
    parser.add_argument("--eer-files-dir", type=str, default="./eer_files", help="Directory containing eer_files")
    parser.add_argument("--output-dir", type=str, default="./filtered_eer_files", help="Output directory for filtered eer_files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    for dataset_dir in os.listdir(args.datasets_dir):
        dataset_path = os.path.join(args.datasets_dir, dataset_dir)
        if os.path.isdir(dataset_path):
            process_dataset(dataset_path, args.eer_files_dir, args.output_dir)
    
    logger.info(f"Filtering complete. Filtered eer_files saved to {args.output_dir}")

if __name__ == "__main__":
    main() 