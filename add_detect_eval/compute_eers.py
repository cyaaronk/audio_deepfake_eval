#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import argparse
from collections import defaultdict
import os
from typing import Dict, List, Optional
import yaml
import numpy as np
from sklearn.metrics import roc_curve

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['model_name', 'datasets', 'eer_files_dir']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file")
    
    # Validate dataset configurations
    for dataset in config['datasets']:
        if 'name' not in dataset:
            raise ValueError("Each dataset must have a 'name' field")
        if 'include_patterns' not in dataset:
            dataset['include_patterns'] = None
    
    return config

def read_scores_and_keys(model_name: str, dataset_name: str, eer_files_dir: str,
                        include_patterns: Optional[List[str]] = None) -> tuple:
    """Read scores and keys files for a given model and dataset."""
    id2meta = defaultdict(dict)
    id2count = defaultdict(lambda: defaultdict(int))
    
    # Read scores file
    scores_path = os.path.join(eer_files_dir, model_name, 'scores', f'{dataset_name}.txt')
    with open(scores_path, 'r') as rf:
        for line in rf:
            if line == "\n":
                continue
            split = line.strip().split(' ')
            id_ = split[0]
            score = float(split[1])
            
            if include_patterns and all(pattern not in id_ for pattern in include_patterns):
                continue
                
            id2meta[id_]["score"] = score

    # Read keys file
    keys_path = os.path.join(eer_files_dir, model_name, 'keys', f'{dataset_name}.txt')
    with open(keys_path, 'r') as rf:
        for line in rf:
            if line == "\n":
                continue
            split = line.strip().split(' ')
            id_ = split[1]
            data_type = split[7]
            label = split[5]
            codec = split[2]
            synthesizer = split[4]
            
            if include_patterns and all(pattern not in id_ for pattern in include_patterns):
                continue

            if data_type == "eval":
                id2meta[id_]["label"] = label
                id2meta[id_]["codec"] = codec
                if synthesizer != "-":
                    id2meta[id_]["synthesizer"] = synthesizer
                    id2count["synthesizer"][synthesizer] += 1
                if codec != "-" and codec != "nocodec":
                    id2count["codec"][codec] += 1

    return id2meta, id2count

def compute_eer(scores: List[float], labels: List[str]) -> float:
    """Compute Equal Error Rate (EER) using scikit-learn's roc_curve."""
    # Convert labels to binary (0 for spoof, 1 for bonafide)
    y = [1 if label == "bonafide" else 0 for label in labels]
    if sum(y) == 0 or sum(y) == len(y):
        raise ValueError("Cannot comput EER because all samples are either spoof or bonafide")
    
    # Compute ROC curve
    fpr, tpr, threshold = roc_curve(y, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find the threshold where FNR and FPR are closest
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_precision = fnr[np.nanargmin(np.absolute((fnr - fpr)))] - fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    # Verify precision as in the notebook
    assert eer_precision < 0.02, eer_precision
    
    print(f"EER threshold = {eer_threshold}")
    return EER

def main():
    parser = argparse.ArgumentParser(description='Compute EERs for audio deepfake detection evaluation')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration YAML file')
    parser.add_argument('--dataset', type=str,
                      help='Specific dataset to evaluate. If not provided, evaluates all datasets in config')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # If dataset not provided, show available datasets and exit
    if not args.dataset:
        print("\nERROR: Please specify a dataset to evaluate using --dataset")
        print("\nAvailable datasets in config:")
        for dataset in config['datasets']:
            print(f"   {dataset['name']}")
        print("\nExample usage:")
        print(f"   python add_detect_eval/compute_eers.py --config {args.config} --dataset <dataset_name>")
        return
    
    # Filter datasets based on --dataset argument
    datasets = [d for d in config['datasets'] if d['name'] == args.dataset]
    if not datasets:
        print(f"\nERROR: Dataset '{args.dataset}' not found in config file")
        print("\nAvailable datasets in config:")
        for dataset in config['datasets']:
            print(f"   {dataset['name']}")
        return
    config['datasets'] = datasets
    
    # Process each dataset
    for dataset_config in config['datasets']:
        dataset_name = dataset_config['name']
        include_patterns = dataset_config.get('include_patterns')
        
        print(f"\nProcessing dataset: [{dataset_name}]")
        if include_patterns:
            print(f"   Including patterns: {', '.join(include_patterns)}")
        
        # Read scores and keys
        id2meta, id2count = read_scores_and_keys(
            config['model_name'],
            dataset_name,
            config['eer_files_dir'],
            include_patterns
        )
        
        # Prepare data for EER computation
        scores = []
        labels = []
        for id_, meta in id2meta.items():
            if "score" in meta and "label" in meta:
                scores.append(meta["score"])
                labels.append(meta["label"])
        
        # Compute EER
        eer = compute_eer(scores, labels)
        print(f"EER: {eer:.2%}")
        
        # Print statistics
        print("\nCodec statistics:")
        for codec, count in id2count["codec"].items():
            print(f"   {codec}: {count} files")
            
        print("\nSynthesizer statistics:")
        for synth, count in id2count["synthesizer"].items():
            print(f"   {synth}: {count} files")

if __name__ == "__main__":
    main() 