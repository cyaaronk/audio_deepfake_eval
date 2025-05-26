#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import argparse
import os
from typing import Dict, List, Optional
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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

def plot_statistics(id2count: Dict, dataset_name: str, output_dir: str):
    """Plot statistics for codecs and synthesizers."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    def get_font_sizes(num_items: int) -> tuple:
        """Calculate font sizes based on number of items."""
        if num_items <= 5:
            return 8, 7, 6, 6  # title, label, tick, value
        elif num_items <= 10:
            return 7, 6, 5, 5
        elif num_items <= 20:
            return 6, 5, 4, 4
        else:
            return 5, 4, 3, 3
    
    # Plot codec statistics
    if id2count["codec"]:
        codecs = list(id2count["codec"].keys())
        codec_counts = list(id2count["codec"].values())
        num_codecs = len(codecs)
        
        # Get font sizes for codec plot
        title_size, label_size, tick_size, value_size = get_font_sizes(num_codecs)
        
        # Create bar plot for codecs
        sns.barplot(x=codec_counts, y=codecs, ax=ax1, palette='viridis')
        ax1.set_title(f'Codec Distribution - {dataset_name}', fontsize=title_size)
        ax1.set_xlabel('Count', fontsize=label_size)
        ax1.set_ylabel('Codec', fontsize=label_size)
        ax1.tick_params(axis='both', which='major', labelsize=tick_size)
        
        # Add value labels on the bars
        for i, v in enumerate(codec_counts):
            ax1.text(v, i, str(v), va='center', fontsize=value_size)
    
    # Plot synthesizer statistics
    if id2count["synthesizer"]:
        synths = list(id2count["synthesizer"].keys())
        synth_counts = list(id2count["synthesizer"].values())
        num_synths = len(synths)
        
        # Get font sizes for synthesizer plot
        title_size, label_size, tick_size, value_size = get_font_sizes(num_synths)
        
        # Create bar plot for synthesizers
        sns.barplot(x=synth_counts, y=synths, ax=ax2, palette='viridis')
        ax2.set_title(f'Synthesizer Distribution - {dataset_name}', fontsize=title_size)
        ax2.set_xlabel('Count', fontsize=label_size)
        ax2.set_ylabel('Synthesizer', fontsize=label_size)
        ax2.tick_params(axis='both', which='major', labelsize=tick_size)
        
        # Add value labels on the bars
        for i, v in enumerate(synth_counts):
            ax2.text(v, i, str(v), va='center', fontsize=value_size)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize statistics from audio deepfake detection evaluation')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='../visualizations',
                      help='Directory to save visualization plots')
    parser.add_argument('--dataset', type=str,
                      help='Specific dataset to visualize. If not provided, shows available datasets')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # If dataset not provided, show available datasets and exit
    if not args.dataset:
        print("\nERROR: Please specify a dataset to visualize using --dataset")
        print("\nAvailable datasets in config:")
        for dataset in config['datasets']:
            print(f"   {dataset['name']}")
        print("\nExample usage:")
        print(f"   python add_detect_eval/visualize_stats.py --config {args.config} --dataset <dataset_name>")
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
        
        # Plot statistics
        plot_statistics(id2count, dataset_name, args.output_dir)
        print(f"Visualization saved to {args.output_dir}/{dataset_name}_statistics.png")

if __name__ == "__main__":
    main() 