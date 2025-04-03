#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import argparse
import numpy as np
from sklearn.metrics import roc_curve
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Dict, List, Tuple, Optional, Any
import yaml
import os
import pandas as pd
import json

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def read_scores_and_keys(model_name: str, dataset_name: str, eer_files_dir: str, 
                        include_patterns: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
    """Read scores and keys files for a model and dataset."""
    id2meta = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    id2count = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Read scores
    scores_path = os.path.join(eer_files_dir, model_name, "scores", f"{dataset_name}.txt")
    with open(scores_path, 'r') as rf:
        lines = rf.readlines()
        for line in lines:
            if line == "\n":
                continue
            split = line.strip().split(' ')
            id_ = split[0]
            score = float(split[1])
            
            if include_patterns and all(pattern not in id_ for pattern in include_patterns):
                continue
                
            id2meta[model_name][dataset_name][id_]["score"] = score
    
    # Read keys
    keys_path = os.path.join(eer_files_dir, model_name, "keys", f"{dataset_name}.txt")
    with open(keys_path, 'r') as rf:
        lines = rf.readlines()
        for line in lines:
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
                id2meta[model_name][dataset_name][id_]["label"] = label
                id2meta[model_name][dataset_name][id_]["codec"] = codec
                
                if synthesizer != "-":
                    id2meta[model_name][dataset_name][id_]["synthesizer"] = synthesizer
                    id2count[model_name][dataset_name]["synthesizer"][synthesizer] = \
                        id2count[model_name][dataset_name]["synthesizer"].get(synthesizer, 0) + 1
                        
                if codec != "-" and codec != "nocodec":
                    id2count[model_name][dataset_name]["codec"][codec] = \
                        id2count[model_name][dataset_name]["codec"].get(codec, 0) + 1
                        
                if label == "bonafide" and "score" in id2meta[model_name][dataset_name][id_]:
                    id2count[model_name][dataset_name]["bonafide"] = id2count[model_name][dataset_name].get("bonafide", 0) + 1
                elif label == "spoof" and "score" in id2meta[model_name][dataset_name][id_]:
                    id2count[model_name][dataset_name]["spoof"] = id2count[model_name][dataset_name].get("spoof", 0) + 1
    return id2meta, id2count

def get_subset(id2meta: Dict, bonafide_patterns: List[Optional[str]] = [None, None, None],
               spoof_patterns: List[Optional[str]] = [None, None, None],
               synthesizer_patterns: List[Optional[str]] = [None]) -> Tuple[List[int], List[float]]:
    """Get subset of data based on patterns."""
    y, score = [], []
    counts = [0, 0]  # spoof count, bona fide count
    
    for model in id2meta.keys():
        if spoof_patterns[0] is None or spoof_patterns[0] in model:
            for dataset in id2meta[model].keys():
                if spoof_patterns[1] is None or spoof_patterns[1] == dataset:
                    for id_ in id2meta[model][dataset].keys():
                        if spoof_patterns[2] is None or spoof_patterns[2] in id_:
                            if id2meta[model][dataset][id_].get("label") == "spoof":
                                if synthesizer_patterns == [None] or any(
                                    pattern in id2meta[model][dataset][id_].get("synthesizer", "")
                                    for pattern in synthesizer_patterns
                                ):
                                    try:
                                        score.append(id2meta[model][dataset][id_]["score"])
                                        y.append(0)
                                        counts[0] += 1
                                    except Exception as e:
                                        print(f"ERROR: Failed to process spoof sample: {id_}")
                                        print(f"Error details: {e}")
        
        if bonafide_patterns[0] is None or bonafide_patterns[0] in model:
            for dataset in id2meta[model].keys():
                if bonafide_patterns[1] is None or bonafide_patterns[1] == dataset:
                    for id_ in id2meta[model][dataset].keys():
                        if bonafide_patterns[2] is None or bonafide_patterns[2] in dataset:
                            if id2meta[model][dataset][id_].get("label") == "bonafide":
                                try:
                                    score.append(id2meta[model][dataset][id_]["score"])
                                    y.append(1)
                                    counts[1] += 1
                                except Exception as e:
                                    print(f"ERROR: Failed to process bonafide sample: {id_}")
                                    print(f"Error details: {e}")
    
    print("Sample counts:")
    print(f"   Spoof samples:    {counts[0]}")
    print(f"   Bonafide samples: {counts[1]}")
    return y, score

def compute_eer(y: List[int], score: List[float]) -> Tuple[float, float]:
    """Compute Equal Error Rate."""
    fpr, tpr, threshold = roc_curve(y, score, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_precision = fnr[np.nanargmin(np.absolute((fnr - fpr)))] - fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    assert eer_precision < 0.02, eer_precision
    print(f"EER: {EER:.2%}")
    return EER, eer_threshold

def save_results(eers: List[List[float]], bonafide_subsets: List[str],
                spoof_subsets: List[str], output_dir: str, model_name: str,
                id2count: Dict = None):
    """Save EER results and statistics."""
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Convert EER matrix to numpy array and ensure correct shape
    eer_matrix = np.array(eers)
    
    # Create DataFrame with proper dimensions
    df = pd.DataFrame(eer_matrix, index=bonafide_subsets)
    
    # Create column names based on spoof subsets and their synthesizers
    columns = []
    for spoof_subset in spoof_subsets:
        if id2count and spoof_subset in id2count[model_name]:
            # Get synthesizers for this spoof subset
            synthesizers = sorted(id2count[model_name][spoof_subset]["synthesizer"].keys())
            if synthesizers:
                # Add columns for each synthesizer
                for synthesizer in synthesizers:
                    columns.append(f"{spoof_subset}_{synthesizer}")
            else:
                # If no synthesizers found, use the spoof subset name as is
                columns.append(spoof_subset)
        else:
            # If no synthesizer info available, use the spoof subset name as is
            columns.append(spoof_subset)
    
    # Verify that number of columns matches the EER matrix width
    if len(columns) != eer_matrix.shape[1]:
        print(f"\nWARNING: Column count mismatch in EER matrix")
        print(f"   Expected columns: {len(columns)}")
        print(f"   Actual columns: {eer_matrix.shape[1]}")
        print(f"   Using generic column names instead")
        # Fall back to generic column names
        columns = [f"spoof_{i+1}" for i in range(eer_matrix.shape[1])]
    
    df.columns = columns
    
    # Save EER matrix as CSV with error handling
    csv_path = os.path.join(model_output_dir, 'eer_matrix.csv')
    try:
        df.to_csv(csv_path)
    except PermissionError:
        print("\nError: Could not save eer_matrix.csv")
        print("Please make sure the file is not currently open in another program (e.g., Excel)")
        print(f"Path: {csv_path}")
        raise
    
    # Save detailed results as JSON with error handling
    json_path = os.path.join(model_output_dir, 'results.json')
    try:
        results = {
            'model_name': model_name,
            'bonafide_subsets': bonafide_subsets,
            'spoof_subsets': spoof_subsets,
            'eer_matrix': eer_matrix.tolist(),
            'mean_eer': float(np.mean(eer_matrix)),
            'std_eer': float(np.std(eer_matrix))
        }
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
    except PermissionError:
        print("\nError: Could not save results.json")
        print("Please make sure the file is not currently open in another program")
        print(f"Path: {json_path}")
        raise

def plot_cross_testing_matrix(eers: List[List[float]], bonafide_subsets: List[str],
                            spoof_subsets: List[str], plot_dir: str, model_name: str, id2count: Dict = None):
    """Plot cross-testing matrix with synthesizers on y-axis and bonafide on x-axis."""
    # Create model-specific plot directory
    model_plot_dir = os.path.join(plot_dir, model_name)
    os.makedirs(model_plot_dir, exist_ok=True)
    
    # Set seaborn font scale based on dataset counts
    font_scale = min(0.7, 1.0 - 0.05 * max(len(bonafide_subsets) // 10, len(spoof_subsets) // 10))
    sns.set(font_scale=font_scale)
    
    # Convert EER matrix to numpy array and transpose
    data = np.array(eers, dtype='float').T
    
    # Calculate figure size based on dataset counts
    # Make width significantly larger than height
    # Base width is 20, add 4 units for every 3 bonafide datasets
    # Base height is 10, add 2 units for every 3 spoof datasets
    width = 20 + 40 * (len(bonafide_subsets) // 3)
    height = 10 + 2 * (len(spoof_subsets) // 3)
    
    # Create figure with dynamic size and high DPI
    plt.figure(dpi=1200)
    
    # Plot heatmap with new orientation
    ax = sns.heatmap(
        data,
        robust=True,
        cmap='Blues',
        xticklabels=bonafide_subsets,
        square=False,  # Allow rectangular shape
        cbar_kws={'format': FuncFormatter(lambda x, _: "%.0f%%" % (x * 100))}
    )
    
    # Calculate aspect ratio based on matrix dimensions
    # Make the plot wider by adjusting the aspect ratio
    matrix_aspect = data.shape[1] / data.shape[0]  # width/height of the data
    desired_aspect = (width / height) * 0.0005  # Increase width relative to height
    ax.set_aspect(desired_aspect / matrix_aspect)
    
    # Customize plot appearance
    ax.set_xlabel('Bona fide Subset')
    ax.set_ylabel('Synthesizer ID')
    
    # Move x-axis labels to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # Remove tick marks
    plt.tick_params(top=False, bottom=False, left=False, right=False)
    
    # Set y-axis rotation
    plt.yticks(rotation=0)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(model_plot_dir, 'cross_testing_matrix.png'), 
                bbox_inches='tight', pad_inches=0.5)
    plt.close()

def load_existing_results(model_output_dir: str) -> Tuple[List[List[float]], List[str], List[str], Dict]:
    """Load existing results from output directory."""
    # Load results.json
    results_path = os.path.join(model_output_dir, 'results.json')
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract data
    eers = results['eer_matrix']
    bonafide_subsets = results['bonafide_subsets']
    spoof_subsets = results['spoof_subsets']
    
    # Load EER details
    eer_details_path = os.path.join(model_output_dir, 'eer_details.json')
    with open(eer_details_path, 'r') as f:
        eer_details = json.load(f)
    
    return eers, bonafide_subsets, spoof_subsets, eer_details

def main():
    parser = argparse.ArgumentParser(description='Compute cross-testing EERs for audio deepfake detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='./output',
                      help='Directory to save EER results and statistics')
    parser.add_argument('--plot-dir', type=str, default='./visualizations',
                      help='Directory to save visualization plots')
    parser.add_argument('--force-recompute', action='store_true',
                      help='Force recomputation of EERs even if results exist')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if results already exist
    model_output_dir = os.path.join(args.output_dir, config['model_name'])
    results_exist = (
        os.path.exists(os.path.join(model_output_dir, 'results.json')) and
        os.path.exists(os.path.join(model_output_dir, 'eer_matrix.csv')) and
        os.path.exists(os.path.join(model_output_dir, 'eer_details.json'))
    )
    
    if results_exist and not args.force_recompute:
        print(f"\nFound existing results for model [{config['model_name']}]")
        print("Loading results and generating visualization...")
        
        # Load existing results
        eers, bonafide_subsets, spoof_subsets, eer_details = load_existing_results(model_output_dir)
        
        # Plot using existing results
        plot_cross_testing_matrix(eers, bonafide_subsets, spoof_subsets, args.plot_dir, 
                                config['model_name'], None)
        
        print(f"\nVisualization updated successfully:")
        print(f"   Using existing results from: {model_output_dir}")
        print(f"   Updated visualization saved to: {args.plot_dir}")
        return
    
    # Initialize data structures
    id2meta = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    id2count = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Read data for all datasets
    for dataset_config in config['datasets']:
        dataset_name = dataset_config['name']
        include_patterns = dataset_config.get('include_patterns')
        
        print(f"\nProcessing dataset: [{dataset_name}]")
        if include_patterns:
            print(f"   Including patterns: {', '.join(include_patterns)}")
        
        meta, count = read_scores_and_keys(
            config['model_name'],
            dataset_name,
            config['eer_files_dir'],
            include_patterns
        )
        
        # Merge results
        for model in meta:
            for dataset in meta[model]:
                for id_ in meta[model][dataset]:
                    id2meta[model][dataset][id_] = meta[model][dataset][id_]
                    if dataset in count[model]:
                        for key in count[model][dataset]:
                            id2count[model][dataset][key] = count[model][dataset][key]
    
    # Get subsets with bonafide and spoof samples
    bonafide_subsets = []
    spoof_subsets = []
    
    for dataset in id2count[config['model_name']].keys():
        bonafide_count = id2count[config['model_name']][dataset].get("bonafide", 0)
        spoof_count = id2count[config['model_name']][dataset].get("spoof", 0)
        
        if isinstance(bonafide_count, int) and bonafide_count > 0:
            bonafide_subsets.append(dataset)
        if isinstance(spoof_count, int) and spoof_count > 0:
            spoof_subsets.append(dataset)
    
    print("\nDataset Summary:")
    print("   Bonafide subsets:", ", ".join(f"[{s}]" for s in bonafide_subsets))
    print("   Spoof subsets:", ", ".join(f"[{s}]" for s in spoof_subsets))

    if len(bonafide_subsets) == 0 or len(spoof_subsets) == 0:
        print("ERROR: Must have at least one bonafide and one spoof subset")
        return
    
    # Compute EERs for all combinations
    eers = []
    eer_details = defaultdict(dict)
    
    for bonafide_subset in bonafide_subsets:
        eers_row = []
        for spoof_subset in spoof_subsets:
            # Get synthesizers for this spoof subset
            synthesizers = sorted(id2count[config['model_name']][spoof_subset]["synthesizer"].keys())
            if not synthesizers:
                synthesizers = [None]  # Use None if no synthesizers found
            
            subset_eers = []
            for synthesizer in synthesizers:
                print(f"\nComputing EER:")
                print(f"   Bonafide subset:  [{bonafide_subset}]")
                print(f"   Spoof subset:     [{spoof_subset}]")
                print(f"   Using synthesizer: [{synthesizer if synthesizer else 'all'}]")
                
                y, score = get_subset(
                    id2meta,
                    [None, bonafide_subset, None],
                    [None, spoof_subset, None],
                    [synthesizer] if synthesizer else [None]
                )
                eer, threshold = compute_eer(y, score)
                subset_eers.append(eer)
                
                # Store detailed results
                key = f"{spoof_subset}_{synthesizer if synthesizer else 'all'}"
                eer_details[bonafide_subset][key] = {
                    'eer': eer,
                    'threshold': threshold,
                    'spoof_subset': spoof_subset,
                    'synthesizer': synthesizer
                }
            
            eers_row.extend(subset_eers)
        eers.append(eers_row)
    
    # Create model-specific output directory for EER details
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Save results and plots
    save_results(eers, bonafide_subsets, spoof_subsets, args.output_dir, config['model_name'], id2count)
    plot_cross_testing_matrix(eers, bonafide_subsets, spoof_subsets, args.plot_dir, 
                            config['model_name'], id2count[config['model_name']])
    
    # Save detailed EER results
    eer_details_path = os.path.join(model_output_dir, 'eer_details.json')
    with open(eer_details_path, 'w') as f:
        json.dump(eer_details, f, indent=2)
    
    print(f"\nResults saved successfully:")
    print(f"   EER results and statistics: {os.path.join(args.output_dir, config['model_name'])}")
    print(f"   Visualization plots: {os.path.join(args.plot_dir, config['model_name'])}")

if __name__ == "__main__":
    main() 