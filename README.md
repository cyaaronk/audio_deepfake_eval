# Audio Deepfake Detection Eval

<p align="center">
    <br>
    <img src="docs/en/_static/images/add_eval_logo.png"/>
    <br>
</p>

<p align="center">
  <a href="README_zh.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<a href="https://github.com/ntu-dtc/audio_deepfake_eval/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>

A comprehensive evaluation framework for audio deepfake detection (ADD) from [NTU DTC](https://www.ntu.edu.sg/dtc), providing standardized tools for computing Equal Error Rates (EERs) and analyzing model performance across diverse datasets.

## 📋 Contents
- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Installation](#️-installation)
- [Quick Start](#-quick-start)
- [Advanced Usage](#-advanced-usage)
- [Supported Datasets](#-supported-datasets)
- [Roadmap](#-roadmap)
- [News](#-news)

## 📝 Introduction

Standard ADD evaluations often rely on datasets that unevenly represent different synthesizers and lack diversity in bona fide speech, leading to biased and less reliable Equal Error Rate (EER) measurements.

Our framework introduces bona fide cross-testing, incorporating multiple bona fide speech datasets to ensure balanced assessments. By aggregating EERs across diverse speech types, BonaFideCrossEval improves robustness, interpretability, and real-world applicability.

We benchmark over 150 synthesizers across nine bona fide speech types and release a new dataset to facilitate further research. 🚀

<p align="center">
  <img src="docs/en/_static/images/eval_framework.png" width="100%">
  <br>Audio Deepfake Detection Eval Framework
</p>

## ✨ Key Features

1. **Spoof Cross Testing**: Evaluate models by pairing a single bona fide speech type with multiple synthesizer datasets
2. **Bona Fide Cross Testing**: Test against diverse bona fide speech types for robust evaluation
3. **EER Aggregation**: Summarize results using maximum pooling to identify challenging synthesizers
4. **Comprehensive Visualization**: Generate detailed plots and statistics for analysis
5. **Flexible Dataset Support**: Evaluate on standard and custom datasets

## 🛠️ Installation

### Method 1: Using pip

1. Create and activate a virtual environment (recommended)
   ```bash
   # Create virtual environment
   python -m venv add_eval_env
   
   # Activate virtual environment
   # On Windows
   .\add_eval_env\Scripts\activate
   # On Unix or MacOS
   source add_eval_env/bin/activate
   ```

2. Install dependencies
   ```bash
   pip install -r add_detect_eval/requirements.txt
   ```

### Method 2: Manual Installation

```bash
git clone https://github.com/ntu-dtc/audio_deepfake_eval.git
cd audio_deepfake_eval
pip install -r add_detect_eval/requirements.txt
```

### Dependencies
- Python >= 3.8
- numpy
- scipy
- matplotlib
- pandas
- PyYAML
- scikit-learn

## 🚀 Quick Start

### 1. Prepare Configuration

Create `config.yaml`:
```yaml
model_name: "conformer-based-classifier-for-anti-spoofing"
eer_files_dir: "./eer_files"

datasets:
  - name: "asvspoof_2021_DF"
    include_patterns:
      - ".wav"
      - "p227"
      - "p228"
  - name: "academicodec_hifi_16k_320d"
    include_patterns:
```

The `include_patterns` field is optional for each dataset. When specified:
- Only audio samples whose IDs match any of the patterns will be included in the evaluation
- Patterns can be file extensions (e.g., ".wav"), speaker IDs (e.g., "p227"), or any other ID pattern
- If not specified, all samples in the dataset will be evaluated
- Multiple patterns can be specified to include different subsets of samples

For example, to evaluate only samples from speakers p227 and p228:
```yaml
datasets:
  - name: "vctk"
    include_patterns:
      - "p227"  # Include samples from speaker p227
      - "p228"  # Include samples from speaker p228
```

### 2. Run Evaluation

```bash
python add_detect_eval/compute_eers.py --config config.yaml --dataset asvspoof_2021_DF
```

If you don't specify a dataset, the script will show available datasets:
```
ERROR: Please specify a dataset to evaluate using --dataset

Available datasets in config:
   asvspoof_2021_DF
   academicodec_hifi_16k_320d

Example usage:
   python add_detect_eval/compute_eers.py --config config.yaml --dataset <dataset_name>
```

Example output:
```
Processing dataset: [asvspoof_2021_DF]
   Including patterns: *.wav

Computing EER...
EER: 3.47%

Codec statistics:
   mp3: 500 files
   aac: 300 files
   opus: 200 files

Synthesizer statistics:
   fastspeech2: 400 files
   tacotron2: 300 files
   glow-tts: 300 files
```

### 3. Visualize Results

```bash
python add_detect_eval/visualize_stats.py \
  --config config.yaml \
  --dataset asvspoof_2021_DF \
  --output-dir ./visualizations
```

If you don't specify a dataset, the script will show available datasets:
```
ERROR: Please specify a dataset to visualize using --dataset

Available datasets in config:
   asvspoof_2021_DF
   academicodec_hifi_16k_320d

Example usage:
   python add_detect_eval/visualize_stats.py --config config.yaml --dataset <dataset_name>
```

Example output:
```
Processing dataset: [asvspoof_2021_DF]
   Including patterns: *.wav

Visualization saved to ./visualizations/asvspoof_2021_DF_statistics.png
```

## 🔍 Advanced Usage

### Cross-Testing Evaluation

Evaluate models across different combinations of bonafide and spoof datasets:

```bash
python add_detect_eval/compute_eers_cross_testing.py \
  --config config.yaml \
  --output-dir ./output \
  --plot-dir ./visualizations
```

#### Example Output Structure

```
output/
├── eer_matrix.csv         # Cross-testing EER results in CSV format
├── eer_details.json       # Detailed EER calculations and thresholds
└── results.json          # Summary statistics and metadata

visualizations/
└── cross_testing_matrix.png  # Heatmap visualization of EER matrix
```

#### Example Log Output

```
Processing dataset: [asvspoof_2021_DF]
   Including patterns: *.wav

Dataset Summary:
   Bonafide subsets: [librispeech_clean], [vctk]
   Spoof subsets: [asvspoof_2021_DF], [av_deepfake_1m]

Computing EER:
   Bonafide subset:  [vctk]
   Spoof subset:     [av_deepfake_1m]
   Using synthesizer: [vits_word]
Sample counts:
   Spoof samples:    180
   Bonafide samples: 755
EER: 3.47%

Results saved successfully:
   EER results and statistics: ./output
   Visualization plots: ./visualizations
```

#### Example Results

1. **EER Matrix (eer_matrix.csv)**:
```csv
Dataset,librispeech_clean,librispeech_other,vctk
fastspeech2,0.0347,0.0412,0.0389
tacotron2,0.0523,0.0678,0.0456
glow-tts,0.0434,0.0567,0.0412
```

2. **Detailed Results (eer_details.json)**:
```json
{
  "librispeech_clean": {
    "asvspoof_2021_fastspeech2": {
      "eer": 0.0347,
      "threshold": -3.621,
      "spoof_subset": "asvspoof_2021_DF",
      "synthesizer": "fastspeech2"
    }
  }
}
```

3. **Cross-Testing Matrix Visualization**:

<p align="center">
  <img src="docs/en/_static/images/cross_testing_matrix.png" width="50%">
  <br>
  <em>Cross-Testing EER Matrix Heatmap</em>
</p>

The heatmap shows EER values (%) for each combination of:
- X-axis: Bonafide speech datasets
- Y-axis: Synthesizer IDs
- Color intensity: EER value (darker = higher EER)

For detailed documentation, see our [📖 User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/index.html).

## 📊 Supported Datasets

### Audio Codecs
✅ CodecFake  
✅ academicodec_hifi_16k_320d  
✅ audiodec_24k_320d  
✅ descript-audio-codec-16khz  
✅ encodec_24khz  
✅ funcodec-funcodec_en_libritts-16k-nq32ds320  

### Speech Datasets  
✅ librispeech_test_clean  
✅ librispeech_test_other  
✅ ami_ihm  
✅ ami_sdm  
✅ vctk  

### Deepfake Detection Datasets
✅ asvspoof_2021_DF  
✅ asvspoof2019_la  
✅ av_deepfake_1m  
✅ emofake  
✅ fakeavceleb  
✅ llamapartialspoof_r01tts0a  
✅ llamapartialspoof_r01tts0b  
✅ mlaad  
✅ partialspoof  
✅ release_in_the_wild  
✅ scenefake  
✅ speech_tokenizer  

## 📈 Leaderboard

<p align="center">
  <img src="docs/en/_static/images/leaderboard.jpeg" width="100%">
  <br>
  <em>Model Performance Comparison on Different Datasets</em>
</p>

The leaderboard shows Equal Error Rate (EER) performance of different models across various datasets. Lower EER indicates better performance in distinguishing between genuine and spoofed audio.

## 🔜 Roadmap

### Completed
✅ Single dataset EER evaluation  
✅ Compute audio statistics for datasets  
✅ Audio statistics visualization  
✅ Cross testing EER evaluation  
✅ Cross testing EER visualization  

### In Progress
⬜ Latex table results generation  
⬜ Support for new audio codecs  
⬜ Integration with more deepfake detection models  
⬜ Real-time evaluation pipeline  

## 🎉 News

- 🔥 **[2025.02.28]** Official release with benchmark results for 164 synthesizers and 9 bona fide speech styles
- 🔥 **[2024.09.18]** Documentation released with technical research and discussions. [📖 Read more](https://evalscope.readthedocs.io/en/refact_readme/blog/index.html)

> ⭐ If you like this project, please star it! Your support motivates us to keep improving.
