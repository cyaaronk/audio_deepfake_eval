# Dataset Download Instructions

This document provides instructions for downloading each dataset supported by the Audio Deepfake Detection Eval framework.

After downloading the datasets, you may:
1. Run your own audio deepfake detection model to obtain the classification scores for each audio in the datasets
2. Tidy your score files and put them in `eer_files/`. The structure of your score files should be similar to the example score files placed in `eer_files/`
3. Evaluate your model's performance by following the README.md in the root directory

## 🚀 Easy Download

For convenience, we provide a pre-processed version of the datasets to evaluate and package them into a single ZIP file for download. 

> 💡 **Note:** To reduce the dataset size (From 2TB to 6GB), we randomly sampled 600 audios from each bona fide speech type and synthesizer type.

### 📦 Quick Start

1. **Download** the pre-processed datasets zip file from [Zenodo](https://zenodo.org/records/15514835)
2. **Extract** the zip file to your workspace
3. **Verify** the dataset structure:

```
datasets/
├── ami_ihm/
│   ├── audios/
│   │   └── *.wav
│   ├── ami_ihm.cm.eval.trl.txt
│   └── trial_metadata.txt
├── ami_sdm/
│   ├── audios/
│   │   └── *.wav
│   ├── ami_sdm.cm.eval.trl.txt
│   └── trial_metadata.txt
├── asvspoof_2021_DF/
│   ├── audios/
│   │   └── *.wav
│   ├── asvspoof_2021_DF.cm.eval.trl.txt
│   └── trial_metadata.txt
└── ... (other datasets)
```

### 📁 Dataset Structure

Each dataset directory contains:

| Component | Description |
|-----------|-------------|
| `audios/` | Directory containing the audio files |
| `*.cm.eval.trl.txt` | List of audio IDs included in the dataset |
| `trial_metadata.txt` | Contains metadata for each audio file |

#### Metadata Fields
- **Audio ID**: Unique identifier for each audio file
- **Label**: Classification (bonafide/spoof)
- **Codec Information**: Audio codec details
- **Synthesizer Information**: Details of the synthesis method (if applicable)

### ⚠️ Important Notes

- 🔒 Only datasets with redistribution-friendly licenses are included
- 📊 Each dataset is randomly sampled to include 600 audios per type
- 💾 The pre-processed version is significantly smaller than the full datasets
- 📝 Original dataset licenses and citations are preserved

---

## 📥 Manual Download Instructions

If you need the full datasets or specific datasets not included in the pre-processed version, follow the manual download instructions below.

## Table of Contents
- [Speech Datasets](#speech-datasets)
  - [LibriSpeech](#librispeech)
  - [AMI Corpus](#ami-corpus)
  - [VCTK](#vctk)
- [Deepfake Detection Datasets](#deepfake-detection-datasets)
  - [ASVspoof 2019](#asvspoof-2019)
  - [ASVspoof 2021 DF](#asvspoof-2021-df)
  - [AVDeepfake1M](#avdeepfake1m)
  - [CodecFake](#codecfake)
  - [EmoFake](#emofake)
  - [FakeAVCeleb](#fakeavceleb)
  - [LlamaPartialSpoof](#llamapartialspoof)
  - [MLAAD](#mlaad)
  - [In the Wild](#in-the-wild)
  - [SceneFake](#scenefake)


## Speech Datasets

### LibriSpeech
1. Visit [OpenSLR](https://www.openslr.org/12)
2. Download the desired test sets:
   - `test-clean`: Standard test set
   - `test-other`: More challenging test set
3. Each archive contains audio files and transcripts

### AMI Corpus
1. Visit [Hugging Face](https://huggingface.co/datasets/edinburghcstr/ami)
2. Download the desired test sets:
   - Individual headset microphone (ihm): `https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/*.tar.gz`
   - Single distant microphone (sdm): `https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/*.tar.gz`

### VCTK
1. Visit [University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2950)
2. Follow the download instructions on the page
3. Dataset includes multiple speakers with various accents

## Deepfake Detection Datasets

### ASVspoof 2019
1. Visit [University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)
2. Follow the DataShare instructions to access the files
3. Download the LA (Logical Access) partition

### ASVspoof 2021 DF
1. Visit [Zenodo](https://zenodo.org/records/4835108)
2. Download the .tar.gz files (~8.6GB each)
3. Extract the archives to access the dataset

### AVDeepfake1M
1. Download the EULA from [GitHub](https://github.com/ControlNet/AV-Deepfake1M/blob/master/eula.pdf)
2. Sign the EULA
3. Email the signed EULA to dataset maintainers
4. Wait for download link to be provided

### CodecFake
1. Visit the [CodecFake GitHub repository](https://github.com/roger-tseng/CodecFake)
2. Download ZIP files from [Hugging Face](https://huggingface.co/datasets/rogertseng/CodecFake_wavs/tree/main)
   - `academicodec_hifi_16k_320d.zip`
   - `audiodec_24k_320d.zip`
   - `descript-audio-codec-16khz.zip`
   - `encodec_24khz.zip`
   - `funcodec-funcodec_en_libritts-16k-nq32ds320.zip`
   - `SpeechTokenizer.zip`

### EmoFake
1. Visit [Zenodo](https://zenodo.org/records/10443769)
2. Download the dataset (8.8GB)
3. Extract the archive

### FakeAVCeleb
1. Visit the [FakeAVCeleb GitHub repository](https://github.com/DASH-Lab/FakeAVCeleb)
2. Fill out the access request form
3. Wait for approval and download instructions
4. Dataset includes both video and synthesized audio deepfakes

### LlamaPartialSpoof
1. Visit [Zenodo](https://zenodo.org/records/14214149)
2. Download the dataset files
3. Follow the provided documentation for dataset organization

### MLAAD
1. Visit [Deepfake Total](https://deepfake-total.com/mlaad)
2. Follow the instructions to request access
3. Download the dataset (MLAAD v3) once access is granted

### In the Wild
1. Visit [Deepfake Total](https://deepfake-total.com/in_the_wild)
2. Follow the access request procedure
3. Contains real-world deepfake examples

### SceneFake
1. Visit [Zenodo](https://zenodo.org/records/7663324)
2. Download the dataset
3. Extract and organize according to documentation

## Additional Notes

1. Some datasets require signing agreements or requesting access. Plan accordingly as approval may take time.
2. Large datasets are often split into multiple files. Ensure you have sufficient storage space.
3. Check the documentation of each dataset for:
   - License terms
   - Citation requirements
   - Specific usage restrictions

## Storage Requirements

Full Dataset size: <2TB in total

Please ensure you have adequate storage space before downloading. 