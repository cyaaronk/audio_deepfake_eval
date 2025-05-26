#!/bin/bash


# LibriSpeech
# Can directly include in custom dataset
wget https://us.openslr.org/resources/12/test-clean.tar.gz
wget https://us.openslr.org/resources/12/test-other.tar.gz

# AMI Corpus
# Can directly include in custom dataset
## IHM eval
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/EN2002a.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/EN2002b.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/EN2002c.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/EN2002d.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/ES2004a.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/ES2004b.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/ES2004c.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/ES2004d.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/IS1009a.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/IS1009b.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/IS1009c.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/IS1009d.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/TS3003a.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/TS3003b.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/TS3003c.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/ihm/eval/TS3003d.tar.gz
## SDM eval
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/EN2002a.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/EN2002b.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/EN2002c.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/EN2002d.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/ES2004a.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/ES2004b.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/ES2004c.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/ES2004d.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/IS1009a.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/IS1009b.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/IS1009c.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/IS1009d.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/TS3003a.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/TS3003b.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/TS3003c.tar.gz
wget https://huggingface.co/datasets/edinburghcstr/ami/blob/main/audio/sdm/eval/TS3003d.tar.gz

# VCTK
# Can directly include in custom dataset
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y

# ASVSpoof2019 LA
# Can directly include in custom dataset
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y

# ASVSpoof2021 DF
# Can directly include in custom dataset
wget https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz?download=1
wget https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part01.tar.gz?download=1
wget https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part02.tar.gz?download=1
wget https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part03.tar.gz?download=1

# AVDeepfake1M
# Manual download required. Please refer to datasets/README.md for details

# CodecFake
# Download and process
wget https://huggingface.co/datasets/rogertseng/CodecFake_wavs/blob/main/academicodec_hifi_16k_320d.zip
wget https://huggingface.co/datasets/rogertseng/CodecFake_wavs/blob/main/audiodec_24k_320d.zip
wget https://huggingface.co/datasets/rogertseng/CodecFake_wavs/blob/main/descript-audio-codec-16khz.zip
wget https://huggingface.co/datasets/rogertseng/CodecFake_wavs/blob/main/encodec_24khz.zip
wget https://huggingface.co/datasets/rogertseng/CodecFake_wavs/blob/main/funcodec-funcodec_en_libritts-16k-nq32ds320.zip
wget https://huggingface.co/datasets/rogertseng/CodecFake_wavs/blob/main/SpeechTokenizer.zip

# EmoFake
# Can directly include in custom dataset
wget https://zenodo.org/records/10443769/files/EmoFake.tar.gz?download=1

# FakeAVCeleb
# Manual download required. Please refer to datasets/README.md for details

# LlamaPartialSpoof
# Can directly include in custom dataset
wget https://zenodo.org/records/14214149/files/label_R01TTS.0.a.txt?download=1
wget https://zenodo.org/records/14214149/files/label_R01TTS.0.b.txt?download=1
wget https://zenodo.org/records/14214149/files/R01TTS.0.a.tgz?download=1
wget https://zenodo.org/records/14214149/files/R01TTS.0.b.tgz?download=1

# MLAAD
# Manual download required. Please refer to datasets/README.md for details

# In The Wild
# Can directly include in custom dataset
wget https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download

# SceneFake
# Download and process
wget https://zenodo.org/records/7663324/files/SceneFake.zip?download=1