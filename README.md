# Turns Analysis

This repository contains code for the manuscript (INSERT MANUSCRIPT NAME HERE).

## Setup

Our code is built with Python 3.10.x and CUDA 12.x. We recommend to create a conda virtual environment using the `environment.yaml` provided in this repository.

We use [TRESTLE](https://github.com/LinguisticAnomalies/harmonized-toolkit) for pre-processing. Please also install TESTLE's dependencies first.

To use spacy, please run `python -m spacy download en_core_web_trf` after creating the corresponding virtual environment.

Before start, please create a config.ini file under the scripts folder, using the following template:

```
[DATA]
component_input = /path/to/raw/data/
component_output = /path/to/preprocessed/data/
[META]
component_meta = /path/to/metadata/
```

## Folders
The structure of this repo is listed as follows

```
├── data
│   ├── component_1_preprocessed_data.csv
│   ├── component_2_preprocessed_data.csv
├── scripts
│   ├── preprocess_talkbank.py
│   ├── match.R
│   ├── turn.Rmd
```

## To reproduce
### Preprocessing

We use [TRESTLE](https://github.com/LinguisticAnomalies/harmonized-toolkit) for preprocessing. Other preprocessing information can be found in `preprocess_talkbank.py`

To get the Pitt corpus preprocessing result on healthy controls participants, one can run:
```shell
python preprocess_talkbank.py pitt --subset control --indicator *PAR --preprocess
```
Similarly, to get the WLS corpus preprocessing result on participants, one can run:
```shell
python preprocess_talkbank.py wls --indicator *PAR --preprocess
```

For the detailed analysis, please refer to `turn.Rmd` and `match.R` for details.