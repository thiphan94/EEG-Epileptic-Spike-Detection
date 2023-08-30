# EEG Epileptic Spike Detection Project

## Overview

Epileptic spike detection in EEG data is a significant challenge in both data science and neural science. The primary objective of this project is to develop an algorithm capable of automatically detecting whether a given instance, such as an EEG data series, contains an epileptic spike or not. This task is pivotal for accurate diagnosis and treatment of epilepsy. The project addresses two key challenges:

1. **Class Imbalance:** Epileptic spike behaviors are minority instances when compared to normal behaviors in EEG data.
2. **Interpatient Variability:** The morphologies of epileptic spikes vary considerably across different patients.

The scope of this project focuses on the analysis of multi-channel EEG data series. The provided datasets stem from two distinct experiments and have undergone preprocessing to ensure consistency.

## Dataset

The dataset employed in this project is composed of multiple multivariate data series, each maintaining a consistent frequency. These data series originate from two separate experiments and have been divided into fixed-length subsequences. Each subsequence includes five channels, with each channel containing 768 sample points, temporally aligned.

The data series are labeled as follows:

- **Positive Instances (Label 1):** These instances represent EEG data segments containing epileptic spikes. The occurrence of a spike is indicated by a vertical line in the corresponding plot.
- **Negative Instances (Label 0):** These instances encompass normal EEG data series.

The visual depiction of the two types of data series is shown in Figure 1, with the positive example featuring a marked vertical line representing the timing of the epileptic spike.

![Figure 1: Positive and Negative Instances](link_to_image)

## Model
To address the challenge of epileptic spike detection, a ResNet model has been employed. The model has been trained on the provided EEG data series to automatically identify the presence of epileptic spikes. The project provides a script train.py that allows you to train the model using various settings.

Getting Started
```
$ git clone https://github.com/thiphan94/EEG-Epileptic-Spike-Detection.git
$ cd EEG-Epileptic-Spike-Detection
$ pip install -r requirements.txt
$ python train.py -M resnet
```

## Conclusion
The EEG Epileptic Spike Detection Project aims to contribute to the accurate detection of epileptic spikes in EEG data series, which is essential for effective diagnosis and treatment of epilepsy. The provided ResNet model serves as a step toward addressing this challenge. By leveraging machine learning techniques, we strive to enhance our understanding of epilepsy and pave the way for more efficient healthcare practices.

For more details and contributions, please refer to the repository and feel free to explore the code and documentation. Your feedback and suggestions are highly appreciated.

