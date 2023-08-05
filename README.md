# Parkinson's Disease Detection using RawNet+MFCC+PLP

This repository contains code for detecting Parkinson's disease in audio files using a combination of MFCC (Mel-Frequency Cepstral Coefficients), PLP (Perceptual Linear Prediction), and a deep learning architecture called RawNet. The RawNet architecture is enhanced with MFCC features and aims to classify audio files as belonging to either a Parkinson's patient or a healthy person based on the training, validation, and test data.

## Overview

Parkinson's disease is a neurodegenerative disorder that affects motor functions. It can be diagnosed through various means, including analyzing audio recordings of patients speaking. This project focuses on building a deep learning-based classifier that utilizes MFCC and PLP features in conjunction with the RawNet architecture to automatically detect Parkinson's disease in audio files.

## Repository Structure

The repository is organized as follows:

1. `mfcc.py`: This script extracts 13 MFCC features from an audio file using standard signal processing techniques.

2. `plp.py`: This script extracts 13 PLP features from an audio file using perceptual linear prediction.

3. `mfcc+Rawnet.py`: This script combines the MFCC feature extraction with RawNet and LSTM-CNN-based architecture to classify audio files as healthy or Parkinson's affected. It also adds 500 frames of silence before and after the audio to improve the model's performance.

4. `data/`: This directory contains the training, validation, and test data needed to train and evaluate the model. The dataset should be organized in separate folders for healthy and Parkinson's patients.

## How to Use

To use the code in this repository, follow these steps:

1. Place the audio dataset in the `data/` directory. Organize the audio files in separate folders for healthy and Parkinson's patients.

2. Run `mfcc.py` to extract MFCC features from the audio dataset.

3. Run `plp.py` to extract PLP features from the audio dataset.

4. Run `mfcc+Rawnet.py` to train the RawNet+MFCC+PLP model using the extracted features and audio data.

5. The script will automatically split the dataset into training, validation, and test sets, train the model, and evaluate its performance.


## Contribution

Contributions to this project are welcome! If you find any issues or improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The RawNet architecture implementation is based on [[cite the relevant paper or repository](https://r.search.yahoo.com/_ylt=AwrKFYmQTc5klUMM.qe7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Ny/RV=2/RE=1691270673/RO=10/RU=https%3a%2f%2farxiv.org%2fabs%2f1904.08104/RK=2/RS=oq5bsTOTSlt8Ze6CpuxcocVa794-)].
- The MFCC and PLP feature extraction have been adapted from existing sources (please cite them appropriately).
- The dataset used in this project is [[cite the dataset source if applicable](https://r.search.yahoo.com/_ylt=AwrKB76wTc5koXElJyy7HAx.;_ylu=Y29sbwNzZzMEcG9zAzEEdnRpZAMEc2VjA3Ny/RV=2/RE=1691270704/RO=10/RU=https%3a%2f%2fzenodo.org%2frecord%2f2867216/RK=2/RS=sR5MvlmGHF5dzDi.WJponQmF0vY-)].

## Disclaimer

This project is for research and educational purposes only and should not be used as a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare provider for proper diagnosis and treatment of Parkinson's disease or any other medical condition.
