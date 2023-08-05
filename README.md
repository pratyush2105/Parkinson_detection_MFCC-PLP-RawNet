# Parkinson's_detection_using_RawNet+MFCC+PLP

## Structure
 - mfcc.py - extracts MFCC features from an audio file(13 MFCC)
 - plp.py - extracts PLP features from an audio file(13 PLP)
 - mfcc+Rawnet.py - extracts MFCC features, adds 500 frames of silences before and after and then applies a CNN and LSTM-CNN-based architecture which classifies an audio files as a Parkinson's patient/ a healthy person based on training, validation and test data.
