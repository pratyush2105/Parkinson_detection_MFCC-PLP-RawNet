#    1. Audio Preprocessing:
#        The audio file is read and pre-emphasis is applied to emphasize high-frequency components.
#        The audio signal is then framed and windowed using the Hamming window to prepare it for further analysis.
#        Power spectrum is computed from the framed audio using the Short-Time Fourier Transform (STFT).
#       Mel filter banks are applied to approximate human auditory perception, and the filter bank energies are computed.
#        Discrete Cosine Transform (DCT) is applied to obtain MFCC features, which are saved in the variable mfcc.

#    2. Data Splitting:
#        The extracted MFCC features are split into training and test sets in a 60:40 ratio.
#       The training set is stored in train_mfcc_features, and the test set is stored in test_mfcc_features.
#       Ground truth labels for training and test sets are defined using train_labels and test_labels, respectively.

#   3. RawNet Model Definition:
#      The RawNet model is defined using the provided architecture, consisting of convolutional layers, ResBlocks, max-pooling, GRU, and fully connected layers.

#  4. Model Training:
#      The model is trained for 100 epochs using the Adam optimizer and CrossEntropyLoss as the loss function.
#      During each epoch, the training loss is printed to monitor the training progress.

#  5. Model Evaluation:
#      After training, the model is set to evaluation mode and tested on the test data (test_mfcc_tensor).
#      The predicted labels are obtained using torch.max and compared to the ground truth test labels (test_labels) to calculate the test accuracy.



import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from scipy.fftpack import dct
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torchvision import models
import torchvision.transforms as transforms
import os


def extract_mfcc_from_folder(folder_path):
    # Create an empty list to store MFCC features from all audio files in the folder
    all_mfcc_features = []

    # List all the audio files in the folder
    audio_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]

    for audio_file in audio_files:
        audio_path = os.path.join(folder_path, audio_file)

        # Load the audio file
        sample_rate, signal = wav.read(audio_path)
        
        frame_size = 500  # Number of samples per frame

        padding = np.zeros(frame_size)  # Create silent padding frames

        audio_with_padding = np.concatenate((padding, signal, padding))

        # Save the modified audio with added silence
        output_audio_path = "output_audio.wav"
        wav.write(output_audio_path, sample_rate, audio_with_padding)

        # Load the modified audio file
        sample_rate, signal = wav.read(output_audio_path)

        # ... (rest of the code for preemphasis, framing, windowing, Mel filter bank, and MFCC extraction)

        #preemphasis - x'[t] = x[t] - alpha*x[t-1]...0.95 < alpha < 0.99
        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        ##framming and windowing
        frame_size = int(0.025 * sample_rate)  # 25ms
        frame_stride = int(0.01 * sample_rate)  # 10ms
        num_frames = int(np.ceil(float(np.abs(len(emphasized_signal) - frame_size)) / frame_stride))

        # Pad the signal to ensure that all frames have equal length
        pad_signal_length = num_frames * frame_stride + frame_size
        padded_signal = np.pad(emphasized_signal, (0, pad_signal_length - len(emphasized_signal)), 'constant')

        # Generate the indices for the frames
        indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_stride, frame_stride), (frame_size, 1)).T

        # Extract the frames using the indices
        frames = padded_signal[indices.astype(np.int32, copy=False)]

        #apply hamming window so that the signals taper smoothly at the edges
        #reduces spectral leakage and improves accuracy of the feature extraction process
        frames *= np.hamming(frame_size)


        NFFT = 512#no. of pts. in FFT for spectral analysis
        magnitudes = np.abs(np.fft.rfft(frames, NFFT))#FFT applied to frames to compute magnitudes(amplitude spectrum)
        power_spectrum = (1.0 / NFFT) * np.square(magnitudes)#calculate power spectrum by squaring mag spec. and scaling them by 1/NFFT

        #MFB is used - to approximate non linear human auditory perception of sound
        num_filters = 40#no. of MMFB used
        #conversion of Hz to Mel scale
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)

        # Generate equally spaced Mel points(used as ref pts. for MFBs)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)

        #Converts mel points back to corres. freq values 
        hz_points = 700 * (10**(mel_points / 2595) - 1)

        # Find the indices corresponding to the points in the power spectrum
        # bin_points represent the bins in the power spectrum that will be used for filtering.
        bin_points = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)

        #initialize an array to store FB coeff.
        filter_bank = np.zeros((num_filters, int(np.floor(NFFT / 2 + 1))))

        #iterate over each filter bank and fill the filter_back array with appropriate filter coeff. 
        for m in range(1, num_filters + 1):
            filter_bank[m - 1, bin_points[m - 1]:bin_points[m]] = (bin_points[m] - bin_points[m - 1]) / (hz_points[m] - hz_points[m - 1])#calculates the slope of the rising edge of the filter
            filter_bank[m - 1, bin_points[m]:bin_points[m + 1]] = (bin_points[m + 1] - bin_points[m]) / (hz_points[m + 1] - hz_points[m])#calculates the slope of the falling edge of the filter

        #apply the MFB coeff. to power spectrum using matrix multiplication to obtain filtered energy values for each FB
        filter_bank_energy = np.dot(power_spectrum, filter_bank.T)
        #this line replaces any 0 values by epsilon to avoid numerical instability issues
        filter_bank_energy = np.where(filter_bank_energy == 0, np.finfo(float).eps, filter_bank_energy)

        filter_bank_energy = 20 * np.log10(filter_bank_energy)  # dB conversion by log application

        num_ceps = 20  # Number of MFCC coefficients to keep
        #apply dct on the filter bank energies - gives mfcc
        mfcc_features = dct(filter_bank_energy, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]

        # Normalize the MFCC features
        mean_mfcc = np.mean(mfcc_features, axis=0)
        std_mfcc = np.std(mfcc_features, axis=0)
        normalized_mfcc = (mfcc_features - mean_mfcc) / std_mfcc


        # Append the MFCC features to the list
        all_mfcc_features.append(normalized_mfcc)

    return all_mfcc_features

# Extract MFCC features for Parkinson's audio files
parkinson_folder_path = "parkinson_folder"
parkinson_mfcc_features_list = extract_mfcc_from_folder(parkinson_folder_path)

# Extract MFCC features for non-Parkinson's audio files
non_parkinson_folder_path = "non_parkinson_folder"
non_parkinson_mfcc_features_list = extract_mfcc_from_folder(non_parkinson_folder_path)

# Concatenate the list of MFCC features into 2D arrays
parkinson_normalized_mfcc = np.concatenate(parkinson_mfcc_features_list, axis=0)
non_parkinson_normalized_mfcc = np.concatenate(non_parkinson_mfcc_features_list, axis=0)

# Save the normalized MFCC features to separate files
output_file_parkinson = "mfcc_parkinson.txt"
output_file_non_parkinson = "mfcc_non_parkinson.txt"

np.savetxt(output_file_parkinson, parkinson_normalized_mfcc)
np.savetxt(output_file_non_parkinson, non_parkinson_normalized_mfcc)







# Assuming you have the MFCC feature matrices for Parkinson's and non-Parkinson's patients
# Replace mfcc_features_parkinson with your actual MFCC feature matrix for Parkinson's patients
# Replace mfcc_features_non_parkinson with your actual MFCC feature matrix for non-Parkinson's patients
mfcc_features_parkinson = parkinson_normalized_mfcc # Replace this with your actual MFCC feature matrix for Parkinson's patients
mfcc_features_non_parkinson = non_parkinson_normalized_mfcc  # Replace this with your actual MFCC feature matrix for non-Parkinson's patients

# Combine the matrices vertically using numpy.concatenate
combined_mfcc_features = np.concatenate((mfcc_features_parkinson, mfcc_features_non_parkinson), axis=0)

# Normalize the MFCC features to have zero mean and unit variance across the dataset
mean = np.mean(combined_mfcc_features, axis=0)
std = np.std(combined_mfcc_features, axis=0)
normalized_mfcc_features = (combined_mfcc_features - mean) / std


# Now the combined_mfcc_features will have shape (num_samples_parkinson + num_samples_non_parkinson, num_features)

# Convert the MFCC feature matrix to PyTorch tensor
mfcc_tensor = torch.tensor(normalized_mfcc_features, dtype=torch.float32)

# Define the ground truth labels for training and test data
# Assuming 80% of data is for training and 20% for testing
num_samples_parkinson = mfcc_features_parkinson.shape[0]
num_samples_non_parkinson = mfcc_features_non_parkinson.shape[0]
num_train_parkinson = int(0.8 * num_samples_parkinson)
num_train_non_parkinson = int(0.8 * num_samples_non_parkinson)


# Split the data for training and testing
train_mfcc_tensor = mfcc_tensor[:num_train_parkinson + num_train_non_parkinson]
test_mfcc_tensor = mfcc_tensor[num_train_parkinson + num_train_non_parkinson:]

# Add a channel dimension to the tensors
train_mfcc_tensor = train_mfcc_tensor.unsqueeze(1)  # Shape: (num_train_parkinson + num_train_non_parkinson, 1, num_ceps)
test_mfcc_tensor = test_mfcc_tensor.unsqueeze(1)    # Shape: (num_test_parkinson + num_test_non_parkinson, 1, num_ceps)

# Define the ground truth labels for training and test data
train_labels_parkinson = torch.tensor([1] * num_train_parkinson, dtype=torch.long)
train_labels_non_parkinson = torch.tensor([0] * num_train_non_parkinson, dtype=torch.long)
train_labels = torch.cat((train_labels_parkinson, train_labels_non_parkinson), dim=0)

test_labels_parkinson = torch.tensor([1] * (num_samples_parkinson - num_train_parkinson), dtype=torch.long)
test_labels_non_parkinson = torch.tensor([0] * (num_samples_non_parkinson - num_train_non_parkinson), dtype=torch.long)
test_labels = torch.cat((test_labels_parkinson, test_labels_non_parkinson), dim=0)



# Assuming you have the RawNet model class and other components defined

#### in PyTorch __init__ and forward func are 2 essential methods used when def. a custom NN model as a subclass of torch.nn.Module
## init = constructor method used to initialize the NN arch. and its params
## forward = where actual forward pass computation happens - def. how input data flows through layers of NN
## to produce final output.



def conv3x3(in_planes, out_planes, stride=1):
    #defines 1D convolution operation with kernel size 3
    #building block for later ResBlocks
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock3x3(nn.Module):
    #implements basic block used in ResBlocks
    #consists of 2 3X3 convolutions with batch normalization and LeakyReLU activation
    #output of 2nd convolution is added to input(residual connection)
    #result is passed through another Leaky ReLU
    #Leaky ReLU activation is applied to introduce non-linearity with a negative slope of 0.01
    #The residual blocks help in effectively training deep architectures and facilitate information flow
    expansion = 1
    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out



class RawNet(nn.Module):
    #this class implements the entire RawNet architecture for speaker recognition
    #firstly, input layer is defined - 1D convolutional layer with 128 output channels
    #kernel size = 3, stride = 3, no padding - this takes mfcc feature matrix as input
    #input_channel - mfcc matrix

    #Architecture - 2 sets of ResBlocks 
    # 1st set - 2 BasicBlock3x3 blocks - each block has 128 input + 128 output channels
    # after each set of ResBlocks - max pooling is applied to reduce temporal dimension by factor of 3
    # 2nd set - 4 BasicBlock3X3 blocks - each block has 256 input + 256 output channels
    
    # after 2nd set of ResBlocks the output is permuted to change the time and feature dim. making it suitable for the GRU layer
    # GRU has 1024 hidden units, drouput is applied with rate of 0.2 to prevent overfitting
    # The GRU layer aggregates the frame-level features into a single utterance-level embedding
    
    # Fully connected layer(speaker embedding) - The GRU's output is passed through a fully connected layer with 128 units, which reduces the dimensionality of the speaker embedding from 1024 to 128
    
    # The speaker embedding obtained from the fully connected layer is connected to the output layer.
    # The number of nodes in the output layer is equal to the number of classes or speakers in the training set (1211 in this case).
    # The output layer produces predictions for the speaker classification task.
    def __init__(self, input_channel, num_classes, dropout_prob=0.2):
        self.inplanes3 = 128
        super(RawNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        #############################################################################
        
        
        self.resblock_1_1 = self._make_layer3(BasicBlock3x3, 128, 1, stride=1)
        self.resblock_1_2 = self._make_layer3(BasicBlock3x3, 128, 1, stride=1)
        self.maxpool_resblock_1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        #############################################################################
        self.resblock_2_1 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.resblock_2_2 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.resblock_2_3 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.resblock_2_4 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.maxpool_resblock_2 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        
        ############################################################################
        self.gru = nn.GRU(input_size=256, hidden_size=1024,dropout=dropout_prob,bidirectional=False,batch_first=True)
        self.spk_emb = nn.Linear(1024,128)
        self.drop = nn.Dropout(p=dropout_prob)
        self.output_layer = nn.Linear(128, num_classes)


    def _make_layer3(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

 
    # this method defines forward pass of the model
    # takes input as MFCC matrix and performs necessary op. through layers of RawNet arch.
    # this produce predictions and speaker embeddings
    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.resblock_1_1(out)
        # out = self.maxpool_resblock_1(out)  # Commented out max pooling
        out = self.resblock_1_2(out)
        # out = self.maxpool_resblock_1(out)  # Commented out max pooling

        out = self.resblock_2_1(out)
        # out = self.maxpool_resblock_2(out)  # Commented out max pooling
        out = self.resblock_2_2(out)
        # out = self.maxpool_resblock_2(out)  # Commented out max pooling
        out = self.resblock_2_3(out)
        # out = self.maxpool_resblock_2(out)  # Commented out max pooling
        out = self.resblock_2_4(out)
        # out = self.maxpool_resblock_2(out)  # Commented out max pooling

        out = out.permute(0, 2, 1)
        out, _ = self.gru(out)
        out = out.permute(0, 2, 1)
        spk_embeddings = self.spk_emb(out[:, :, -1])
        preds = self.output_layer(spk_embeddings)

        return preds, spk_embeddings
    # Each element in 'preds' represents the predicted probability of the corresponding class or speaker
    # spk_embeddings denote utterance-level embedding


# Define batch size
batch_size = 2  # Set the desired batch size

# Create TensorDatasets for training and test data
train_dataset = TensorDataset(train_mfcc_tensor, train_labels)
test_dataset = TensorDataset(test_mfcc_tensor, test_labels)

# Create DataLoaders for training and test data with the updated batch size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Split the dataset into train and validation sets for early stopping and learning rate scheduling
train_mfcc_tensor, val_mfcc_tensor, train_labels, val_labels = train_test_split(
    train_mfcc_tensor, train_labels, test_size=0.2, random_state=42)

# Create TensorDatasets for training and validation data
train_dataset = TensorDataset(train_mfcc_tensor, train_labels)
val_dataset = TensorDataset(val_mfcc_tensor, val_labels)

# Create DataLoaders for training and validation data with the updated batch size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Create an instance of the RawNet model with dropout regularization
input_channel = 20  # Number of MFCC features
num_classes = 2  # Binary classification: Parkinson's patients or non-Parkinson's patients
dropout_prob = 0.2  # Adjust dropout probability as needed
rawnet_model = RawNet(input_channel=input_channel, num_classes=num_classes, dropout_prob=dropout_prob)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rawnet_model.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1, verbose=True)

# Training loop with early stopping
best_val_accuracy = 0.0
patience = 5  # Number of epochs to wait before stopping if validation accuracy does not improve
early_stopping_counter = 0


# Training model
num_epochs = 100
for epoch in range(num_epochs):
    rawnet_model.train()
    train_loss = 0.0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs, _ = rawnet_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Compute training accuracy
    rawnet_model.eval()
    with torch.no_grad():
        train_correct = 0
        for inputs, labels in train_dataloader:
            outputs, _ = rawnet_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        train_accuracy = train_correct / len(train_dataset)

    # Compute validation accuracy
    with torch.no_grad():
        val_correct = 0
        for inputs, labels in val_dataloader:
            outputs, _ = rawnet_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
        val_accuracy = val_correct / len(val_dataset)

    # Update learning rate scheduler based on validation accuracy
    scheduler.step(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss / len(train_dataloader):.4f}, "
          f"Train Accuracy: {train_accuracy:.4f}, "
          f"Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping check
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0
        # Save the best model if desired
        torch.save(rawnet_model.state_dict(), "best_model.pt")
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print("Early stopping.")
        break

# Load the best model for testing
rawnet_model.load_state_dict(torch.load("best_model.pt"))

# Evaluate on the test set
rawnet_model.eval()
with torch.no_grad():
    test_correct = 0
    for inputs, labels in test_dataloader:
        outputs, _ = rawnet_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum().item()
    test_accuracy = test_correct / len(test_dataset)

print(f"Test Accuracy: {test_accuracy:.4f}")
