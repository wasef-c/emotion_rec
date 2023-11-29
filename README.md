# emotion_rec
Emotion Recognition 

The  proposed model will be a multi-class LSTM model for emotion recognition of speech signals. 

It uses data from the following datasets

      Toronto Emotional Speech Set Data (TESS) - Two actresses, 2800 samples
      
      Ryerson Audio-Visual Database of Emotional Speech and Song(RAVDESS) - 24 Actors, 1440 samples
      
      Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D) - 91 Actors,  7,442 samples
      
      Interactive Emotional Dyadic Motion Capture (IEMOCAP) - 10 actors, 4,532 samples

In total, over  16,000 samples are used, produced by 127 different actors/voices



The Target column (y) will be the emotion classification.

Feature extraction will be done to extract Mel-frequency cepstral coefficients as numerical features

**STEPS FOR DEPLOYMENT**


   **_NOTE: if you don't want to download over 30gb of original data from the audio datasets, numpy arrays of the MFCCs and labels are available under the release tagged Data in X_Tr008, Y_Tr008 for training and X_Te008 and Y_Te008 for testing. That way you can start at Step 2_**
   

**1. DATA_PREPROCESSING.ipynb**
   This notebook goes through the process of retrieving the data and producing the MFC coefficicents

**2.Emotion_Classification_ModelComparison.ipynb**
  This notebook tests multiple model architectures and uses pycaret to compare different models
  
**3. CNN_Training_BiDirectional_LSTM**
  This notebook trains a bi-directional LSTM model and utilizes hyperparameter tuning
  
**4. BiLSTM_Training.ipynb**
  This notebook evaluates the performance of the final bi-directional LSTM model before and after hyperparameter tuning
