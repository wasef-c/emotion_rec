# emotion_rec
Emotion Recognition 

The  proposed model will be a multi-class LSTM model for emotion recognition of speech signals. 

It will uses data from the following datasets

      Toronto Emotional Speech Set Data (TESS)
      
      Ryerson Audio-Visual Database of Emotional Speech and Song(RAVDESS)
      
      Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D)
      
      Interactive Emotional Dyadic Motion Capture (IEMOCAP)



The Target column (y) will be the emotion classification.

Feature extraction will be done to extract Mel-frequency cepstral coefficients as numerical features

**STEPS FOR DEPLOYMENT**

**1. DATA_PREPROCESSING.ipynb**
   This notebook goes through the process of retrieving the data and producing the MFC coefficicents
   
**2.Emotion_Classification_ModelComparison.ipynb**
  This notebook tests multiple model architectures and uses pycaret to compare different models
  
**3. CNN_Training_BiDirectional_LSTM**
  This notebook trains a bi-directional LSTM model and utilizes hyperparameter tuning
  
**4. Bi_DirectionalLSTM_Performance_Evaluation.ipynb**
  This notebook evaluates the performance of the final bi-directional LSTM model before and after hyperparameter tuning
