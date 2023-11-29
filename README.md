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



  **ABSTRACT**
  
Emotion recognition is a rapidly growing field of research in the world of AI. However, many existing approaches require multi-modal inputs (ecg, facial features, speech etc). This project specifically focuses on emotion recognition of vocal speech features. Additionally, a novel approach is taken by making use of four separate datasets comprising over 16,000 samples, produced by 127 different voices to ensure comprehensiveness and improve results on real world data. After careful consideration of various models and feature sets, Mel-Frequency Cepstral Coefficients (MFCC) emerged as the focal point for audio sample representation. A Hybrid LSTM model with Attention was deployed on the data and achieved an accuracy of 60.38% and an F1 score of 0.60 after rigorous hyperparameter optimization. The final model is a robust RNN with less built in bias than existing models. 


**LITERATURE REVIEW**

The field of speech emotion recognition (SER) faces challenges in artificial intelligence due to the variation in individual expression features. Additionally, the features themselves are less relevant but their characteristics between features. For example, happiness may have a higher pitch range than neutral [1]. To address this, a CNN with Long Short-Term Memory (LSTM) and Transformer Encoder model was developed to discern temporal dependencies in speech signals [1]. 
A hybrid bi-directional LSTM, using MFCC, showcased significant recognition improvements over traditional classifiers, achieving notable performance with an accuracy of 75.62% and on the RAVDESS database alone, which consists of 24  professional actors [2]. After conducting vocal track length perturbation and layer optimizing, an accuracy of 61.7% was achieved on the IEMOCAP database alone which only consists of 10 different voices [1].
Moreover, a novel approach utilizing Mel Spectrogram in conjunction with Mel Frequency Cepstral Coefficients (MFCC) was proposed, enhancing the 1D CNN with two unidirectional LSTM layers. The model was trained on the IEMOCAP database. Results indicated superior performance compared to baseline models with a combined accuracy of 58.29% when using speech only [3]. 
Additionally, a Random Forest Model was deployed on the RAVDESS database to avoid overfitting. Bagging techniques were used to extend functionality and provided an unbiased generalized classification [4]. However, the predictions on Random Forest are less interpretable with data this complex. An accuracy of 67% was achieved on the 24 different voices [4]
However, all these papers were focusing on one or two smaller datasets that consist of 10-20 voices, and are therefore not a good representation of emotion recognition on a real-world scale. The model being proposed in this project is a multi-dataset Hybrid bidirectional LSTM with activation which uses 4 datasets by a combined total of 127 voices and 16,000 samples. 

[1] M. Soleymani, S. Asghari-Esfeden, S. Fu, J. P. Bartlett, L. Schmidt, and M. Pantic, "A Multimodal Database for Affect Recognition and Implicit tagging," in IEEE Transactions on Affective Computing, vol. 3, no. 1, pp. 42-55, Jan.-March 2012, doi: 10.1109/T-AFFC.2011.37. [Online]. Available: https://arxiv.org/pdf/1802.05630v2.pdf

[2] F. Andayani, L. B. Theng, M. T. Tsun, and C. Chua, "Hybrid LSTM-Transformer Model for Emotion Recognition From Speech Audio Files," in IEEE Access, vol. 10, pp. 1-1, 2022, doi: 10.1109/ACCESS.2022.3163856. [Online]. Available: https://doi.org/10.1109/ACCESS.2022.3163856
IEEE Access, 10. https://doi.org/10.1109/ACCESS.2022.3163856

[3] B. T. Atmaja, K. Shirai and M. Akagi, "Speech Emotion Recognition Using Speech Feature and Word Embedding," 2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), Lanzhou, China, 2019, pp. 519-523, doi: 10.1109/APSIPAASC47483.2019.9023098.

[4] A. Madhavi, Albert Valentina, Mounika Karakavalasa, Rohit Boddeda, and Sheripally Nagma, "Comparative Analysis of Different Classifiers for Speech Emotion Recognition," in Proceedings of [Conference Name or Book Title], pp. 523-538, 2021, doi: 10.1007/978-981-15-9293-5_48.


**FEATURE ENGINEERING**

FEATURE: Numerical
For emotion recognition Mel-frequency cepstral coefficients (MFCC) are the industry standard feature to extract. MFCCs are effective in compressing the information present in the speech signal by representing it in a more compact form. MFCC features can reach a high recognition rate compared to other cepstrum features because it is a short term spectral based feature that leads to extract a rich amount of information from speech signals [5]. The human auditory system is more sensitive to changes in frequencies at lower ranges. MFCCs utilize a mel-frequency scale, which is more aligned with the human auditory system's perception of sound. This makes the features extracted by MFCCs more perceptually relevant for human speech [5].

No other features were used due to these reasons, the MFCC is more than sufficient on its own and any other features (e.g. pitch, pauses etc) are all represented within the MFCC.

To conduct feature analysis, the MFCCs were reduced into 435 unique numerical features for each audio sample. After feature analysis the 100 least important features were dropped. The overall performance of the model was largely unchanged.

[5] Abdul, Z. K., & Al-Talabani, A. K. (2022). Mel Frequency Cepstral Coefficient and its Applications: A Review. In IEEE Access (Vol. 10). https://doi.org/10.1109/ACCESS.2022.3223444


**PERFORMANCE METRICS**

The performance metrics used were:
**Accuracy:** I chose accuracy as it offers an overall assessment of my model's performance in predicting emotions across all classes. It's an intuitive metric to understand, but I'm aware that if there's a class imbalance, accuracy might not fully represent the model's effectiveness, especially when one class is more dominant.

**F1 Score:** I find the F1 score crucial due to the imbalance in classes, particularly with 'neutral' being more prevalent. It considers both precision and recall, effectively balancing false positives and false negatives. This metric provides a comprehensive overview of the model's performance, giving equal importance to all emotions without biasing towards a specific one.

**Kappa:** I've included Kappa as it helps me assess the agreement between my predicted and actual emotions while accounting for chance agreement. Considering the imbalanced dataset, Kappa offers insights into whether my model's predictions are better than random chance, providing a more nuanced evaluation beyond what accuracy alone offers.

**MCC (Matthews Correlation Coefficient):** MCC is invaluable in my evaluation process as it takes into account true positives, true negatives, false positives, and false negatives. It quantifies the agreement between predictions and actual classes, considering all error types. Additionally, it serves as a clear indicator of whether my model performs better than random chance, which is crucial for understanding its true effectiveness.

**AUC:** While AUC was used to compare models, it was not selected for final decision making due to the limitations of AUC on multi-class classification. It can be a reductive measure of performance. 


When conducting hyperparameter optimization validation loss, loss, training accuracy validation accuracy and F1 score were used. 

For the Hybrid Bi-Directional LSTM CNN, I conducted hyperparameter optimization on the learning rate, dropout rate, momentum, LSTM hidden size and LSTM number of layers. 

However after full training, the Test accuracy, Test Loss and Test F1 score were calculated to conclude the better model. 



**EXPERIMENTAL RESULTS**

After performing testing on all the models available in the Multiclass classification code, the highest accuracy, F1, Kappa and MCC were 0.4563, 0.4511, 0.3352 and 0.3365 respectively with a Linear Discriminant Analysis Model. These results were quite disappointing. Therefore after conducting a literature review a Hybrid Bi-directional LSTM Activation RNN was shown to be an effective model.

Therefore Bi-directional LSTM was implemented and the accuracy, F1, Kappa and MCC were 0.6038, 0.6, 0.52 and 0.52 respectively

