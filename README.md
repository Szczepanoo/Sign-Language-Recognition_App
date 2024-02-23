
## Real-time Sign Language Recognition

This Python code implements a sign language recognition application using the `MediaPipe` library for hand tracking and the `scikit-learn` library for machine learning. The application captures hand gestures through a webcam, processes the hand landmarks using MediaPipe's Hand module, and feeds the extracted features into a pre-trained `Random Forest Classifier` for sign language prediction. The training data is collected by capturing images of hand gestures using the `OpenCV` library and saving them in labeled directories. The collected data is then preprocessed, padded to a fixed sequence length, and used to train the Random Forest model. The real-time sign language recognition results are displayed on the screen, showing the predicted sign and its confidence level.


<img width="478" alt="Zrzut ekranu 2024-02-23 212051" src="https://github.com/Szczepanoo/Sign-Language-Recognition_App/assets/125917209/2a411e63-77c0-48f1-b719-38eb27560bdb">
<img width="479" alt="Zrzut ekranu 2024-02-23 211912" src="https://github.com/Szczepanoo/Sign-Language-Recognition_App/assets/125917209/398ee28e-a55c-4c3d-84f1-e5a8de820922">
<img width="477" alt="Zrzut ekranu 2024-02-23 211827" src="https://github.com/Szczepanoo/Sign-Language-Recognition_App/assets/125917209/b049672b-57c7-4ceb-9bba-d24326dfbf56">


Improved version of: [LINK](https://www.youtube.com/watch?v=MJCSjXepaAM)
