
## Real-time Sign Language Recognition

This Python code implements a sign language recognition application using the `MediaPipe` library for hand tracking and the `scikit-learn` library for machine learning. The application captures hand gestures through a webcam, processes the hand landmarks using MediaPipe's Hand module, and feeds the extracted features into a pre-trained `Random Forest Classifier` for sign language prediction. The training data is collected by capturing images of hand gestures using the `OpenCV` library and saving them in labeled directories. The collected data is then preprocessed, padded to a fixed sequence length, and used to train the Random Forest model. The real-time sign language recognition results are displayed on the screen, showing the predicted sign and its confidence level.


Improved version of: [LINK](https://www.youtube.com/watch?v=MJCSjXepaAM)
