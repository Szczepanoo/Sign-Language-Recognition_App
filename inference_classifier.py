import pickle
import cv2
import mediapipe as mp
import numpy as np

# Wczytanie modelu
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

metadata = pickle.load(open('metadata.p', 'rb'))  # Wczytaj metadane
scaler = metadata['scaler']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Słownik etykiet gestów
labels_dict = {
    0: "A", 1: "B", 12: "C", 19: "D", 20: "E", 21: "F", 22: "G", 23: "H", 24: "I",
    25: "J", 2: "K", 3: "L", 4: "M", 5: "N", 6: "O", 7: "P",
    8: "Q", 9: "R", 10: "S", 11: "T", 13: "U", 14: "V", 15: "W", 16: "X", 17: "Y", 18: "Z"
}


"""
Kolejność etykiet: ['0' '1' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '2' '20' '21'
 '22' '23' '24' '25' '3' '4' '5' '6' '7' '8' '9']

metadata = pickle.load(open('metadata.p', 'rb'))
label_encoder = metadata['label_encoder']
print("Kolejność etykiet:", label_encoder.classes_)

"""


while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Dopasowanie do wymaganej długości 84
        data_aux = data_aux + [0] * (42 - len(data_aux)) if len(data_aux) < 42 else data_aux[:42]

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # **Rozwiązanie błędu** → Dodanie wymiaru batcha
        data_aux = np.array(data_aux, dtype=np.float32).reshape(1, -1)  # Teraz shape = (1, 84)

        data_aux = scaler.transform(data_aux)

        # **Naprawiony błąd predict_proba()**
        probabilities = model.predict(data_aux)  # predict() zwraca prawdopodobieństwa dla każdej klasy

        predicted_character = labels_dict[
            int(np.argmax(probabilities))]  # Zwrócenie klasy o najwyższym prawdopodobieństwie
        probability_of_prediction = np.max(probabilities)  # Największe prawdopodobieństwo

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        if probability_of_prediction > 0.9:
            cv2.putText(
                frame, f"{predicted_character} - {probability_of_prediction:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA
            )
        # print("Probabilities", probabilities)
        # print("Input shape: ", data_aux.shape)

    cv2.imshow('Sign language recognition app', frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
