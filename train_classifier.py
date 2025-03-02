import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Wczytanie danych
data_dict = pickle.load(open('data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Normalizacja danych wejściowych
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Zakodowanie etykiet numerycznie
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Zamiana etykiet na one-hot encoding
num_classes = len(np.unique(labels_encoded))
labels_onehot = tf.keras.utils.to_categorical(labels_encoded, num_classes=num_classes)

# Podział danych
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_onehot, test_size=0.2, stratify=labels_encoded, random_state=42
)

# Bardziej złożony model
model = Sequential([
    Dense(1024, input_shape=(data.shape[1],), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer=AdamW(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callback - zatrzymuje trening, jeśli dokładność przestaje rosnąć
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Trenowanie modelu
epochs = 500  # Można trenować dłużej, ale EarlyStopping zapobiegnie przeuczeniu
history = model.fit(x_train, y_train, epochs=epochs, batch_size=32,
                    validation_data=(x_test, y_test), callbacks=[early_stopping])

# Ewaluacja na zbiorze testowym
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {score[1]*100:.2f}%")

pickle.dump({'model': model}, open('model.p', 'wb'))

pickle.dump({'scaler': scaler, 'label_encoder': le}, open('metadata.p', 'wb'))
