import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 3. Обучение модели
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=15,
    validation_split=0.2,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nТестовая точность: {test_acc:.4f}")
print(f"Тестовая ошибка: {test_loss:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# График точности
axes[0].plot(history.history['accuracy'], label='Точность на обучении')
axes[0].plot(history.history['val_accuracy'], label='Точность на валидации')
axes[0].set_xlabel('Эпохи')
axes[0].set_ylabel('Точность')
axes[0].set_title('Точность модели')
axes[0].legend()
axes[0].grid(True)

# График потерь
axes[1].plot(history.history['loss'], label='Потери на обучении')
axes[1].plot(history.history['val_loss'], label='Потери на валидации')
axes[1].set_xlabel('Эпохи')
axes[1].set_ylabel('Потери')
axes[1].set_title('Потери модели')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


num_samples = 10

random_indices = np.random.choice(len(x_test), num_samples, replace=False)

sample_images = x_test[random_indices]
sample_labels = y_test[random_indices]

predictions = model.predict(sample_images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(sample_labels, axis=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for i in range(num_samples):
    img = sample_images[i].reshape(28, 28)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Прогноз: {predicted_labels[i]}\nИстина: {true_labels[i]}')
    axes[i].axis('off')

plt.suptitle('Случайные примеры предсказаний модели', fontsize=14)
plt.tight_layout()
plt.show()