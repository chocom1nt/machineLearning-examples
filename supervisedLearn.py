import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


np.random.seed(42)


n_samples = 200

class0_x = np.random.normal(2, 1, n_samples)
class0_y = np.random.normal(2, 1, n_samples)
class0_labels = np.zeros(n_samples)

class1_x = np.random.normal(6, 1, n_samples)
class1_y = np.random.normal(6, 1, n_samples)
class1_labels = np.ones(n_samples)

X = np.column_stack([np.concatenate([class0_x, class1_x]), 
                     np.concatenate([class0_y, class1_y])])
y = np.concatenate([class0_labels, class1_labels])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Создаем и обучаем модель
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2%}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# До обучения
ax1.scatter(class0_x, class0_y, c='red', marker='^', alpha=0.6, 
           label='Красные треугольники (0)')
ax1.scatter(class1_x, class1_y, c='yellow', marker='o', alpha=0.6,
           label='Желтые круги (1)', edgecolors='black')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Исходные данные (разные классы фигур)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# После обучения
xx, yy = np.meshgrid(np.arange(0, 8, 0.1), 
                     np.arange(0, 8, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax2.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
ax2.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], 
           c='red', marker='^', label='Истина: 0')
ax2.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
           c='yellow', marker='o', label='Истина: 1', edgecolors='black')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title(f'Граница решения модели (Точность: {accuracy:.2%})')
ax2.legend()
ax2.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()

print("\nОбучение с учителем:")
print("- Есть размеченные данные (метки: 0 и 1)")
print("- Модель учится сопоставлять признаки (координаты) с метками")
print("- Цель: предсказать класс новых фигур")
