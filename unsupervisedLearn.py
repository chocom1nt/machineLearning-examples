import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Генерация данных без меток
np.random.seed(42)
n_samples = 300

# Создаем три группы фигур разной формы
X, true_labels = make_blobs(n_samples=n_samples, 
                           centers=3, 
                           cluster_std=1.0,
                           random_state=42)

# Добавляем шум для реалистичности
X += np.random.normal(0, 0.3, X.shape)

# Визуализация до кластеризации
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Исходные данные (без меток)
colors_original = ['red', 'yellow', 'blue']
shapes = ['^', 'o', 's']  # треугольник, круг, квадрат

for i in range(3):
    points = X[true_labels == i]
    ax1.scatter(points[:, 0], points[:, 1], 
               c=colors_original[i], 
               marker=shapes[i],
               alpha=0.6,
               s=50,
               label=f'Группа {i+1}')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Исходные данные (без знания о классах)')
ax1.legend()
ax1.grid(True, alpha=0.3)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
predicted_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Визуализация результатов кластеризации
for i in range(3):
    points = X[predicted_labels == i]
    ax2.scatter(points[:, 0], points[:, 1], 
               c=colors_original[i], 
               marker=shapes[i],
               alpha=0.6,
               s=50,
               label=f'Кластер {i+1}')

# Отображаем центроиды
ax2.scatter(centroids[:, 0], centroids[:, 1],
           c='black', marker='X', s=200, 
           label='Центры кластеров', alpha=0.8)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Результаты кластеризации K-means')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Оцениваем качество кластеризации
silhouette_avg = silhouette_score(X, predicted_labels)
print(f"Силуэтный коэффициент: {silhouette_avg:.3f}")

print("\nОбучение без учителя:")
print("- Нет размеченных данных (только признаки)")
print("- Модель ищет скрытые структуры/паттерны")
print("- Цель: найти естественные группы в данных")
print("- Пример: группировка фигур по похожим характеристикам")