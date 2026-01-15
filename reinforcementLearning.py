import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import matplotlib.patches as patches

class MazeEnv:
    """Простая среда-лабиринт для демонстрации RL"""
    
    def __init__(self, size=5):
        self.size = size
        self.reset()
        
        # Определяем действия: 0-вверх, 1-вправо, 2-вниз, 3-влево
        self.actions = 4
        
    def reset(self):
        """Сброс среды в начальное состояние"""
        # Агент начинает в левом верхнем углу
        self.agent_pos = [0, 0]
        # Цель в правом нижнем углу
        self.goal_pos = [self.size-1, self.size-1]
        # Препятствия
        self.obstacles = [[1, 1], [2, 3], [3, 2], [4, 0]]
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """Возвращает текущее состояние"""
        return tuple(self.agent_pos)
    
    def step(self, action):
        """Выполняет действие и возвращает (новое_состояние, награда, завершено)"""
        if self.done:
            return self._get_state(), 0, True
        
        # Движение агента
        new_pos = self.agent_pos.copy()
        
        if action == 0:  # Вверх
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Вправо
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        elif action == 2:  # Вниз
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == 3:  # Влево
            new_pos[1] = max(0, new_pos[1] - 1)
        
        # Проверка на столкновение с препятствием
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos
        
        # Вычисление награды
        if self.agent_pos == self.goal_pos:
            reward = 10  # Большая награда за достижение цели
            self.done = True
        elif self.agent_pos in self.obstacles:
            reward = -5  # Штраф за препятствие
        else:
            reward = -0.1  # Небольшой штраф за каждый шаг
        
        return self._get_state(), reward, self.done

# Простой Q-learning агент
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        
        # Инициализация Q-таблицы
        self.q_table = {}
        self.training_history = []  # Для записи истории обучения
        
    def get_q_value(self, state, action):
        """Получение Q-значения для состояния и действия"""
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]
    
    def choose_action(self, state):
        """Выбор действия (ε-жадная стратегия)"""
        if random.random() < self.epsilon:
            # Исследование: случайное действие
            return random.randint(0, self.env.actions - 1)
        else:
            # Использование: лучшее действие
            q_values = [self.get_q_value(state, a) for a in range(self.env.actions)]
            return np.argmax(q_values)
    
    def learn(self, state, action, reward, next_state, done):
        """Обновление Q-таблицы"""
        current_q = self.get_q_value(state, action)
        
        if done:
            max_next_q = 0
        else:
            max_next_q = max([self.get_q_value(next_state, a) 
                            for a in range(self.env.actions)])
        
        # Формула Q-learning
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
        
        # Уменьшаем exploration rate
        if done:
            self.epsilon *= self.epsilon_decay
            
        # Сохраняем статистику
        if done:
            self.training_history.append({
                'epsilon': self.epsilon,
                'q_table_size': len(self.q_table),
                'average_q': np.mean(list(self.q_table.values()))
            })

# Создание среды и агента
env = MazeEnv(size=5)
agent = QLearningAgent(env)

# Списки для хранения истории для анимации
episode_rewards = []
episode_paths = []  # Будем хранить пути для каждого эпизода
all_steps = []  # Все шаги для анимации

# Обучение агента
total_episodes = 100
steps_per_episode = []

for episode in range(total_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    path = [tuple(env.agent_pos.copy())]  # Начальная позиция
    
    while not env.done and steps < 100:  # Максимум 100 шагов на эпизод
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        steps += 1
        path.append(tuple(env.agent_pos.copy()))
        
        # Сохраняем каждый шаг для анимации
        all_steps.append({
            'episode': episode,
            'step': steps,
            'position': env.agent_pos.copy(),
            'reward': total_reward,
            'epsilon': agent.epsilon,
            'done': done
        })
    
    episode_rewards.append(total_reward)
    episode_paths.append(path)
    steps_per_episode.append(steps)
    
    if (episode + 1) % 20 == 0:
        print(f"Эпизод {episode + 1}, Средняя награда: {np.mean(episode_rewards[-20:]):.1f}, Epsilon: {agent.epsilon:.3f}")

# Функция для отрисовки лабиринта
def draw_maze(ax, current_step_info=None, episode_num=0):
    """Отрисовка лабиринта с агентом"""
    ax.clear()
    
    # Рисуем сетку лабиринта
    for i in range(env.size + 1):
        ax.plot([0, env.size], [i, i], 'k-', linewidth=2)
        ax.plot([i, i], [0, env.size], 'k-', linewidth=2)
    
    # Рисуем препятствия
    for obstacle in env.obstacles:
        x, y = obstacle
        rect = patches.Rectangle((y, env.size-1-x), 1, 1, 
                                linewidth=1, edgecolor='black', 
                                facecolor='gray', alpha=0.7)
        ax.add_patch(rect)
        ax.text(y + 0.5, env.size-1-x + 0.5, 'X', 
               ha='center', va='center', fontsize=20, fontweight='bold', color='white')
    
    # Рисуем цель
    goal_x, goal_y = env.goal_pos
    goal_rect = patches.Rectangle((goal_y, env.size-1-goal_x), 1, 1,
                                 linewidth=1, edgecolor='black',
                                 facecolor='gold', alpha=0.9)
    ax.add_patch(goal_rect)
    ax.text(goal_y + 0.5, env.size-1-goal_x + 0.5, 'G', 
           ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Рисуем путь агента в текущем эпизоде
    if current_step_info and episode_num < len(episode_paths):
        current_path = episode_paths[episode_num]
        # Отображаем только пройденную часть пути
        max_step = current_step_info['step'] if current_step_info else 0
        path_to_show = current_path[:min(max_step, len(current_path))]
        
        if len(path_to_show) > 1:
            # Преобразуем координаты для отрисовки
            path_y = [env.size-1-p[0] + 0.5 for p in path_to_show]
            path_x = [p[1] + 0.5 for p in path_to_show]
            ax.plot(path_x, path_y, 'b-', alpha=0.3, linewidth=2)
            
            # Отмечаем точки пути
            ax.scatter(path_x, path_y, c='blue', alpha=0.2, s=20)
    
    # Рисуем агента
    if current_step_info:
        agent_pos = current_step_info['position']
        agent_x, agent_y = agent_pos
        # Рисуем агента как закрашенный кружок
        agent_circle = patches.Circle((agent_y + 0.5, env.size-1-agent_x + 0.5), 
                                     0.3, facecolor='red', edgecolor='black', linewidth=1)
        ax.add_patch(agent_circle)
        ax.text(agent_y + 0.5, env.size-1-agent_x + 0.5, 'A', 
               ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Настройки графика
    ax.set_xlim(-0.1, env.size + 0.1)
    ax.set_ylim(-0.1, env.size + 0.1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Заголовок с информацией
    if current_step_info:
        title = f"Эпизод: {current_step_info['episode']+1}, Шаг: {current_step_info['step']}"
        title += f"\nНаграда: {current_step_info['reward']:.1f}, Epsilon: {current_step_info['epsilon']:.3f}"
        if current_step_info['done']:
            title += " [ЦЕЛЬ ДОСТИГНУТА]"
        ax.set_title(title, fontsize=10, fontweight='bold')
    else:
        ax.set_title(f"Эпизод: {episode_num+1}", fontsize=10, fontweight='bold')

# Создаем анимацию
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Выберем интересные эпизоды для анимации
interesting_episodes = [0, total_episodes//3, total_episodes-1]
animation_steps = []

# Собираем шаги для выбранных эпизодов
for ep_num in interesting_episodes:
    # Находим все шаги этого эпизода
    ep_steps = [step for step in all_steps if step['episode'] == ep_num]
    animation_steps.extend(ep_steps)
    
    # Добавляем паузу между эпизодами
    for _ in range(5):  # 5 кадров паузы
        animation_steps.append(ep_steps[-1] if ep_steps else None)

# Функция для обновления кадра анимации
def update_frame(frame_idx):
    """Обновление кадра анимации"""
    if frame_idx >= len(animation_steps) or animation_steps[frame_idx] is None:
        return
    
    step_info = animation_steps[frame_idx]
    ep_num = step_info['episode']
    
    # Определяем, на каком графике отображать (первый, второй или третий)
    ep_idx = interesting_episodes.index(ep_num)
    
    # Очищаем все оси
    for ax in axes:
        ax.clear()
    
    # Отрисовываем все три эпизода
    for i, ep_to_show in enumerate(interesting_episodes):
        # Находим последний шаг для этого эпизода на текущем кадре
        current_ep_steps = [s for s in animation_steps[:frame_idx+1] 
                          if s and s['episode'] == ep_to_show]
        
        if current_ep_steps:
            last_step = current_ep_steps[-1]
            draw_maze(axes[i], last_step, ep_to_show)
        else:
            # Если шагов еще нет, показываем начальное состояние
            draw_maze(axes[i], None, ep_to_show)
    
    # Добавляем общую информацию
    fig.suptitle(f'Обучение с подкреплением: Q-learning в лабиринте\n'
                f'Кадр: {frame_idx+1}/{len(animation_steps)}', 
                fontsize=14, fontweight='bold')

# Создаем анимацию
print("\nСоздание анимации...")
animation = FuncAnimation(fig, update_frame, 
                         frames=len(animation_steps),
                         interval=200,  # 200ms между кадрами
                         repeat=False)

# Сохраняем анимацию в файл
try:
    animation.save('rl_maze_learning.gif', writer='pillow', fps=10)
    print("Анимация сохранена как 'rl_maze_learning.gif'")
except Exception as e:
    print(f"Не удалось сохранить анимацию: {e}")
    print("Попробуйте установить pillow: pip install pillow")

# Отображаем анимацию
plt.tight_layout()
plt.show()

# Дополнительная визуализация: график обучения
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# График наград по эпизодам
axes2[0, 0].plot(episode_rewards, 'b-', alpha=0.6, label='Награда за эпизод')
axes2[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes2[0, 0].set_xlabel('Эпизод')
axes2[0, 0].set_ylabel('Суммарная награда')
axes2[0, 0].set_title('Награды по эпизодам')
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)

# График количества шагов
axes2[0, 1].plot(steps_per_episode, 'g-', alpha=0.6)
axes2[0, 1].set_xlabel('Эпизод')
axes2[0, 1].set_ylabel('Шагов до цели')
axes2[0, 1].set_title('Эффективность агента')
axes2[0, 1].grid(True, alpha=0.3)

# График epsilon (исследование)
epsilon_history = [h['epsilon'] for h in agent.training_history]
axes2[1, 0].plot(epsilon_history, 'r-', alpha=0.6)
axes2[1, 0].set_xlabel('Эпизод')
axes2[1, 0].set_ylabel('Epsilon (исследование)')
axes2[1, 0].set_title('Стратегия исследования (ε)')
axes2[1, 0].grid(True, alpha=0.3)

# График средней Q-значения
avg_q_history = [h['average_q'] for h in agent.training_history]
axes2[1, 1].plot(avg_q_history, 'purple', alpha=0.6)
axes2[1, 1].set_xlabel('Эпизод')
axes2[1, 1].set_ylabel('Среднее Q-значение')
axes2[1, 1].set_title('Качество Q-таблицы')
axes2[1, 1].grid(True, alpha=0.3)

plt.suptitle('Статистика обучения агента', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Визуализация финальной Q-таблицы (для небольшого лабиринта)
print("\nФинальная Q-таблица (первые 10 записей):")
for i, ((state, action), q_value) in enumerate(list(agent.q_table.items())[:10]):
    action_names = ['Вверх', 'Вправо', 'Вниз', 'Влево']
    print(f"Состояние {state}, действие {action_names[action]}: Q = {q_value:.3f}")

# Демонстрация итоговой политики для нескольких стартовых позиций
print("\nДемонстрация финальной политики:")
test_positions = [(0, 0), (0, 2), (2, 0), (3, 4)]

fig3, axes3 = plt.subplots(2, 2, figsize=(10, 10))
axes3 = axes3.flatten()

for idx, start_pos in enumerate(test_positions):
    # Временно меняем стартовую позицию
    env.agent_pos = list(start_pos)
    env.done = False
    
    path = [start_pos]
    total_reward = 0
    steps = 0
    
    while not env.done and steps < 20:
        state = env._get_state()
        # Используем обученную политику (без исследования)
        action = np.argmax([agent.get_q_value(state, a) for a in range(env.actions)])
        next_state, reward, done = env.step(action)
        path.append(next_state)
        total_reward += reward
        steps += 1
    
    # Визуализация
    ax = axes3[idx]
    
    # Рисуем лабиринт
    for i in range(env.size + 1):
        ax.plot([0, env.size], [i, i], 'k-', linewidth=2)
        ax.plot([i, i], [0, env.size], 'k-', linewidth=2)
    
    # Препятствия
    for obstacle in env.obstacles:
        x, y = obstacle
        rect = patches.Rectangle((y, env.size-1-x), 1, 1, 
                                linewidth=1, edgecolor='black', 
                                facecolor='gray', alpha=0.7)
        ax.add_patch(rect)
        ax.text(y + 0.5, env.size-1-x + 0.5, 'X', 
               ha='center', va='center', fontsize=15, fontweight='bold', color='white')
    
    # Цель
    goal_x, goal_y = env.goal_pos
    goal_rect = patches.Rectangle((goal_y, env.size-1-goal_x), 1, 1,
                                 linewidth=1, edgecolor='black',
                                 facecolor='gold', alpha=0.9)
    ax.add_patch(goal_rect)
    ax.text(goal_y + 0.5, env.size-1-goal_x + 0.5, 'G', 
           ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Путь
    path_y = [env.size-1-p[0] + 0.5 for p in path]
    path_x = [p[1] + 0.5 for p in path]
    ax.plot(path_x, path_y, 'b-', alpha=0.7, linewidth=2)
    ax.scatter(path_x, path_y, c='blue', alpha=0.5, s=30)
    
    # Стартовая позиция (зеленый кружок)
    start_circle = patches.Circle((start_pos[1] + 0.5, env.size-1-start_pos[0] + 0.5), 
                                  0.2, facecolor='green', edgecolor='black', linewidth=1)
    ax.add_patch(start_circle)
    ax.text(start_pos[1] + 0.5, env.size-1-start_pos[0] + 0.5, 'S', 
           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Конечная позиция (красный кружок)
    if len(path) > 1:
        end_pos = path[-1]
        end_circle = patches.Circle((end_pos[1] + 0.5, env.size-1-end_pos[0] + 0.5), 
                                    0.2, facecolor='red', edgecolor='black', linewidth=1)
        ax.add_patch(end_circle)
        ax.text(end_pos[1] + 0.5, env.size-1-end_pos[0] + 0.5, 'E', 
               ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    ax.set_xlim(-0.1, env.size + 0.1)
    ax.set_ylim(-0.1, env.size + 0.1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Старт: {start_pos}\nШаги: {steps}, Награда: {total_reward:.1f}', fontsize=10)

plt.suptitle('Финальная политика агента из разных стартовых позиций', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Сводная таблица результатов
print("\n" + "="*60)
print("ИТОГИ ОБУЧЕНИЯ С ПОДКРЕПЛЕНИЕМ")
print("="*60)
print(f"Всего эпизодов обучения: {total_episodes}")
print(f"Средняя награда (первые 20 эпизодов): {np.mean(episode_rewards[:20]):.1f}")
print(f"Средняя награда (последние 20 эпизодов): {np.mean(episode_rewards[-20:]):.1f}")
print(f"Финальный уровень исследования (epsilon): {agent.epsilon:.4f}")
print(f"Размер Q-таблицы: {len(agent.q_table)} записей")
print(f"Среднее количество шагов до цели: {np.mean(steps_per_episode[-20:]):.1f}")
print(f"Максимальная достигнутая награда: {max(episode_rewards):.1f}")
print("="*60)

# Легенда обозначений
print("\nОБОЗНАЧЕНИЯ В ЛАБИРИНТЕ:")
print("  A  - Агент (Agent)")
print("  G  - Цель (Goal)")
print("  X  - Препятствие (преграда)")
print("  S  - Стартовая позиция")
print("  E  - Конечная позиция")
print("  Синяя линия - Путь агента")
print("  Зеленый кружок - Начало пути")
print("  Красный кружок - Конец пути")