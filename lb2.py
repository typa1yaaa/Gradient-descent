import random
import numpy as np
import matplotlib.pyplot as plt

# функция активации - сигмоида
def act_sigmoid(x):
    return 1 / (1 + np.exp(-x))

# функция активации - линейная
def act_linear(x):
    return x

# Производная сигмоиды для использования в градиентном спуске
def sigmoid_derivative(x):
    return act_sigmoid(x) * (1 - act_sigmoid(x))

# прямой проход
def forward_pass(x, y, weigth):
    f1 = act_linear(weigth[0] * x)
    f2 = act_linear(weigth[1] * y)
    f_out = act_sigmoid(weigth[2]*f1 + weigth[3]*f2)
    return f_out, f1, f2

# генерим датасет
def generate_dataset(num_points):
    dataset = []

    for _ in range(num_points):
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        if y > -x : key = 1 # ключ, что если 
        else:
            key = 0
        dataset.append(((x, y), key))

    return dataset

# инициализируем веса для 3-х нейронов
weights = [random.uniform(-5, 5) for _ in range(4)]

# инициализируем выборки: тестовую и валидирующая
training_dataset = generate_dataset(10000)
validation_dataset = generate_dataset(100)

lambda_coeff = 0.001 

# обучение сети
for epoch in range(100):  # количество эпох
    for (x, y), key in training_dataset:
        # вычисляем выходное значение условного нейрона
        f_out, f1, f2 = forward_pass(x, y, weights)

        # функция потери e = (f_out - key)**2
        # производная функции потери
        d_error = 2*(f_out - key)

        # производная для каждого веса
        d_sigmoid = sigmoid_derivative(f_out)  # производная сигмоиды по выходу
        
        # обновление весов с помощью градиентного спуска       
        weights[0] -= lambda_coeff * d_error * d_sigmoid * weights[2] * x
        weights[1] -= lambda_coeff * d_error * d_sigmoid * weights[3] * y
        weights[2] -= lambda_coeff * d_error * d_sigmoid * f1 # weights[0] * x  = f1, поэтому заменяем
        weights[3] -= lambda_coeff * d_error * d_sigmoid * f2 # weights[1] * y  = f2, поэтому заменяем

# проверка на валидирующей выборке
correct_predictions = 0
validation_predicted = []
for (x, y), key in validation_dataset:
    f_out, _, _ = forward_pass(x, y, weights)
    prediction = 1 if f_out == 1 else 0
    validation_predicted.append(prediction)
    if prediction == key: correct_predictions += 1

print(f"Доля правильно определенных точек: {correct_predictions / len(validation_dataset)}")
print(f"Финальные веса: weights[0] = {weights[0]:.3f}, weights[1] = {weights[1]:.3f}, weights[2] = {weights[2]:.3f}, weights[3] = {weights[3]:.3f}")

def graphics(dataset, title, predictions):
    plt.figure(figsize=(8, 8))
    
    correctly_above = []
    correctly_below = []
    errors = []
    
    for i, ((x, y), true_key) in enumerate(dataset):
        if true_key == predictions[i]:
            if true_key == 1:
                color = 'green'  # зеленый: правильно выше линии
                correctly_above.append((x, y))
            else:
                color = 'blue'  # синий: правильно ниже линии
                correctly_below.append((x, y))
        else:
            color = 'red'  # красный: ошибки
            errors.append((x, y))

        plt.scatter(x, y, color=color, alpha=0.6)

    plt.axline((0, 0), slope=-1, color='black', linestyle='--', label='y = -x')
    
    plt.scatter([], [], color='green', label='Правильно выше линии')
    plt.scatter([], [], color='blue', label='Правильно ниже линии')
    plt.scatter([], [], color='red', label='Ошибки')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

graphics(validation_dataset, "Валидационная выборка с предсказаниями", validation_predicted)