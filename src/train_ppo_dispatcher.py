# train_ppo_dispatcher.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from training import SimpleDispatcher
import random
from datetime import datetime, timedelta
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_episode(model, resources, orders, device='cpu'):
    """
    Генерирует один эпизод: назначает все операции с помощью текущей политики (модели),
    вычисляет суммарное опоздание и логи вероятностей действий.
    Возвращает loss (для обратного распространения), среднюю награду (отрицательное опоздание),
    и последнюю награду.
    """
    model.eval()
    work_until = {r.id: datetime.now() for r in resources}
    log_probs = []
    rewards = []

    all_ops = []
    for order in orders:
        for op in order.ops:
            all_ops.append((order, op))
    all_ops.sort(key=lambda x: x[0].due_date)

    for order, op in all_ops:
        # Простейший вектор признаков (рандомный, т.к. PPO‑пример условный)
        # В реальной задаче нужно формировать признаки как в plan_with_reliability
        state = torch.randn(1, 36).to(device)   # 36 = 6 (операция) + 5*6 (ресурсы)
        logits = model(state)
        probs = torch.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[0, action] + 1e-8)
        log_probs.append(log_prob)

        chosen_res = resources[action % len(resources)]
        start = work_until[chosen_res.id]
        dur = timedelta(minutes=op.norm_duration)
        end = start + dur
        work_until[chosen_res.id] = end

        tardiness = max(0.0, (end - order.due_date).total_seconds() / 3600.0)  # в часах
        rewards.append(-tardiness)

    # Простейший policy gradient (без advantage)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, device=device)

    # Loss = - log_prob * discounted_reward
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    loss = torch.stack(policy_loss).sum()

    return loss, rewards[-1] if rewards else 0, np.mean(rewards) if rewards else 0


def train_ppo(episodes=50, lr=0.001, n_orders=30, n_resources=5,
              progress_callback=None, log_callback=None, stop_event=None):
    """
    Обучает модель SimpleDispatcher с помощью упрощённого PPO.
    Возвращает обученную модель.
    """
    device = torch.device("cpu")
    input_dim = 36          # 6 (операция) + 5 * 6 (ресурсы)
    output_dim = n_resources
    model = SimpleDispatcher(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Фиксируем среду для воспроизводимости результатов
    from datamodels import Resource, Order, Operation
    from database import get_operation_types

    random.seed(42)
    np.random.seed(42)

    types = list(get_operation_types().keys()) if get_operation_types() else ['фрезеровка', 'сборка', 'сварка']
    resources = [Resource(f'R{i+1}', f'Станок {i+1}') for i in range(5)]
    for r in resources:
        r.reliability = 0.5 + 0.5 * random.random()
        r.repair = 0
        r.status = 'Работает'

    orders = []
    base = datetime(2026, 4, 27, 8, 0, 0)
    for i in range(n_orders):
        due = base + timedelta(hours=random.randint(10, 100))
        order = Order(f'ORD_{i:03d}', due, random.uniform(0.5, 2.0))
        op = Operation(
            f'{order.id}_op1', order.id,
            f'Деталь_{random.randint(1,5)}', 1,
            random.uniform(20, 120)
        )
        op.type_id = random.choice(types) if types else None
        op.slack_hours = random.uniform(0, 48)
        order.ops.append(op)
        orders.append(order)

    rewards_history = []

    for ep in range(1, episodes + 1):
        if stop_event and stop_event.is_set():
            break

        loss, last_reward, avg_reward = generate_episode(model, resources, orders, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_history.append(avg_reward)

        if progress_callback:
            progress_callback(ep, avg_reward)
        if log_callback and (ep % 10 == 0 or ep == 1):
            log_callback(f"Эпизод {ep:3d} | Средняя награда: {avg_reward:.4f}")

    torch.save(model.state_dict(), "ppo_dispatcher.pth")
    save_ppo_metrics(rewards_history, episodes, "PPO")
    return model


def save_ppo_metrics(rewards, total_episodes, model_name):
    """Сохраняет графики и JSON с метриками обучения PPO."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"metrics/{model_name}/{timestamp}"
    os.makedirs(folder, exist_ok=True)

    # JSON-история
    history = {
        "episode": list(range(1, total_episodes + 1)),
        "avg_tardiness": [float(r) for r in rewards]
    }
    with open(f"{folder}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # График средней награды (опоздания)
    plt.figure()
    plt.plot(range(1, total_episodes + 1), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Avg Tardiness (hours)')
    plt.title(f'{model_name} Training Progress')
    plt.savefig(f"{folder}/tardiness.png")
    plt.close()

    # Скользящее среднее (если эпизодов >= 10)
    if len(rewards) >= 10:
        ma = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        plt.figure()
        plt.plot(range(10, total_episodes + 1), ma)
        plt.xlabel('Episode')
        plt.ylabel('Avg Tardiness (10-ep MA)')
        plt.title(f'{model_name} Moving Average')
        plt.savefig(f"{folder}/tardiness_ma.png")
        plt.close()

    print(f"Метрики PPO сохранены в {folder}")