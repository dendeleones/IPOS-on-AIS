# train_ppo_dispatcher.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from training import SimpleDispatcher
import random
from datetime import datetime, timedelta
import os

def generate_episode(model, resources, orders, device='cpu'):
    """
    Генерирует один эпизод: назначает все операции с помощью политики (модель),
    вычисляет суммарную награду (например, отрицательное среднее опоздание).
    """
    model.eval()
    work_until = {r.id: datetime.now() for r in resources}
    total_tardiness = 0.0
    log_probs = []
    rewards = []

    # Собираем все операции и сортируем по EDD
    all_ops = []
    for order in orders:
        for op in order.ops:
            all_ops.append((order, op))
    all_ops.sort(key=lambda x: x[0].due_date)

    for order, op in all_ops:
        # Строим признаки (такие же, как в plan_with_reliability)
        type_map = {'фрезеровка': 0, 'сборка': 1, 'сварка': 2}
        type_onehot = np.zeros(3)
        if op.type_id:
            tname = op.type_id if isinstance(op.type_id, str) else ''
            if tname in type_map:
                type_onehot[type_map[tname]] = 1.0
        op_feat = np.array([
            *type_onehot,
            op.norm_duration / 120.0,
            order.priority_weight,
            op.slack_hours / 48.0 if hasattr(op, 'slack_hours') else 0.0
        ])
        res_feat_list = []
        for r in resources:
            cur_load = (work_until[r.id] - datetime.now()).total_seconds() / 60.0 / 240.0
            res_vec = np.array([
                1.0 if r.status == 'Работает' else 0.0,
                cur_load,
                1.0,  # совместимость упрощённо
                r.load_minutes / 240.0,
                float(r.repair),
                r.reliability
            ])
            res_feat_list.append(res_vec)
        full = np.concatenate([op_feat] + res_feat_list)

        # Действие
        state_tensor = torch.tensor(full, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(state_tensor)
        probs = torch.softmax(logits, dim=1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[0, action] + 1e-8)
        log_probs.append(log_prob)

        # Выполняем действие
        chosen_res = resources[action]
        start = max(work_until[chosen_res.id], datetime.now())
        dur = timedelta(minutes=op.norm_duration)
        end = start + dur
        work_until[chosen_res.id] = end

        # Награда: отрицательное опоздание
        tardiness = max(0.0, (end - order.due_date).total_seconds() / 3600.0)  # в часах
        total_tardiness += tardiness
        rewards.append(-tardiness)

    # PPO-подобное обновление (очень упрощённое)
    if log_probs:
        loss = -torch.stack(log_probs).sum()  # максимизируем сумму логарифмов
    else:
        loss = torch.tensor(0.0)
    return loss, total_tardiness, rewards

def train_ppo(episodes=50, lr=0.001, n_orders=30, n_resources=3,
              progress_callback=None, stop_event=None):
    device = torch.device("cpu")
    input_dim = 6 + 5 * 6  # фиксировано 5 ресурсов
    output_dim = 5
    model = SimpleDispatcher(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Создаём синтетические заказы и ресурсы для каждой эпохи (или используем переданные)
    from datamodels import Resource, Order, Operation
    from database import get_operation_types

    types = list(get_operation_types().keys()) if get_operation_types() else ['фрезеровка', 'сборка', 'сварка']

    for ep in range(1, episodes + 1):
        if stop_event and stop_event.is_set():
            break
        # Генерируем случайные ресурсы и заказы
        resources = [Resource(f'R{i+1}', f'Станок {i+1}') for i in range(5)]
        for r in resources:
            r.reliability = 0.5 + 0.5 * random.random()
            r.repair = 1 if random.random() < 0.1 else 0
            r.status = 'Работает' if r.reliability > 0.3 else 'Авария'
        orders = []
        base = datetime(2026, 4, 27, 8, 0, 0)
        for _ in range(n_orders):
            due = base + timedelta(hours=random.randint(10, 100))
            order = Order(f'ORD_{len(orders):03d}', due, random.uniform(0.5, 2.0))
            op = Operation(f'{order.id}_op1', order.id, f'Деталь_{random.randint(1,5)}', 1,
                           random.uniform(20, 120))
            op.type_id = random.choice(types) if types else None
            op.slack_hours = random.uniform(0, 48)
            order.ops.append(op)
            orders.append(order)

        loss, total_tard, _ = generate_episode(model, resources, orders, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if progress_callback:
            progress_callback(ep, total_tard / n_orders if n_orders else 0)

    # Сохраняем модель
    torch.save(model.state_dict(), "ppo_dispatcher.pth")
    return model