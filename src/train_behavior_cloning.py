# train_behavior_cloning.py
import numpy as np
import random
from datetime import datetime, timedelta
from datamodels import Order, Operation, Resource
from job_shop_env import JobShopEnv
from sklearn.neural_network import MLPClassifier
import pickle

def generate_random_orders(n_orders=50, n_resources=10, seed=None):
    if seed is not None:
        random.seed(seed)
    resources = [Resource(id=f'R{i}', name=f'Станок {i}') for i in range(1, n_resources+1)]
    items = [f'Item_{j}' for j in range(1, 5)]
    orders = []
    base_time = datetime(2026, 4, 27, 8, 0, 0)
    for i in range(1, n_orders+1):
        due = base_time + timedelta(hours=random.randint(10, 200))
        order = Order(id=f'ORD_{i:03d}', due_date=due, priority_weight=random.uniform(0.5, 2.0))
        n_ops = random.randint(2, 4)
        prev_id = None
        for op_j in range(1, n_ops+1):
            op_id = f'{order.id}_op{op_j}'
            res = random.choice(resources)
            duration = random.uniform(20, 180)
            preds = [prev_id] if prev_id else []
            order.ops.append(Operation(
                id=op_id, order_id=order.id, item=random.choice(items),
                op_number=op_j, resource_id=res.id,
                norm_duration=duration, predecessors=preds
            ))
            prev_id = op_id
        orders.append(order)
    return orders, resources

def expert_cr(env):
    ready = env.get_ready_ops_list()
    if not ready:
        return None
    best_op = None
    best_cr = float('inf')
    for op_info in ready:
        op_id = op_info['op_id']
        # находим заказ и операцию
        for order in env.orders:
            for op in order.ops:
                if op.id == op_id:
                    due_minutes = env._order_due_minutes(order)
                    remaining_time = env.op_remaining[op_id]
                    slack = due_minutes - env.current_time
                    if slack <= 0:
                        cr = -float('inf')
                    else:
                        cr = slack / remaining_time if remaining_time > 0 else -float('inf')
                    if cr < best_cr:
                        best_cr = cr
                        best_op = op_id
                    break
    return best_op

def generate_training_data(num_samples=2000):
    X, y = [], []
    ACTION_DIM = 150
    for _ in range(num_samples):
        orders, resources = generate_random_orders(30, 3, seed=None)
        env = JobShopEnv(orders, resources)
        state = env.reset()
        done = False
        while not done:
            ready = env.get_ready_ops_list()
            if not ready:
                break
            expert_op = expert_cr(env)
            if expert_op is None:
                break
            # индекс эксперта в глобальном списке
            global_ready = env.get_ready_ops_list()
            try:
                expert_idx = next(i for i, op in enumerate(global_ready) if op['op_id'] == expert_op)
            except StopIteration:
                break
            mask = [0]*ACTION_DIM
            if expert_idx < ACTION_DIM:
                mask[expert_idx] = 1
            X.append(state)
            y.append(mask)
            state, _, done, _ = env.step(expert_op)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Генерация данных...")
    X, y = generate_training_data(2000)
    print(f"Собрано {len(X)} примеров.")
    model = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=100, verbose=True)
    model.fit(X, y)
    with open("behavior_cloning.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Модель behaviour cloning сохранена как behavior_cloning.pkl")