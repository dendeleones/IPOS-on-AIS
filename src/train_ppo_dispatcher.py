# train_ppo_dispatcher.py
import numpy as np
import random
from datetime import datetime, timedelta
from datamodels import Order, Operation, Resource
from job_shop_env import JobShopEnv
from ppo_agent import PPO
import torch

def generate_random_orders(n_orders=30, n_resources=3, seed=None):
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

def train_ppo(episodes=200, lr=3e-4, n_orders=30, n_resources=3,
              progress_callback=None, stop_event=None):
    # Создаём временную среду, чтобы получить точные размерности
    orders, resources = generate_random_orders(n_orders, n_resources)
    env = JobShopEnv(orders, resources)
    base_state = env.reset()
    busy = np.array([1.0 if env.resource_remaining[rid] > 0 else 0.0 for rid in env.resource_ids])
    ext_state = np.concatenate([base_state, busy])
    state_dim = len(ext_state)
    op_feat_dim = 5
    action_dim = env.max_actions

    agent = PPO(state_dim, op_feat_dim, action_dim, lr=lr, device='cpu', epochs=5)

    for ep in range(1, episodes+1):
        if stop_event and stop_event.is_set():
            break

        orders, resources = generate_random_orders(n_orders, n_resources)
        env = JobShopEnv(orders, resources)
        base_state = env.reset()
        busy = np.array([1.0 if env.resource_remaining[rid] > 0 else 0.0 for rid in env.resource_ids])
        state = np.concatenate([base_state, busy])

        done = False
        ep_reward = 0
        memory = []

        while not done:
            ready_ops = env.get_ready_ops_list()
            if not ready_ops:
                env._advance_until_resource_free()
                continue

            op_feat = np.zeros((action_dim, op_feat_dim))
            mask = np.zeros(action_dim, dtype=bool)
            for i, op_info in enumerate(ready_ops):
                if i >= action_dim:
                    break
                op_id = op_info['op_id']
                rid = op_info['resource_id']
                due_min = 0; rem_time = 0; weight = 0; progress = 0
                for order in env.orders:
                    for op in order.ops:
                        if op.id == op_id:
                            due_min = (order.due_date - env.start_datetime).total_seconds() / 60.0
                            rem_time = env.op_remaining[op.id]
                            weight = order.priority_weight
                            completed = sum(1 for o in order.ops if env.op_status[o.id] == 'completed')
                            progress = completed / len(order.ops)
                            break
                slack = (due_min - env.current_time) / 1440.0
                rem_time_norm = rem_time / 1440.0
                weight_norm = weight / 10.0
                res_free = 0.0 if env.resource_remaining[rid] > 0 else 1.0
                op_feat[i] = [slack, rem_time_norm, weight_norm, progress, res_free]
                mask[i] = True

            action, log_prob = agent.select_action(state, op_feat, mask)
            op_id = ready_ops[action]['op_id']
            next_base_state, reward, done, _ = env.step(op_id)

            reward -= 0.01 * env.current_time

            busy = np.array([1.0 if env.resource_remaining[rid] > 0 else 0.0 for rid in env.resource_ids])
            next_state = np.concatenate([next_base_state, busy])

            memory.append((state, op_feat, action, log_prob, reward, mask,
                           next_state, op_feat.copy(), done))
            ep_reward += reward

            state = next_state

        if memory:
            loss = agent.update(memory)

        if progress_callback:
            progress_callback(ep, loss)

        print(f"Episode {ep:4d} | Reward: {ep_reward:8.2f} | Loss: {loss:.4f}")

    agent.save("ppo_dispatcher.pth")
    print("Модель PPO сохранена в ppo_dispatcher.pth")

if __name__ == "__main__":
    train_ppo(episodes=200, lr=3e-4, n_orders=30, n_resources=3)