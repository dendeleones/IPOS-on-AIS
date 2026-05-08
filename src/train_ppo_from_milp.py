# train_ppo_from_milp.py
import numpy as np
from datetime import datetime
from job_shop_env import JobShopEnv
from ppo_agent import PPO
from benchmark_runner import parse_mk01, MK01_DATA
from optimizer_milp import build_milp_schedule
from generate_bc_data import generate_training_from_milp

def train():
    # 1. Получаем экспертные данные от MILP
    orders, resources = parse_mk01(MK01_DATA)
    schedule = build_milp_schedule(orders, resources, datetime(2026, 4, 27, 8, 0, 0))
    if not schedule:
        print("MILP не нашёл решение.")
        return
    dataset = generate_training_from_milp(orders, resources, schedule)
    print(f"Собрано {len(dataset)} экспертных примеров.")

    # 2. Создаём агента с правильными размерами
    env = JobShopEnv(orders, resources)
    base_state = env.reset()
    busy = np.array([1.0 if env.resource_remaining[rid] > 0 else 0.0 for rid in env.resource_ids])
    ext_state = np.concatenate([base_state, busy])
    state_dim = len(ext_state)
    op_feat_dim = 5
    action_dim = env.max_actions

    agent = PPO(state_dim, op_feat_dim, action_dim, lr=3e-4, device='cpu')

    # 3. Имитационное обучение (Behaviour Cloning) на MILP‑примерах
    for (state, op_feat, mask, action) in dataset:
        loss = agent.update_single(state, op_feat, mask, action)
    print("Имитационное обучение завершено.")

    # 4. Несколько RL‑эпизодов для донастройки (shaping награда)
    for ep in range(50):
        env = JobShopEnv(orders, resources)
        base_state = env.reset()
        busy = np.array([1.0 if env.resource_remaining[rid] > 0 else 0.0 for rid in env.resource_ids])
        state = np.concatenate([base_state, busy])
        done = False
        memory = []

        while not done:
            ready_ops = env.get_ready_ops_list()
            if not ready_ops:
                env._advance_until_resource_free()
                continue

            # Признаки операций
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
            next_base_state, _, done, _ = env.step(op_id)

            # Shaping‑награда
            reward = -0.01 * env.current_time - 0.05 * len(ready_ops)
            if done:
                reward += 50  # бонус за завершение

            busy = np.array([1.0 if env.resource_remaining[rid] > 0 else 0.0 for rid in env.resource_ids])
            next_state = np.concatenate([next_base_state, busy])

            memory.append((state, op_feat, action, log_prob, reward, mask,
                           next_state, op_feat.copy(), done))
            state = next_state

        if memory:
            loss = agent.update(memory)
            print(f"RL Ep {ep:3d} | Loss: {loss:.4f}")
        else:
            print(f"RL Ep {ep:3d} | no memory")

    agent.save("ppo_from_milp.pth")
    print("Модель сохранена как ppo_from_milp.pth")

if __name__ == "__main__":
    train()