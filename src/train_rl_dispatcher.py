# train_rl_dispatcher.py
import numpy as np
import random
from datetime import datetime, timedelta
from datamodels import Order, Operation, Resource
from job_shop_env import JobShopEnv
from dqn_agent import DQNAgent

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

def train_agent():
    ACTION_DIM = 150
    agent = DQNAgent(state_dim=61, action_dim=ACTION_DIM, lr=3e-4, gamma=0.95,
                     epsilon=1.0, epsilon_decay=0.9999, min_epsilon=0.15, device='cpu')
    num_episodes = 200
    target_update = 5
    log_interval = 1
    best_mean_reward = -float('inf')
    reward_history = []
    patience = 200
    no_improve = 0

    for episode in range(1, num_episodes+1):
        orders, resources = generate_random_orders(30, 3, seed=None)
        env = JobShopEnv(orders, resources, max_queue_per_resource=5)
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            ready_ops = env.get_ready_ops_list()
            mask = [False] * ACTION_DIM
            for i, op in enumerate(ready_ops):
                if i < ACTION_DIM:
                    mask[i] = True
            if not any(mask):
                # Нет доступных действий - продвигаем время, пока не появится
                env._advance_until_resource_free()
                # обновляем state
                state = env._get_state()
                continue
            action = agent.select_action(state, mask)
            op_id = ready_ops[action]['op_id']
            next_state, reward, done, _ = env.step(op_id)
            agent.store(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            ep_reward += reward

        reward_history.append(ep_reward)

        if episode % target_update == 0:
            agent.update_target()

        # Скользящее среднее за 100 эпизодов
        if episode >= 100:
            mean100 = np.mean(reward_history[-100:])
            if mean100 > best_mean_reward:
                best_mean_reward = mean100
                no_improve = 0
                agent.save("rl_dispatcher_best.pth")
                print(f"Ep {episode}: New best mean100: {mean100:.2f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at episode {episode}, no improvement for {patience} episodes.")
                    break

        if episode % log_interval == 0:
            print(f"Ep {episode:4d} | Reward: {ep_reward:8.2f} | Epsilon: {agent.epsilon:.4f} | Best mean100: {best_mean_reward:.2f}")

    print("Обучение завершено.")

if __name__ == "__main__":
    train_agent()