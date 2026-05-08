# train_ppo_from_bc.py
import pickle
import numpy as np
from ppo_agent import PPO
from job_shop_env import JobShopEnv
from generate_bc_data import generate_training_from_milp, parse_mk01, MK01_DATA
from optimizer_milp import build_milp_schedule
from datetime import datetime

def main():
    orders, resources = parse_mk01(MK01_DATA)
    schedule = build_milp_schedule(orders, resources, datetime(2026, 4, 27, 8, 0, 0))
    if not schedule:
        print("MILP failed")
        return

    dataset = generate_training_from_milp(orders, resources, schedule)

    env = JobShopEnv(orders, resources)
    state_dim = env.state_dim + len(env.resource_ids)
    op_feat_dim = 5
    action_dim = env.max_actions

    agent = PPO(state_dim, op_feat_dim, action_dim, device='cpu')

    for (state, op_feat, mask, action) in dataset:
        agent.update_single(state, op_feat, mask, action)

    agent.save("ppo_imitation.pth")
    print("PPO дообучен и сохранён как ppo_imitation.pth")

if __name__ == "__main__":
    main()