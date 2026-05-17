# ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DenseActor(nn.Module):
    def __init__(self, state_dim, op_feat_dim, action_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + op_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, state, op_feat, mask=None):
        # state: [batch, state_dim]
        # op_feat: [batch, max_ops, op_feat_dim]
        batch_size, max_ops, _ = op_feat.shape
        # Повторяем state для каждой операции
        state_exp = state.unsqueeze(1).expand(-1, max_ops, -1)   # [batch, max_ops, state_dim]
        combined = torch.cat([state_exp, op_feat], dim=2)        # [batch, max_ops, state_dim+op_feat_dim]
        logits = self.net(combined)                                # [batch, max_ops, action_dim]
        logits = logits.sum(dim=2)                                 # [batch, max_ops]
        if mask is not None:
            logits = logits.masked_fill(~mask, -float('inf'))
        return logits


class PPO:
    def __init__(self, state_dim, op_feat_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_eps=0.2, epochs=5, device='cpu'):
        self.device = device
        self.actor = DenseActor(state_dim, op_feat_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.action_dim = action_dim

    def select_action(self, state_np, op_feat_np, mask_np):
        state = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
        op_feat = torch.FloatTensor(op_feat_np).unsqueeze(0).to(self.device)
        mask = torch.BoolTensor(mask_np).unsqueeze(0).to(self.device)

        logits = self.actor(state, op_feat, mask)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, memory):
        if not memory:
            return 0.0
        states = torch.FloatTensor(np.array([m[0] for m in memory])).to(self.device)
        op_feats = torch.FloatTensor(np.array([m[1] for m in memory])).to(self.device)
        actions = torch.LongTensor(np.array([m[2] for m in memory])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array([m[3] for m in memory])).to(self.device)
        rewards = torch.FloatTensor(np.array([m[4] for m in memory])).to(self.device)
        masks = torch.BoolTensor(np.array([m[5] for m in memory])).to(self.device)

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.epochs):
            logits = self.actor(states, op_feats, masks)
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            loss = policy_loss  # больше не вычитаем энтропию

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / self.epochs

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))

    def update_single(self, state_np, op_feat_np, mask_np, expert_action):
        """Один шаг имитационного обучения (Behaviour Cloning)."""
        state = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
        op_feat = torch.FloatTensor(op_feat_np).unsqueeze(0).to(self.device)
        mask = torch.BoolTensor(mask_np).unsqueeze(0).to(self.device)
        expert_a = torch.LongTensor([expert_action]).to(self.device)

        logits = self.actor(state, op_feat, mask)
        loss = nn.CrossEntropyLoss()(logits, expert_a)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()