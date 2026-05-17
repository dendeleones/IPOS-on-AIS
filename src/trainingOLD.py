# training.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from PySide6.QtCore import Signal, QObject
import threading
import random
from datetime import datetime, timedelta
from datamodels import Order, Operation, Resource
from optimizer_milp import build_milp_schedule
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------ Нейросеть ------------------
class ResourceOpPointerNet(nn.Module):
    def __init__(self, d_res, d_op, embed_dim=128, hidden_dim=256, num_heads=8):
        super().__init__()
        self.d_res = d_res
        self.d_op = d_op
        self.embed_dim = embed_dim

        self.res_encoder = nn.Sequential(
            nn.Linear(d_res, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.op_encoder = nn.Sequential(
            nn.Linear(d_op, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, res_feat, op_feat, mask=None):
        batch_size, R, _ = res_feat.shape
        _, O, _ = op_feat.shape
        r_emb = self.res_encoder(res_feat)
        o_emb = self.op_encoder(op_feat)
        attn_out, _ = self.attn(query=o_emb, key=r_emb, value=r_emb)
        o_exp = attn_out.unsqueeze(2).expand(-1, -1, R, -1)
        r_exp = r_emb.unsqueeze(1).expand(-1, O, -1, -1)
        combined = torch.cat([o_exp, r_exp], dim=-1)
        combined = combined.reshape(batch_size * O * R, -1)
        logits = self.proj(combined).view(batch_size, O, R)
        if mask is not None:
            logits = logits.masked_fill(mask, -float('inf'))
        return torch.softmax(logits, dim=2)


# ------------------ Генерация данных через MILP ------------------
def generate_milp_dataset(num_samples=200, R=5, O=20, save_path=None, force_regen=False):
    """
    Генерирует (или загружает) датасет. Если save_path задан и файл существует (и не force_regen),
    загружает данные из него. Иначе генерирует и сохраняет.
    """
    if save_path and os.path.exists(save_path) and not force_regen:
        data = np.load(save_path)
        return data['res'], data['op'], data['tgt']

    d_res = 7
    d_op = 6
    types = ['фрезеровка', 'сборка', 'сварка']
    base_time = datetime(2026, 4, 27, 8, 0, 0)

    res_list, op_list, tgt_list = [], [], []

    for _ in range(num_samples):
        resources = []
        for r in range(R):
            res = Resource(id=f'R{r+1}', name=f'Станок {r+1}')
            res.status = 'Работает' if np.random.random() > 0.1 else 'Авария'
            res.load_minutes = np.random.uniform(0, 240)
            res.downtime_minutes = np.random.uniform(0, 60)
            res.work_hours = np.random.choice([24, 12, 6])
            res.repair = 1 if res.status == 'Авария' else 0
            res.reliability = np.random.uniform(0.7, 1.0)
            res.supported_types = np.random.choice(types, size=np.random.randint(1, 4), replace=False).tolist()
            resources.append(res)

        orders = []
        for o in range(O):
            op_type = np.random.choice(types)
            duration = np.random.uniform(10, 120)
            urgency = np.random.uniform(0.5, 2.0)
            slack = np.random.uniform(0, 48)
            due = base_time + timedelta(hours=random.randint(10, 200))
            order = Order(id=f'ORD_{o:03d}', due_date=due, priority_weight=urgency)
            op = Operation(
                id=f'ORD_{o:03d}_op1', order_id=order.id,
                item=f'Деталь_{random.randint(1,5)}', op_number=1,
                norm_duration=duration, type_id=op_type
            )
            op.urgency = urgency
            op.slack_hours = slack
            order.ops.append(op)
            orders.append(order)

        schedule = build_milp_schedule(orders, resources, base_time)
        if schedule is None:
            continue

        res_feat = np.zeros((R, d_res))
        for r_idx, res in enumerate(resources):
            res_feat[r_idx, 0] = 1.0 if res.status == 'Работает' else 0.0
            res_feat[r_idx, 1] = res.load_minutes / 240.0
            res_feat[r_idx, 2] = float(res.work_hours)
            type_onehot = np.zeros(3)
            for t in res.supported_types:
                type_onehot[types.index(t)] = 1.0
            res_feat[r_idx, 3:6] = type_onehot
            res_feat[r_idx, 6] = float(res.repair)

        op_feat = np.zeros((O, d_op))
        target = np.zeros(O, dtype=int)
        for o_idx, order in enumerate(orders):
            op = order.ops[0]
            type_onehot = np.zeros(3)
            if op.type_id in types:
                type_onehot[types.index(op.type_id)] = 1.0
            op_feat[o_idx, :3] = type_onehot
            op_feat[o_idx, 3] = op.norm_duration / 120.0
            op_feat[o_idx, 4] = op.urgency
            op_feat[o_idx, 5] = op.slack_hours / 48.0
            if op.id in schedule:
                assigned_rid = schedule[op.id]['resource_id']
                target[o_idx] = int(assigned_rid[1]) - 1
            else:
                target[o_idx] = 0

        res_list.append(res_feat)
        op_list.append(op_feat)
        tgt_list.append(target)

        if len(res_list) >= num_samples:
            break

    res_arr = np.array(res_list, dtype=np.float32)
    op_arr = np.array(op_list, dtype=np.float32)
    tgt_arr = np.array(tgt_list, dtype=np.long)

    if save_path:
        np.savez(save_path, res=res_arr, op=op_arr, tgt=tgt_arr)

    return res_arr, op_arr, tgt_arr


# ------------------ Сигналы и Тренер ------------------
class TrainingSignals(QObject):
    progress = Signal(int, float)
    log = Signal(str)

class SimpleDispatcher(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class Trainer:
    def __init__(self, signals=None):
        self.signals = signals
        self.model = None
        self.stop_event = threading.Event()
        self.history = {'epoch': [], 'train_loss': [], 'val_acc': []}   # <-- новое

    def save(self, path):
        """Сохраняет текущие веса модели в файл."""
        if self.model is None:
            return False
        try:
            torch.save(self.model.state_dict(), path)
            return True
        except Exception as e:
            if self.signals:
                self.signals.log.emit(f"Ошибка сохранения весов: {e}")
            return False

    def load(self, path):
        """Загружает веса из файла. Модель должна быть уже создана."""
        if self.model is None:
            return False
        try:
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
            if self.signals:
                self.signals.log.emit(f"Веса загружены из {path}")
            return True
        except Exception as e:
            if self.signals:
                self.signals.log.emit(f"Не удалось загрузить веса: {e}")
            return False

    def train_bc(self, epochs=50, batch_size=32, lr=1e-3, load_path=None, force_regen_data=False):
        self.stop_event.clear()

        # Загружаем новый правильный датасет
        if os.path.exists("gpss_dataset.npz"):
            data = np.load("gpss_dataset.npz")
            X = data['X']  # массив (samples, features)
            y = data['y']  # массив (samples,) – номера ресурсов
            if self.signals:
                self.signals.log.emit(
                    f"Загружен датасет GPSS: {X.shape[0]} примеров, признаков: {X.shape[1]}, классов: {np.max(y) + 1}")
        else:
            if self.signals:
                self.signals.log.emit("Файл gpss_dataset.npz не найден. Сгенерируйте его с помощью gpss_simulator.py")
            return None

        # Преобразуем в тензоры
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        train_size = int(0.8 * len(X))
        val_size = len(X) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        input_dim = X.shape[1]
        output_dim = len(np.unique(y))

        # Используем простой MLP (уже есть в файле), а не PointerNet, чтобы быстро получить результат
        self.model = SimpleDispatcher(input_dim, output_dim).to(torch.device("cpu"))

        if load_path:
            self.load(load_path)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            if self.stop_event.is_set():
                if self.signals:
                    self.signals.log.emit("Обучение остановлено пользователем.")
                break

            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    out = self.model(batch_x)
                    pred = out.argmax(dim=1)
                    correct += (pred == batch_y).sum().item()
                    total += batch_y.size(0)
            val_acc = correct / total if total > 0 else 0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict()

            if self.signals:
                self.signals.progress.emit(epoch, avg_train_loss)
                if epoch % 10 == 0 or epoch == 1:
                    self.signals.log.emit(
                        f"Эпоха {epoch:4d} | Потери: {avg_train_loss:.4f} | Точность: {val_acc:.4f} | Лучшая: {best_val_acc:.4f}"
                    )

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if self.signals:
                self.signals.log.emit(f"Лучшая точность валидации: {best_val_acc:.4f}")
        return self.model

    def stop(self):
        self.stop_event.set()