# train_slim.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from training import ResourceOpPointerNet, generate_milp_dataset, TrainingSignals
from optimizer_milp import build_milp_schedule
from datetime import datetime, timedelta
import random
import os
import threading

class SelfLabelingTrainer:
    def __init__(self, signals=None, d_res=7, d_op=6, R=5, O=20):
        self.signals = signals
        self.model = ResourceOpPointerNet(d_res, d_op)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.R = R
        self.O = O
        self.device = 'cpu'
        self.stop_event = threading.Event()

    def generate_initial_dataset(self, num_samples=200):
        """Генерирует начальный набор данных через MILP."""
        if self.signals:
            self.signals.log.emit("Генерация начальных данных MILP...")
        return generate_milp_dataset(num_samples, self.R, self.O)

    def _create_random_instance(self):
        """Создаёт случайные ресурсы и заказы (аналогично generate_milp_dataset)."""
        from datamodels import Order, Operation, Resource
        types = ['фрезеровка', 'сборка', 'сварка']
        base_time = datetime(2026, 4, 27, 8, 0, 0)

        resources = []
        for r in range(self.R):
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
        for o in range(self.O):
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

        return resources, orders

    def _resources_to_tensor(self, resources):
        """Преобразует список ресурсов в np.array (R, d_res)."""
        types = ['фрезеровка', 'сборка', 'сварка']
        d_res = 7
        res_feat = np.zeros((self.R, d_res))
        for r_idx, res in enumerate(resources):
            res_feat[r_idx, 0] = 1.0 if res.status == 'Работает' else 0.0
            res_feat[r_idx, 1] = res.load_minutes / 240.0
            res_feat[r_idx, 2] = float(res.work_hours)
            type_onehot = np.zeros(3)
            for t in res.supported_types:
                type_onehot[types.index(t)] = 1.0
            res_feat[r_idx, 3:6] = type_onehot
            res_feat[r_idx, 6] = float(res.repair)
        return res_feat

    def _operations_to_tensor(self, orders):
        """Преобразует список заказов в np.array (O, d_op)."""
        types = ['фрезеровка', 'сборка', 'сварка']
        d_op = 6
        op_feat = np.zeros((self.O, d_op))
        for o_idx, order in enumerate(orders):
            op = order.ops[0]
            type_onehot = np.zeros(3)
            if op.type_id in types:
                type_onehot[types.index(op.type_id)] = 1.0
            op_feat[o_idx, :3] = type_onehot
            op_feat[o_idx, 3] = op.norm_duration / 120.0
            op_feat[o_idx, 4] = op.urgency
            op_feat[o_idx, 5] = op.slack_hours / 48.0
        return op_feat

    def _apply_predictions(self, orders, resources, pred_indices, base_time):
        """Создаёт расписание на основе предсказанных индексов ресурсов."""
        schedule = {}
        current_starts = {r.id: base_time for r in resources}
        # Сортируем операции по срочности (для упрощения)
        sorted_ops = sorted(
            [(o_idx, order.ops[0]) for o_idx, order in enumerate(orders)],
            key=lambda x: x[1].urgency, reverse=True
        )
        for o_idx, op in sorted_ops:
            res_id = f'R{pred_indices[o_idx]+1}'
            start = current_starts[res_id]
            end = start + timedelta(minutes=op.norm_duration)
            schedule[op.id] = {'start': start, 'end': end, 'resource_id': res_id}
            current_starts[res_id] = end
        return schedule

    def _calc_makespan(self, schedule, base_time):
        if not schedule:
            return float('inf')
        return max(info['end'] for info in schedule.values()) - base_time

    def generate_self_labeled(self, model, num_samples=50):
        """Генерирует новые примеры, используя текущую модель + MILP‑оценку."""
        res_list, op_list, tgt_list = [], [], []
        base_time = datetime(2026, 4, 27, 8, 0, 0)

        for _ in range(num_samples):
            resources, orders = self._create_random_instance()
            res_feat = self._resources_to_tensor(resources)
            op_feat = self._operations_to_tensor(orders)

            # Предсказание модели
            with torch.no_grad():
                probs = model(
                    torch.FloatTensor(res_feat).unsqueeze(0),
                    torch.FloatTensor(op_feat).unsqueeze(0)
                )
                pred = probs.argmax(dim=2).squeeze(0).numpy()

            # Применяем предсказание и запускаем MILP
            our_schedule = self._apply_predictions(orders, resources, pred, base_time)
            milp_schedule = build_milp_schedule(orders, resources, base_time)
            if milp_schedule is None:
                continue

            our_makespan = self._calc_makespan(our_schedule, base_time)
            milp_makespan = self._calc_makespan(milp_schedule, base_time)

            # Если наше решение достаточно хорошее — берём MILP как цель
            if our_makespan <= milp_makespan * 1.2:
                target = np.zeros(self.O, dtype=int)
                for o_idx, order in enumerate(orders):
                    op = order.ops[0]
                    if op.id in milp_schedule:
                        assigned_rid = milp_schedule[op.id]['resource_id']
                        target[o_idx] = int(assigned_rid[1]) - 1
                res_list.append(res_feat)
                op_list.append(op_feat)
                tgt_list.append(target)

        if len(res_list) == 0:
            return None, None, None
        return (np.array(res_list, dtype=np.float32),
                np.array(op_list, dtype=np.float32),
                np.array(tgt_list, dtype=np.long))

    def train_slim(self, epochs=500, batch_size=32, self_label_interval=50, self_label_samples=50):
        """Основной цикл SLIM‑обучения."""
        # 1. Загрузка или создание начального датасета
        if os.path.exists("slim_initial_data.npz"):
            data = np.load("slim_initial_data.npz")
            res_feat, op_feat, target = data['res'], data['op'], data['tgt']
            if self.signals:
                self.signals.log.emit("Загружены сохранённые начальные данные.")
        else:
            res_feat, op_feat, target = self.generate_initial_dataset(200)
            np.savez("slim_initial_data.npz", res=res_feat, op=op_feat, tgt=target)

        dataset = TensorDataset(
            torch.FloatTensor(res_feat), torch.FloatTensor(op_feat), torch.LongTensor(target)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            if self.stop_event.is_set():
                if self.signals:
                    self.signals.log.emit("Обучение остановлено пользователем.")
                break

            # Дообучение на текущем датасете
            self.model.train()
            total_loss = 0
            for res, ops, tgt in loader:
                self.optimizer.zero_grad()
                probs = self.model(res, ops)
                loss = self.criterion(probs.view(-1, probs.size(-1)), tgt.view(-1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)

            # Каждые self_label_interval эпох генерируем новые данные
            if epoch % self_label_interval == 0:
                if self.signals:
                    self.signals.log.emit(f"Эпоха {epoch}: генерация self‑labeled данных...")
                new_res, new_op, new_tgt = self.generate_self_labeled(self.model, self_label_samples)
                if new_res is not None:
                    res_feat = np.concatenate([res_feat, new_res])
                    op_feat = np.concatenate([op_feat, new_op])
                    target = np.concatenate([target, new_tgt])
                    dataset = TensorDataset(
                        torch.FloatTensor(res_feat), torch.FloatTensor(op_feat), torch.LongTensor(target)
                    )
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    if self.signals:
                        self.signals.log.emit(f"Добавлено {len(new_res)} новых примеров.")

            if self.signals:
                self.signals.progress.emit(epoch, avg_loss)
                if epoch % 10 == 0:
                    self.signals.log.emit(f"Эпоха {epoch:4d} | Потери: {avg_loss:.4f}")

        return self.model

    def stop(self):
        self.stop_event.set()