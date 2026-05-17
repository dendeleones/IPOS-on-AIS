# train_slim.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from training import SimpleDispatcher, TrainingSignals
import os

class SelfLabelingTrainer:
    def __init__(self, signals=None):
        self.signals = signals
        self.model = None
        self.device = torch.device("cpu")

    def train_slim(self, epochs=100, batch_size=32, lr=1e-3,
                   save_path="slim_pointer.pth",
                   dataset_path="gpss_dataset.npz",
                   unlabeled_path=None,
                   self_label_interval=50,
                   confidence_threshold=0.9):
        """
        Параметры:
          - unlabeled_path: путь к .npz с массивом X_unlabeled (без меток).
          - self_label_interval: каждые сколько эпох делать срез.
          - confidence_threshold: минимальная уверенность для добавления примера.
        """
        # Загружаем размеченные данные
        if not os.path.exists(dataset_path):
            if self.signals:
                self.signals.log.emit(f"Файл {dataset_path} не найден. Сгенерируйте его с помощью gpss_simulator.py")
            return None
        data = np.load(dataset_path)
        X, y = data['X'], data['y']
        if self.signals:
            self.signals.log.emit(f"Загружен датасет GPSS: {X.shape[0]} примеров, признаков: {X.shape[1]}, классов: {np.max(y)+1}")

        input_dim = X.shape[1]
        output_dim = len(np.unique(y))

        # Инициализируем модель
        self.model = SimpleDispatcher(input_dim, output_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Создаём DataLoader для начального датасета
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        # Загружаем неразмеченные данные (если есть)
        unlabeled_X = None
        if unlabeled_path and os.path.exists(unlabeled_path):
            unlabeled_data = np.load(unlabeled_path)
            unlabeled_X = unlabeled_data['X']  # shape (N, input_dim)
            if self.signals:
                self.signals.log.emit(f"Загружено неразмеченных примеров: {len(unlabeled_X)}")
        else:
            if self.signals:
                self.signals.log.emit("Неразмеченные данные не указаны. Self‑labeling отключен.")

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            # Обычная эпоха обучения
            self.model.train()
            train_loss = 0
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_loss = train_loss / len(train_loader)

            # Валидация
            self.model.eval()
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            correct = 0
            total = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    out = self.model(bx)
                    pred = out.argmax(dim=1)
                    correct += (pred == by).sum().item()
                    total += by.size(0)
            val_acc = correct / total if total > 0 else 0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict()

            if self.signals:
                self.signals.progress.emit(epoch, avg_loss)
                if epoch % 10 == 0 or epoch == 1:
                    self.signals.log.emit(
                        f"Эпоха {epoch:4d} | Потери: {avg_loss:.4f} | Точность: {val_acc:.4f} | Лучшая: {best_val_acc:.4f}"
                    )

            # Self‑labeling (если есть неразмеченные данные и подошёл интервал)
            if unlabeled_X is not None and epoch % self_label_interval == 0 and epoch > 0:
                self.model.eval()
                new_examples = 0
                with torch.no_grad():
                    # Разбиваем неразмеченные данные на батчи для предсказаний
                    unlabeled_tensor = torch.FloatTensor(unlabeled_X)
                    pred_dataset = TensorDataset(unlabeled_tensor)
                    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
                    confident_X, confident_y = [], []
                    for (batch_x,) in pred_loader:
                        logits = self.model(batch_x)
                        probs = torch.softmax(logits, dim=1)
                        max_probs, preds = probs.max(dim=1)
                        mask = max_probs >= confidence_threshold
                        if mask.any():
                            confident_X.append(batch_x[mask])
                            confident_y.append(preds[mask])
                    if confident_X:
                        new_X = torch.cat(confident_X).cpu().numpy()
                        new_y = torch.cat(confident_y).cpu().numpy()
                        # Добавляем в тренировочный набор
                        new_dataset = TensorDataset(torch.FloatTensor(new_X), torch.LongTensor(new_y))
                        train_set = ConcatDataset([train_set, new_dataset])
                        new_examples = len(new_X)
                if self.signals and new_examples > 0:
                    self.signals.log.emit(f"Self‑labeling: добавлено {new_examples} новых примеров.")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if self.signals:
                self.signals.log.emit(f"Лучшая точность валидации: {best_val_acc:.4f}")
        torch.save(self.model.state_dict(), save_path)
        if self.signals:
            self.signals.log.emit(f"Модель сохранена как {save_path}")
        return self.model