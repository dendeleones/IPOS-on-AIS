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
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # для красивой матрицы (опционально), если нет, можно без

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

# ------------------ Простая модель ------------------
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

# ------------------ Сигналы и Тренер ------------------
class TrainingSignals(QObject):
    progress = Signal(int, float)
    log = Signal(str)

class Trainer:
    def __init__(self, signals=None):
        self.signals = signals
        self.model = None
        self.stop_event = threading.Event()
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.y_true = None
        self.y_pred = None
        self.class_names = None

    def save(self, path):
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
        if self.signals:
            self.signals.log.emit("Загрузка/генерация данных...")

        # Используем GPSS датасет
        if os.path.exists("gpss_dataset.npz"):
            data = np.load("gpss_dataset.npz")
            X = data['X']
            y = data['y']
            if self.signals:
                self.signals.log.emit(f"Загружен датасет GPSS: {X.shape[0]} примеров, признаков: {X.shape[1]}, классов: {np.max(y)+1}")
        else:
            if self.signals:
                self.signals.log.emit("gpss_dataset.npz не найден. Сгенерируйте его.")
            return None

        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        self.class_names = [str(i) for i in range(output_dim)]  # имена классов (индексы ресурсов)

        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        self.model = SimpleDispatcher(input_dim, output_dim)
        if load_path:
            self.load(load_path)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_model_state = None

        self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(1, epochs + 1):
            if self.stop_event.is_set():
                if self.signals:
                    self.signals.log.emit("Обучение остановлено пользователем.")
                break

            self.model.train()
            train_loss = 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for bx, by in val_loader:
                    out = self.model(bx)
                    loss = criterion(out, by)
                    val_loss += loss.item()
                    pred = out.argmax(dim=1)
                    correct += (pred == by).sum().item()
                    total += by.size(0)
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(by.cpu().numpy())
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total if total > 0 else 0

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict()
                # сохраняем предсказания лучшей эпохи для матрицы ошибок
                self.y_true = all_labels
                self.y_pred = all_preds

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

        # Сохраняем метрики
        self.save_metrics("PointerNet")
        return self.model

    def save_metrics(self, model_name):
        if not self.history:
            return
        import os, json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"metrics/{model_name}/{timestamp}"
        os.makedirs(folder, exist_ok=True)

        # JSON с историей
        with open(f"{folder}/history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        # График потерь
        plt.figure()
        plt.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Learning Curve')
        plt.legend()
        plt.savefig(f"{folder}/loss.png")
        plt.close()

        # График точности
        plt.figure()
        plt.plot(self.history['epoch'], self.history['val_acc'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name} Validation Accuracy')
        plt.legend()
        plt.savefig(f"{folder}/accuracy.png")
        plt.close()

        # Матрица ошибок и отчёт по классификации (только если есть разметка)
        if self.y_true is not None and self.y_pred is not None and self.class_names is not None:
            cm = confusion_matrix(self.y_true, self.y_pred)
            plt.figure(figsize=(8, 6))
            try:
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
            except ImportError:
                plt.imshow(cm, interpolation='nearest', cmap='Blues')
                plt.colorbar()
            plt.title(f'{model_name} Confusion Matrix (Validation)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f"{folder}/confusion_matrix.png")
            plt.close()

            # Classification report
            report = classification_report(self.y_true, self.y_pred, target_names=self.class_names, output_dict=True)
            with open(f"{folder}/classification_report.json", 'w') as f:
                json.dump(report, f, indent=2)

        if self.signals:
            self.signals.log.emit(f"Метрики сохранены в {folder}")

    def stop(self):
        self.stop_event.set()