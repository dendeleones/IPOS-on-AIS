# train_slim.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from training import SimpleDispatcher, TrainingSignals
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class SelfLabelingTrainer:
    def __init__(self, signals=None):
        self.signals = signals
        self.model = None
        self.device = torch.device("cpu")
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.y_true = None
        self.y_pred = None
        self.class_names = None

    def train_slim(self, epochs=100, batch_size=32, lr=1e-3,
                   save_path="slim_pointer.pth",
                   dataset_path="gpss_dataset.npz",
                   unlabeled_path=None,
                   self_label_interval=30,
                   confidence_threshold=0.95):
        if not os.path.exists(dataset_path):
            if self.signals:
                self.signals.log.emit(f"Файл {dataset_path} не найден.")
            return None
        data = np.load(dataset_path)
        X, y = data['X'], data['y']
        if self.signals:
            self.signals.log.emit(f"Загружен датасет GPSS: {X.shape[0]} примеров, признаков: {X.shape[1]}, классов: {np.max(y)+1}")

        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        self.class_names = [str(i) for i in range(output_dim)]

        self.model = SimpleDispatcher(input_dim, output_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        unlabeled_X = None
        if unlabeled_path and os.path.exists(unlabeled_path):
            unlabeled_data = np.load(unlabeled_path)
            unlabeled_X = unlabeled_data['X']
            if self.signals:
                self.signals.log.emit(f"Загружено неразмеченных примеров: {len(unlabeled_X)}")

        best_val_acc = 0.0
        best_state = None
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(1, epochs + 1):
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

            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            val_loader = DataLoader(val_set, batch_size=batch_size)
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
            self.history['train_loss'].append(avg_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict()
                self.y_true = all_labels
                self.y_pred = all_preds

            if self.signals:
                self.signals.progress.emit(epoch, avg_loss)
                if epoch % 10 == 0 or epoch == 1:
                    self.signals.log.emit(
                        f"Эпоха {epoch:4d} | Потери: {avg_loss:.4f} | Точность: {val_acc:.4f} | Лучшая: {best_val_acc:.4f}"
                    )

            # Self‑labeling
            if unlabeled_X is not None and epoch % self_label_interval == 0 and epoch > 0:
                self.model.eval()
                new_examples = 0
                with torch.no_grad():
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
        self.save_metrics("SLIM")
        return self.model

    def save_metrics(self, model_name):
        if not self.history:
            return
        import os, json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"metrics/{model_name}/{timestamp}"
        os.makedirs(folder, exist_ok=True)

        with open(f"{folder}/history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        plt.figure()
        plt.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Learning Curve')
        plt.legend()
        plt.savefig(f"{folder}/loss.png")
        plt.close()

        plt.figure()
        plt.plot(self.history['epoch'], self.history['val_acc'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name} Validation Accuracy')
        plt.legend()
        plt.savefig(f"{folder}/accuracy.png")
        plt.close()

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

            report = classification_report(self.y_true, self.y_pred, target_names=self.class_names, output_dict=True)
            with open(f"{folder}/classification_report.json", 'w') as f:
                json.dump(report, f, indent=2)

        if self.signals:
            self.signals.log.emit(f"Метрики SLIM сохранены в {folder}")