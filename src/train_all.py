# train_all.py
import os
import sys
import torch
import numpy as np

# Убедимся, что рабочая директория – src/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QObject, Signal
from training import Trainer, SimpleDispatcher
from train_slim import SelfLabelingTrainer
from train_ppo_dispatcher import train_ppo   # импортированная функция остаётся train_ppo


# ---------- Консольные сигналы ----------
class ConsoleSignals(QObject):
    progress = Signal(int, float)
    log = Signal(str)

    def __init__(self):
        super().__init__()
        self.progress.connect(lambda e, l: None)
        self.log.connect(lambda msg: print(msg))


# ---------- Обучение ----------
def train_pointer_net():
    print("===== Обучение PointerNet (BC) =====")
    signals = ConsoleSignals()
    trainer = Trainer(signals=signals)
    model = trainer.train_bc(epochs=100, lr=0.001, force_regen_data=False)
    if model:
        torch.save(model.state_dict(), "pointer_net.pth")
        print("Модель PointerNet сохранена в pointer_net.pth")


def train_slim():
    print("===== Обучение SLIM =====")
    signals = ConsoleSignals()
    slim = SelfLabelingTrainer(signals=signals)
    model = slim.train_slim(epochs=100, lr=0.001, dataset_path="gpss_dataset.npz",
                           unlabeled_path="unlabeled.npz", self_label_interval=30)
    if model:
        torch.save(model.state_dict(), "slim_pointer.pth")
        print("Модель SLIM сохранена в slim_pointer.pth")


def run_ppo():                                     # <-- переименовано
    print("===== Обучение PPO =====")
    def log_callback(msg):
        print(msg)
    model = train_ppo(episodes=200, lr=0.001, n_orders=30, n_resources=5,
                     progress_callback=None, log_callback=log_callback)
    if model:
        torch.save(model.state_dict(), "ppo_dispatcher.pth")
        print("Модель PPO сохранена в ppo_dispatcher.pth")


# ---------- Точка входа ----------
if __name__ == "__main__":
    if not os.path.exists("unlabeled.npz"):
        print("unlabeled.npz не найден, создаю заглушку из 10 примеров...")
        if os.path.exists("gpss_dataset.npz"):
            data = np.load("gpss_dataset.npz")
            X_sample = data['X'][:10]
            np.savez("unlabeled.npz", X=X_sample)
        else:
            print("Ошибка: gpss_dataset.npz не найден. Сгенерируйте его с помощью gpss_simulator.py")
            sys.exit(1)

    #train_pointer_net()
    #train_slim()
    run_ppo()          # <-- переименовано
    print("Все модели обучены.")