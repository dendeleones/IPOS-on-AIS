import numpy as np
from datamodels import Order, Operation, Resource
import os

# --- Чтение FJSP-файла (формат Brandimarte/Kacem) ---
def load_fjsp_instance(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    n_jobs, n_machines = map(int, lines[0].split())
    jobs = []
    idx = 1
    for j in range(n_jobs):
        op_count = int(lines[idx])
        idx += 1
        ops = []
        for _ in range(op_count):
            parts = lines[idx].split()
            idx += 1
            # Для простоты берём только первую альтернативу (или все? лучше все)
            # В формате: для каждой операции: количество_альтернатив (alt_count), затем alt_count пар (machine_id, time)
            alt_count = int(parts[0])
            alts = []
            for a in range(alt_count):
                m_id = int(parts[1 + 2*a])
                time = int(parts[1 + 2*a + 1])
                alts.append((m_id, time))
            ops.append(alts)
        jobs.append(ops)
    return jobs, n_machines

# --- Создание признаков ---
# Ресурсы: допустим, столько станков, сколько в задаче
# Признаки ресурса (7 чисел): статус, загрузка, часы работы, one-hot типов (3), ремонт
# Для упрощения: загрузка = 0, статус = 1, рабочие часы = 24, типы не учитываем (one-hot нули), ремонт = 0
def make_resource_features(n_machines):
    res = np.zeros((n_machines, 7), dtype=np.float32)
    res[:, 0] = 1.0    # все работают
    res[:, 2] = 24.0   # часы работы (нормировку сделаем позже)
    return res

# Признаки операции (6 чисел): one-hot типа (3), длительность, приоритет, slack
# Тип операции зададим как номер заказа (для разнообразия), но лучше взять случайный категориальный признак
# Пока закодируем номер заказа как one-hot по модулю 3
def make_op_features(jobs, job_id, op_idx, all_ops):
    op = jobs[job_id][op_idx]
    # Берём минимальное время среди альтернатив как длительность
    min_time = min(t for _, t in op)
    # Приоритет и slack генерируем случайно либо берём из порядка
    urgency = np.random.uniform(0.5, 2.0)
    slack = np.random.uniform(0, 48)
    # One-hot типа: три бита, можно закодировать (job_id % 3) и т.д.
    type_onehot = np.zeros(3, dtype=np.float32)
    type_onehot[job_id % 3] = 1.0
    feat = np.array([
        *type_onehot,
        min_time / 120.0,      # нормировка как в обучении
        urgency,
        slack / 48.0
    ], dtype=np.float32)
    return feat

# Собираем датасет
def build_dataset_from_fjsp(dir_path, max_samples=2000):
    res_list, op_list, tgt_list = [], [], []
    for fname in os.listdir(dir_path):
        if not fname.endswith('.fjs') and not fname.endswith('.txt'):
            continue
        full_path = os.path.join(dir_path, fname)
        jobs, n_machines = load_fjsp_instance(full_path)

        # Для каждого заказа и операции создаём отдельный пример
        # Целевой ресурс: выбираем станок с минимальным временем (жадное правило)
        for jid, job_ops in enumerate(jobs):
            for op_idx, alts in enumerate(job_ops):
                # Выбор цели: argmin по времени
                best_m, best_t = min(alts, key=lambda x: x[1])
                target = best_m - 1   # индексация с 0

                res_feat = make_resource_features(n_machines)
                op_feat = make_op_features(jobs, jid, op_idx, job_ops)

                res_list.append(res_feat)
                op_list.append(op_feat)
                tgt_list.append(target)

                if len(res_list) >= max_samples:
                    break
            if len(res_list) >= max_samples:
                break
        if len(res_list) >= max_samples:
            break

    return np.array(res_list), np.array(op_list), np.array(tgt_list)

if __name__ == "__main__":
    # Укажите папку с файлами .fjs (например, Brandimarte)
    res, op, tgt = build_dataset_from_fjsp("Brandimarte", max_samples=5000)
    np.savez("fjsp_dataset.npz", res=res, op=op, tgt=tgt)
    print(f"Датасет сохранён: {res.shape[0]} примеров")