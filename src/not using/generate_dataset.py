import numpy as np
import random

def generate_dataset(num_samples=5000, R=5, O=20, save_path="smart_dataset.npz"):
    """
    Генерирует обучающие примеры по правилу LWKR:
    - Случайно создаёт ресурсы и операции.
    - Для каждой операции допустимые ресурсы — те, у которых есть общий тип.
    - Целевой ресурс = среди допустимых с минимальной суммарной загрузкой (workload).
    Возвращает X (признаки) и y (целевые номера ресурсов).
    """
    # Типы операций (3 штуки)
    types = [0, 1, 2]  # можно заменить на имена, но для one-hot достаточно индексов
    d_op = 6           # признаков операции
    d_res = 6          # признаков ресурса

    X, y = [], []

    for _ in range(num_samples):
        # --- Создаём ресурсы ---
        # Каждый ресурс имеет случайный набор типов, которые он поддерживает
        res_supported = [np.random.choice(types, size=np.random.randint(1, 4), replace=False).tolist() for _ in range(R)]
        # Признаки ресурсов, которые не зависят от операции (готовность, ремонт, надёжность)
        res_status   = [1.0 if random.random() > 0.1 else 0.0 for _ in range(R)]  # готовность
        res_repair   = [1.0 if st == 0.0 else 0.0 for st in res_status]
        res_reliab   = [0.9 + 0.1*random.random() for _ in range(R)]

        # --- Генерируем операции ---
        op_types    = [random.choice(types) for _ in range(O)]
        durations   = [random.uniform(10, 120) for _ in range(O)]
        priorities  = [random.uniform(0.5, 2.0) for _ in range(O)]
        slacks      = [random.uniform(0, 48) for _ in range(O)]

        # Моделируем назначение по правилу: проходим по операциям, поддерживая текущую загрузку ресурсов
        workloads = [0.0] * R

        for i in range(O):
            # Допустимые ресурсы для этой операции (пересечение типа)
            allowed = [r for r in range(R) if op_types[i] in res_supported[r]]
            if not allowed:
                allowed = list(range(R))   # если ни один не подходит, разрешаем все

            # Правило LWKR: выбираем ресурс с минимальной загрузкой среди допустимых
            best_r = min(allowed, key=lambda r: workloads[r])
            y.append(best_r)

            # Признаки операции (6 чисел)
            type_onehot = np.zeros(3)
            type_onehot[op_types[i]] = 1.0
            op_feat = np.concatenate([
                type_onehot,
                [durations[i] / 120.0, priorities[i], slacks[i] / 48.0]
            ])

            # Признаки ресурсов для этой операции
            res_feat = []
            for r in range(R):
                ready = res_status[r]
                # Текущая загрузка (workload) нормированная
                cur_load = workloads[r] / 240.0   # max 240 min
                # Совместимость: 1.0 если ресурс в allowed, иначе 0
                compat = 1.0 if r in allowed else 0.0
                # Дополнительные признаки
                repair = res_repair[r]
                reliability = res_reliab[r]
                res_feat.append([ready, cur_load, compat, workloads[r]/240.0, repair, reliability])
            res_feat = np.array(res_feat).flatten()

            full_feat = np.concatenate([op_feat, res_feat])
            X.append(full_feat)

            # Обновляем загрузку выбранного ресурса
            workloads[best_r] += durations[i]

            if len(X) >= num_samples:
                break
        if len(X) >= num_samples:
            break

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    if save_path:
        np.savez(save_path, X=X, y=y)
    print(f"Сгенерирован датасет: {X.shape[0]} примеров, {X.shape[1]} признаков, {np.max(y)+1} ресурсов")
    return X, y

if __name__ == "__main__":
    generate_dataset(10000, R=5, O=20, save_path="smart_dataset.npz")