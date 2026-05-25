import numpy as np
import random

def simulate(num_samples=5000, R=5, save_path="gpss_dataset.npz"):
    types = ['фрезеровка', 'сборка', 'сварка']
    d_op = 6
    d_res = 6
    X, y = [], []

    for _ in range(num_samples):
        # Ресурсы со случайной готовностью и ремонтом
        res_supported = [random.sample(types, k=random.randint(1, 3)) for _ in range(R)]
        res_reliability = [0.5 + 0.5 * random.random() for _ in range(R)]  # 0.5..1.0
        res_repair = [1 if random.random() < 0.1 else 0 for _ in range(R)]
        # Если в ремонте, готовность падает
        for i in range(R):
            if res_repair[i] == 1:
                res_reliability[i] *= 0.5  # сильное снижение
        # Статус (готовность > 0.5 считается работающей, иначе авария)
        res_status = [1.0 if rel > 0.5 else 0.0 for rel in res_reliability]

        workloads = [0.0] * R
        # Генерируем одну операцию (на самом деле генерируем цикл по операциям?
        # Лучше генерировать несколько операций последовательно, чтобы загрузка накапливалась.
        # Сделаем от 1 до 10 операций за шаг, чтобы модель видела разную загрузку.
        n_ops = random.randint(1, 10)
        for _ in range(n_ops):
            op_type = random.choice(types)
            duration = random.uniform(10, 120)
            urgency = random.uniform(0.5, 2.0)
            slack = random.uniform(0, 48)

            allowed = [r for r in range(R) if op_type in res_supported[r]]
            if not allowed:
                allowed = list(range(R))

            # Правило: выбираем ресурс с минимальной эффективной загрузкой (workload / reliability)
            def effective_load(r):
                if res_reliability[r] == 0:
                    return float('inf')
                return workloads[r] / res_reliability[r]
            best_r = min(allowed, key=effective_load)
            y.append(best_r)

            # Признаки операции
            type_onehot = np.zeros(3)
            type_onehot[types.index(op_type)] = 1.0
            op_feat = np.concatenate([type_onehot, [duration / 120.0, urgency, slack / 48.0]])

            # Признаки ресурсов
            res_feat = []
            for r in range(R):
                ready = res_status[r]
                cur_load = workloads[r] / 240.0
                compat = 1.0 if r in allowed else 0.0
                # repair и reliability используем актуальные
                res_feat.append([ready, cur_load, compat, workloads[r] / 240.0, res_repair[r], res_reliability[r]])
            res_feat = np.array(res_feat).flatten()

            full_feat = np.concatenate([op_feat, res_feat])
            X.append(full_feat)

            # Обновляем загрузку выбранного ресурса
            workloads[best_r] += duration

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    np.savez(save_path, X=X, y=y)
    print(f"Датасет сохранён: {X.shape[0]} примеров, {X.shape[1]} признаков, {np.max(y)+1} классов.")
    return X, y

if __name__ == "__main__":
    simulate(5000, R=5)