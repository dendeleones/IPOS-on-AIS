# generate_bc_data.py
import numpy as np
import pickle
from datetime import datetime, timedelta
from datamodels import Order, Operation, Resource
from job_shop_env import JobShopEnv
from sklearn.neural_network import MLPClassifier
from optimizer_milp import build_milp_schedule
from benchmark_runner import parse_mk01, MK01_DATA, BEST_KNOWN_MAKESPAN

def generate_training_from_milp(orders, resources, schedule):
    """
    Проходит по расписанию шаг за шагом, собирая пары (state -> action).
    Возвращает список (state, action_mask, chosen_action_index).
    """
    env = JobShopEnv(orders, resources)
    state = env.reset()
    dataset = []
    # Сортируем операции по времени старта
    sorted_ops = sorted(schedule.items(), key=lambda kv: kv[1]['start'])
    # Для отслеживания запущенных операций
    op_to_idx = {op_info['op_id']: i for i, op_info in enumerate(env.get_ready_ops_list())} if False else None
    # Более простой подход: перебираем отсортированные операции и на каждом шаге
    # дожидаемся момента, когда операция становится ready (или сразу запускаем, если уже ready)
    for op_id, info in sorted_ops:
        # Дожидаемся, пока операция станет ready (может, она уже готова)
        while True:
            ready_ops = env.get_ready_ops_list()
            if any(op['op_id'] == op_id for op in ready_ops):
                break
            # Если не готова, продвигаем время до следующего события (освобождения ресурса)
            env._advance_until_resource_free()
        ready_ops = env.get_ready_ops_list()
        # Строим состояние и маску
        state = env._get_state()
        busy = np.array([1.0 if env.resource_remaining[rid] > 0 else 0.0 for rid in env.resource_ids])
        ext_state = np.concatenate([state, busy])
        op_feat_dim = 5
        action_dim = env.max_actions
        op_feat = np.zeros((action_dim, op_feat_dim))
        mask = np.zeros(action_dim, dtype=bool)
        chosen_idx = None
        for i, op_info in enumerate(ready_ops):
            if i >= action_dim:
                break
            rid = op_info['resource_id']
            # Заполняем признаки операции
            op_id_i = op_info['op_id']
            due_min = 0; rem_time = 0; weight = 0; progress = 0
            for order in env.orders:
                for op in order.ops:
                    if op.id == op_id_i:
                        due_min = (order.due_date - env.start_datetime).total_seconds() / 60.0
                        rem_time = env.op_remaining[op.id]
                        weight = order.priority_weight
                        completed = sum(1 for o in order.ops if env.op_status[o.id] == 'completed')
                        progress = completed / len(order.ops)
                        break
            slack = (due_min - env.current_time) / 1440.0
            rem_time_norm = rem_time / 1440.0
            weight_norm = weight / 10.0
            res_free = 0.0 if env.resource_remaining[rid] > 0 else 1.0
            op_feat[i] = [slack, rem_time_norm, weight_norm, progress, res_free]
            mask[i] = True
            if op_id_i == op_id:
                chosen_idx = i
        if chosen_idx is not None:
            dataset.append((ext_state, op_feat, mask, chosen_idx))
            # Теперь выполняем действие
            env.step(op_id)
    return dataset

def main():
    print("Парсинг данных MK01...")
    orders, resources = parse_mk01(MK01_DATA)
    print("Запуск MILP для получения оптимального расписания...")
    schedule = build_milp_schedule(orders, resources, datetime(2026, 4, 27, 8, 0, 0))
    if not schedule:
        print("MILP не нашёл решение.")
        return
    makespan = max(info['end'] for info in schedule.values()) - min(info['start'] for info in schedule.values())
    print(f"MILP makespan: {makespan.total_seconds()/60:.2f} мин")

    print("Генерация обучающих данных (состояние -> действие)...")
    dataset = generate_training_from_milp(orders, resources, schedule)
    print(f"Собрано {len(dataset)} примеров.")

    # Подготовка данных для обучения
    X = []
    y = []
    for (state, op_feat, mask, action) in dataset:
        # Объединяем состояние и признаки операций в один плоский вектор?
        # Проще обучить модель, которая принимает на вход конкатенацию state и op_feat.
        # Но размер op_feat фиксирован (max_actions * op_feat_dim). Используем паддинг.
        flat_op = op_feat.flatten()
        combined = np.concatenate([state, flat_op])
        X.append(combined)
        y.append(action)  # целевой индекс действия

    X = np.array(X)
    y = np.array(y)

    # Обучаем MLPClassifier (можно заменить на PointerNet, но для демонстрации подойдёт)
    model = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=1000, verbose=True)
    model.fit(X, y)

    # Сохраняем модель
    with open("bc_benchmark.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Модель BC сохранена как bc_benchmark.pkl")

if __name__ == "__main__":
    main()