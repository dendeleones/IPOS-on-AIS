# compare_models_extended.py
import os, sys, json
import numpy as np
import torch
from datetime import datetime, timedelta
import random
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from datamodels import Resource, Order, Operation
from database import get_operation_types
from training import SimpleDispatcher
from core import SimpleAgent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== НАСТРОЙКИ (как в последнем рабочем варианте) ====================
N_ORDERS = 120
SEED = 42

OPERATION_TYPES = {
    "фрезеровка": ["R1", "R2"],
    "сборка":     ["R3", "R4"],
    "сварка":     ["R4", "R5"],
}

RESOURCES = [
    ("R1", 1.0),
    ("R2", 0.9),
    ("R3", 0.7),
    ("R4", 0.8),
    ("R5", 0.3),
]
# ===================================================================================

def build_flat_state(order, op, resources, work_until, selected_date, allowed_resources):
    type_map = get_operation_types()
    types_list = list(OPERATION_TYPES.keys())
    type_to_idx = {name: i for i, name in enumerate(types_list)}

    type_onehot = np.zeros(3)
    if op.type_id and op.type_id in type_map:
        tname = type_map[op.type_id]
        if tname in type_to_idx:
            type_onehot[type_to_idx[tname]] = 1.0
    op_feat = np.array([
        *type_onehot,
        op.norm_duration / 120.0,
        order.priority_weight,
        (op.slack_hours if hasattr(op, 'slack_hours') else 0) / 48.0
    ])

    res_feats = []
    for res in resources:
        cur_load = (work_until[res.id] - selected_date).total_seconds() / 60.0 / 240.0
        compat = 1.0 if res in allowed_resources else 0.0
        res_vec = np.array([
            1.0 if res.status == 'Работает' else 0.0,
            cur_load,
            compat,
            res.load_minutes / 240.0,
            float(res.repair),
            res.reliability
        ])
        res_feats.append(res_vec)
    return np.concatenate([op_feat] + res_feats)


def simulate_plan(model_agent, resources, orders, selected_date, use_agent=True):
    work_until = {r.id: selected_date for r in resources}
    schedule = {}
    all_ops = []
    for order in orders:
        for op in order.ops:
            all_ops.append((order, op))
    all_ops.sort(key=lambda x: x[0].due_date)

    for order, op in all_ops:
        allowed_rids = OPERATION_TYPES.get(op.type_id, [r.id for r in resources])
        allowed_res = [r for r in resources if r.id in allowed_rids]
        if not allowed_res:
            allowed_res = resources

        if use_agent and model_agent is not None:
            full = build_flat_state(order, op, resources, work_until, selected_date, allowed_res)
            action = model_agent.select_action(full)
            chosen = resources[action]
            if chosen not in allowed_res:
                chosen = min(allowed_res, key=lambda r: (work_until[r.id] - selected_date).total_seconds() / max(r.reliability, 0.01))
        else:
            def effective_load(res):
                rel = res.reliability if res.reliability > 0 else 0.01
                if res.repair:
                    rel *= 0.5
                workload = (work_until[res.id] - selected_date).total_seconds() / 60.0
                return workload / rel
            chosen = min(allowed_res, key=effective_load)

        start = max(work_until[chosen.id], selected_date)
        dur = timedelta(minutes=op.norm_duration)
        end = start + dur
        schedule[op.id] = {'resource_id': chosen.id, 'start': start, 'end': end}
        work_until[chosen.id] = end
    return schedule


def collect_detailed_metrics(schedule, orders, resources, model_name):
    """Собирает расширенную статистику по заказам и ресурсам."""
    tardiness = []
    op_data = []  # список словарей для каждого заказа
    for order in orders:
        if not order.ops:
            continue
        op = order.ops[0]
        if op.id in schedule:
            start = schedule[op.id]['start']
            end = schedule[op.id]['end']
            tard = max(0.0, (end - order.due_date).total_seconds() / 3600.0)
            tardiness.append(tard)
            op_data.append({
                'order_id': order.id,
                'op_id': op.id,
                'type': op.type_id,
                'duration': op.norm_duration,
                'due': order.due_date.isoformat(),
                'start': start.isoformat(),
                'end': end.isoformat(),
                'resource': schedule[op.id]['resource_id'],
                'tardiness_h': tard
            })

    tardiness = np.array(tardiness) if tardiness else np.array([0])
    makespan = max(info['end'] for info in schedule.values())
    total_time = (makespan - min(info['start'] for info in schedule.values())).total_seconds() / 3600.0

    # Загрузка ресурсов
    res_load = {r.id: 0.0 for r in resources}
    for info in schedule.values():
        dur = (info['end'] - info['start']).total_seconds() / 3600.0
        res_load[info['resource_id']] += dur

    # Очередь во времени: считаем число ожидающих операций в каждый момент
    events = []
    for data in op_data:
        events.append((data['start'], 1))   # начало операции
        events.append((data['end'], -1))   # конец операции
    events.sort(key=lambda x: x[0])
    queue_length = []
    current = 0
    for t, delta in events:
        current += delta
        queue_length.append((t, current))

    return {
        'tardiness_array': tardiness.tolist(),
        'mean_tardiness': float(np.mean(tardiness)),
        'max_tardiness': float(np.max(tardiness)),
        'percent_late': float(sum(tardiness > 0) / len(tardiness) * 100) if len(tardiness) > 0 else 0,
        'makespan_hours': total_time,
        'resource_load': res_load,
        'op_data': op_data,
        'queue_length': queue_length
    }


def plot_combined_histogram(metrics_dict, folder):
    """Гистограммы опозданий на одном графике."""
    plt.figure(figsize=(12, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E']
    for (name, metrics), color in zip(metrics_dict.items(), colors):
        tard = np.array(metrics['tardiness_array'])
        plt.hist(tard, bins=30, alpha=0.5, label=name, color=color)
    plt.xlabel('Опоздание (часы)')
    plt.ylabel('Количество заказов')
    plt.title('Распределение опозданий по методам')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{folder}/histogram_tardiness.png")
    plt.close()


def plot_boxplot_tardiness(metrics_dict, folder):
    """Ящики с усами для опозданий."""
    plt.figure(figsize=(8, 6))
    data = [metrics['tardiness_array'] for metrics in metrics_dict.values()]
    plt.boxplot(data, tick_labels=list(metrics_dict.keys()))   # исправлено
    plt.ylabel('Опоздание (часы)')
    plt.title('Boxplot опозданий')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{folder}/boxplot_tardiness.png")
    plt.close()


def plot_resource_load_comparison(metrics_dict, resources, folder):
    """Горизонтальные столбцы загрузки ресурсов для каждого метода."""
    methods = list(metrics_dict.keys())
    res_ids = [r.id for r in resources]
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6), sharex=True)
    if n_methods == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        load = metrics_dict[method]['resource_load']
        values = [load.get(rid, 0) for rid in res_ids]
        ax.barh(res_ids, values, color='skyblue', edgecolor='black')
        ax.set_title(method)
        ax.set_xlabel('Часы работы')
        ax.set_xlim(0, max(values) * 1.2)
    plt.suptitle('Загрузка ресурсов по методам')
    plt.tight_layout()
    plt.savefig(f"{folder}/resource_load.png")
    plt.close()


def plot_resource_distribution_pie(metrics_dict, folder):
    """Круговые диаграммы распределения задач по ресурсам для каждого метода."""
    methods = list(metrics_dict.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 4))
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        op_data = metrics_dict[method]['op_data']
        res_count = defaultdict(int)
        for d in op_data:
            res_count[d['resource']] += 1
        labels = list(res_count.keys())
        sizes = list(res_count.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(method)
    plt.suptitle('Распределение задач по ресурсам')
    plt.tight_layout()
    plt.savefig(f"{folder}/resource_distribution.png")
    plt.close()


def plot_cumulative_completion(metrics_dict, folder):
    """Кумулятивное количество завершённых операций во времени."""
    plt.figure(figsize=(10, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E']
    for (name, metrics), color in zip(metrics_dict.items(), colors):
        op_data = metrics['op_data']
        end_times = sorted([datetime.fromisoformat(d['end']) for d in op_data])
        start_time = min(datetime.fromisoformat(d['start']) for d in op_data) if op_data else datetime.now()
        times = [(t - start_time).total_seconds() / 3600.0 for t in end_times]
        cumulative = np.arange(1, len(times) + 1)
        plt.step(times, cumulative, where='post', label=name, color=color)
    plt.xlabel('Время (часы)')
    plt.ylabel('Завершено операций')
    plt.title('Прогресс выполнения заказов')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{folder}/cumulative_completion.png")
    plt.close()


def plot_queue_length(metrics_dict, folder):
    """Динамика длины очереди (количество ожидающих операций) во времени."""
    plt.figure(figsize=(10, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E']
    for (name, metrics), color in zip(metrics_dict.items(), colors):
        queue = metrics['queue_length']
        if not queue:
            continue
        # Парсим ISO-даты в datetime
        parsed = [(datetime.fromisoformat(t), delta) for t, delta in queue]
        start_t = parsed[0][0]
        times = [(t - start_t).total_seconds() / 3600.0 for t, _ in parsed]
        lengths = [l for _, l in parsed]
        plt.step(times, lengths, where='post', label=name, color=color)
    plt.xlabel('Время (часы)')
    plt.ylabel('Число ожидающих операций')
    plt.title('Динамика очереди')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{folder}/queue_length.png")
    plt.close()


def plot_tardiness_by_type(metrics_dict, folder):
    """Среднее опоздание по типам операций."""
    plt.figure(figsize=(10, 6))
    methods = list(metrics_dict.keys())
    types = list(OPERATION_TYPES.keys())
    x = np.arange(len(types))
    width = 0.2
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E']
    for i, method in enumerate(methods):
        op_data = metrics_dict[method]['op_data']
        type_tard = defaultdict(list)
        for d in op_data:
            type_tard[d['type']].append(d['tardiness_h'])
        means = [np.mean(type_tard.get(t, [0])) for t in types]
        plt.bar(x + i * width, means, width, label=method, color=colors[i])
    plt.xlabel('Тип операции')
    plt.ylabel('Среднее опоздание (часы)')
    plt.title('Среднее опоздание по типам операций')
    plt.xticks(x + width * 1.5, types)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{folder}/tardiness_by_type.png")
    plt.close()


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Создание ресурсов
    resources = [Resource(rid, f"Станок {i+1}") for i, (rid, rel) in enumerate(RESOURCES)]
    for res, (_, rel) in zip(resources, RESOURCES):
        res.reliability = rel
        res.repair = 0
        res.status = 'Работает'

    selected_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Генерация заказов (жёсткий сценарий)
    orders = []
    type_names = list(OPERATION_TYPES.keys())
    for i in range(N_ORDERS):
        due_hours = random.randint(3, 12)
        due = selected_date + timedelta(hours=due_hours)
        order = Order(f'ORD_{i:03d}', due, random.uniform(0.5, 2.0))
        op_type = random.choice(type_names)
        op = Operation(f'{order.id}_op1', order.id, f'Деталь_{random.randint(1,5)}', 1,
                       random.uniform(40, 100))
        op.type_id = op_type
        op.slack_hours = random.uniform(0, 12)
        order.ops.append(op)
        orders.append(order)

    # Загрузка моделей
    models = {}
    for name, path in [("PointerNet", "pointer_net.pth"), ("SLIM", "slim_pointer.pth"), ("PPO", "ppo_dispatcher.pth")]:
        if os.path.exists(path):
            models[name] = SimpleAgent(path)
        else:
            models[name] = None

    # Запуск планирования и сбор метрик
    metrics_dict = {}
    for method, agent in models.items():
        sched = simulate_plan(agent, resources, orders, selected_date, use_agent=(agent is not None))
        metrics_dict[method] = collect_detailed_metrics(sched, orders, resources, method)

    sched_heur = simulate_plan(None, resources, orders, selected_date, use_agent=False)
    metrics_dict["Heuristic"] = collect_detailed_metrics(sched_heur, orders, resources, "Heuristic")

    # Создание папки для сохранения
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_dir = f"metrics/comparison/{timestamp}"
    os.makedirs(comp_dir, exist_ok=True)

    # Сохранение полного JSON с метриками
    with open(f"{comp_dir}/full_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    # Построение всех графиков
    plot_combined_histogram(metrics_dict, comp_dir)
    plot_boxplot_tardiness(metrics_dict, comp_dir)
    plot_resource_load_comparison(metrics_dict, resources, comp_dir)
    plot_resource_distribution_pie(metrics_dict, comp_dir)
    plot_cumulative_completion(metrics_dict, comp_dir)
    plot_queue_length(metrics_dict, comp_dir)
    plot_tardiness_by_type(metrics_dict, comp_dir)

    # Также строим старые bar charts (makespan, баланс)
    methods = list(metrics_dict.keys())
    makespans = [metrics_dict[m]['makespan_hours'] for m in methods]
    cv_loads = []
    for m in methods:
        loads = list(metrics_dict[m]['resource_load'].values())
        if loads and np.mean(loads) > 0:
            cv = np.std(loads) / np.mean(loads)
        else:
            cv = 0
        cv_loads.append(cv)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E']

    # Makespan bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, makespans, color=colors, edgecolor='black')
    plt.ylabel('Makespan (часы)')
    plt.title('Длительность расписания')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, val in zip(bars, makespans):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}',
                 ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{comp_dir}/makespan.png")
    plt.close()

    # Balance bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, cv_loads, color=colors, edgecolor='black')
    plt.ylabel('Коэффициент вариации загрузки')
    plt.title('Равномерность загрузки (меньше = лучше)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, val in zip(bars, cv_loads):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                 ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{comp_dir}/load_balance.png")
    plt.close()

    # Сводная таблица в CSV
    import csv
    with open(f"{comp_dir}/summary.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Метод', 'Среднее опоздание (ч)', 'Макс. опоздание (ч)', '% опоздавших',
                         'Makespan (ч)', 'Коэф. вариации загрузки'])
        for m in methods:
            met = metrics_dict[m]
            cv = cv_loads[methods.index(m)]
            writer.writerow([m, f"{met['mean_tardiness']:.2f}", f"{met['max_tardiness']:.2f}",
                             f"{met['percent_late']:.1f}", f"{met['makespan_hours']:.2f}",
                             f"{cv:.3f}"])

    print(f"Расширенное сравнение сохранено в {comp_dir}")
    print(f"Сгенерировано 10+ графиков, JSON и CSV")

if __name__ == "__main__":
    main()