# optimizer_milp.py
from ortools.sat.python import cp_model
from collections import defaultdict
from datetime import datetime, timedelta
from datamodels import Order, Resource

def build_milp_schedule(orders, resources, current_time, fixed_ops=None):
    """
    orders: list of Order
    resources: list of Resource
    current_time: datetime, точка отсчёта
    fixed_ops: dict operation_id -> {'resource_id': str, 'start': datetime, 'end': datetime}
               операции с жёстко заданными назначением и временем

    return: dict operation_id -> {'start': datetime, 'end': datetime, 'resource_id': str}
            или None, если решение не найдено
    """
    model = cp_model.CpModel()
    horizon = 30 * 24 * 60  # 30 дней в минутах

    # Переменные и интервалы
    op_starts = {}
    op_ends = {}
    op_durations = {}
    intervals_per_resource = defaultdict(list)

    # Ресурсы для быстрого доступа
    resource_dict = {r.id: r for r in resources}

    # Сначала создаём переменные для всех операций
    for order in orders:
        for op in order.ops:
            dur = int(round(op.norm_duration))
            s_var = model.NewIntVar(0, horizon, f'start_{op.id}')
            e_var = model.NewIntVar(0, horizon, f'end_{op.id}')
            op_starts[op.id] = s_var
            op_ends[op.id] = e_var
            op_durations[op.id] = dur

    # Если есть фиксированные операции, добавляем ограничения
    if fixed_ops:
        for op_id, fix in fixed_ops.items():
            # Ограничение по ресурсу: интервал должен быть на указанном ресурсе
            # Мы не можем напрямую ограничить ресурс через переменную, потому что операции уже
            # привязаны к ресурсу через op.resource_id. Если фиксированный ресурс отличается,
            # нужно создать интервал на целевом ресурсе, но это усложняет модель.
            # Для простоты предполагаем, что фиксированная операция остаётся на том же ресурсе,
            # либо мы меняем ресурс в самом заказе перед вызовом MILP.
            # Здесь мы реализуем только временные ограничения, а изменение ресурса
            # будет производиться до вызова MILP путём обновления op.resource_id.
            # Фиксируем время
            start_minutes = int((fix['start'] - current_time).total_seconds() / 60)
            end_minutes = int((fix['end'] - current_time).total_seconds() / 60)
            model.Add(op_starts[op_id] == start_minutes)
            model.Add(op_ends[op_id] == end_minutes)

    # Теперь создаём интервалы с учётом возможного изменения ресурса
    # (если фиксированная операция перенесена на другой ресурс, op.resource_id уже должен быть изменён)
    for order in orders:
        for op in order.ops:
            dur = op_durations[op.id]
            s = op_starts[op.id]
            e = op_ends[op.id]
            interval = model.NewIntervalVar(s, dur, e, f'interval_{op.id}')
            intervals_per_resource[op.resource_id].append(interval)

    # Ограничения: на каждом ресурсе интервалы не пересекаются
    for rid, intervals in intervals_per_resource.items():
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Технологическая последовательность
    for order in orders:
        for op in order.ops:
            for pred_id in op.predecessors:
                model.Add(op_starts[op.id] >= op_ends[pred_id])

    # Целевая функция: минимизация взвешенного запаздывания
    tardiness_vars = []
    for order in orders:
        last_op = order.ops[-1]
        due_minutes = int((order.due_date - current_time).total_seconds() / 60)
        due_var = model.NewConstant(due_minutes)
        tard = model.NewIntVar(0, horizon, f'tard_{order.id}')
        model.AddMaxEquality(tard, [0, op_ends[last_op.id] - due_minutes])
        weight = int(order.priority_weight * 100)
        tardiness_vars.append(tard * weight)

    model.Minimize(sum(tardiness_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = {}
        for order in orders:
            for op in order.ops:
                start = solver.Value(op_starts[op.id])
                end = solver.Value(op_ends[op.id])
                schedule[op.id] = {
                    'start': current_time + timedelta(minutes=start),
                    'end': current_time + timedelta(minutes=end),
                    'resource_id': op.resource_id  # ресурс может быть изменён до вызова
                }
        return schedule
    else:
        return None