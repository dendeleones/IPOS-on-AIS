# optimizer_milp.py
from ortools.sat.python import cp_model
from collections import defaultdict
from datetime import datetime, timedelta
from datamodels import Order, Resource
from database import get_resource_types, get_operation_types

def build_milp_schedule(orders, resources, current_time, fixed_ops=None):
    model = cp_model.CpModel()
    horizon = 30 * 24 * 60  # 30 дней в минутах

    op_starts = {}
    op_ends = {}
    intervals_per_resource = defaultdict(list)  # resource_id -> list of intervals (может быть опциональными)

    # Собираем информацию о типах и ресурсах
    type_to_resources = defaultdict(list)
    res_types = get_resource_types()
    for rid in res_types:
        for tid in res_types[rid]:
            type_to_resources[tid].append(rid)

    # Если у ресурса нет ни одного типа, он может выполнять любые операции (резерв)
    all_resource_ids = [r.id for r in resources]

    for order in orders:
        for op in order.ops:
            dur = int(round(op.norm_duration))
            s_var = model.NewIntVar(0, horizon, f'start_{op.id}')
            e_var = model.NewIntVar(0, horizon, f'end_{op.id}')
            op_starts[op.id] = s_var
            op_ends[op.id] = e_var

            if getattr(op, 'type_id', None):
                # Найти ресурсы, поддерживающие данный тип
                candidate_res = type_to_resources.get(op.type_id, [])
                if not candidate_res:
                    # Если нет специализированных, можно использовать любой ресурс
                    candidate_res = all_resource_ids[:]
                # Создаём опциональные интервалы
                presence_vars = []
                for rid in candidate_res:
                    pres = model.NewBoolVar(f'presence_{op.id}_{rid}')
                    interval = model.NewOptionalIntervalVar(s_var, dur, e_var, pres, f'interval_{op.id}_{rid}')
                    intervals_per_resource[rid].append(interval)
                    presence_vars.append(pres)
                # Ровно один ресурс должен быть выбран
                model.AddExactlyOne(presence_vars)
            else:
                # Закреплён за конкретным ресурсом (или ресурс не указан – ошибка, но поставим на первый)
                res_id = op.resource_id if op.resource_id else all_resource_ids[0]
                interval = model.NewIntervalVar(s_var, dur, e_var, f'interval_{op.id}')
                intervals_per_resource[res_id].append(interval)

    # Ограничения: на каждом ресурсе интервалы не пересекаются (опциональные автоматически учитываются)
    for rid, intervals in intervals_per_resource.items():
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Технологическая последовательность
    for order in orders:
        for op in order.ops:
            for pred_id in op.predecessors:
                if pred_id in op_starts and op.id in op_starts:
                    model.Add(op_starts[op.id] >= op_ends[pred_id])

    # Фиксированные операции
    if fixed_ops:
        for op_id, fix in fixed_ops.items():
            if op_id in op_starts:
                start_m = int((fix['start'] - current_time).total_seconds() / 60)
                end_m = int((fix['end'] - current_time).total_seconds() / 60)
                model.Add(op_starts[op_id] == start_m)
                model.Add(op_ends[op_id] == end_m)
                # Также нужно зафиксировать выбор ресурса для операции с типом, но это сложнее.
                # Пока проигнорируем, предполагая, что фиксированные операции уже имеют конкретный resource_id.

    # Целевая функция: минимизация взвешенного запаздывания
    tardiness_vars = []
    for order in orders:
        last_op = order.ops[-1]
        if last_op.id not in op_starts:
            continue
        due_minutes = int((order.due_date - current_time).total_seconds() / 60)
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
                if op.id not in op_starts:
                    continue
                start = solver.Value(op_starts[op.id])
                end = solver.Value(op_ends[op.id])
                # Определяем выбранный ресурс
                res_id = op.resource_id  # по умолчанию
                if getattr(op, 'type_id', None):
                    candidate_res = type_to_resources.get(op.type_id, all_resource_ids)
                    for rid in candidate_res:
                        # Проверим, активен ли опциональный интервал
                        interval_var = next((iv for iv in intervals_per_resource[rid] if iv.Name() == f'interval_{op.id}_{rid}'), None)
                        if interval_var and solver.BooleanValue(interval_var.PresenceLit()):
                            res_id = rid
                            break
                schedule[op.id] = {
                    'start': current_time + timedelta(minutes=start),
                    'end': current_time + timedelta(minutes=end),
                    'resource_id': res_id
                }
        return schedule
    else:
        return None