# optimizer_milp.py
from ortools.sat.python import cp_model
from collections import defaultdict
from datetime import datetime, timedelta
from datamodels import Order, Resource
from database import get_resource_operation_types, get_operation_types

def build_milp_schedule(orders, resources, current_time, fixed_ops=None):
    model = cp_model.CpModel()
    horizon = 60 * 24 * 60  # 60 дней в минутах

    op_starts = {}
    op_ends = {}
    intervals_per_resource = defaultdict(list)
    presence_map = {}

    type_to_resources = defaultdict(list)
    res_types = get_resource_operation_types()
    for rid in res_types:
        for tid in res_types[rid]:
            type_to_resources[tid].append(rid)

    all_resource_ids = [r.id for r in resources]
    # Ресурсы без указанных типов считаются универсальными
    universal_resources = [rid for rid in all_resource_ids if rid not in res_types]

    for order in orders:
        for op in order.ops:
            dur = int(round(op.norm_duration))
            s_var = model.NewIntVar(0, horizon, f'start_{op.id}')
            e_var = model.NewIntVar(0, horizon, f'end_{op.id}')
            op_starts[op.id] = s_var
            op_ends[op.id] = e_var

            # Если у операции задан список конкретных станков с временами
            if hasattr(op, 'machine_options') and op.machine_options:
                presence_vars = []
                for rid, p_time in op.machine_options:
                    pres = model.NewBoolVar(f'presence_{op.id}_{rid}')
                    interval = model.NewOptionalIntervalVar(s_var, int(round(p_time)), e_var, pres,
                                                            f'interval_{op.id}_{rid}')
                    intervals_per_resource[rid].append(interval)
                    presence_vars.append(pres)
                    presence_map[(op.id, rid)] = pres
                model.AddExactlyOne(presence_vars)
            else:
                # Старая логика на основе type_id или всех ресурсов
                if getattr(op, 'type_id', None) and op.type_id:
                    candidate_res = type_to_resources.get(op.type_id, [])
                    candidate_res.extend(universal_resources)
                    if not candidate_res:
                        candidate_res = all_resource_ids[:]
                    candidate_res = list(set(candidate_res))
                else:
                    candidate_res = all_resource_ids[:]

                presence_vars = []
                for rid in candidate_res:
                    pres = model.NewBoolVar(f'presence_{op.id}_{rid}')
                    interval = model.NewOptionalIntervalVar(s_var, dur, e_var, pres, f'interval_{op.id}_{rid}')
                    intervals_per_resource[rid].append(interval)
                    presence_vars.append(pres)
                    presence_map[(op.id, rid)] = pres
                model.AddExactlyOne(presence_vars)

    # Ограничения непересечения на каждом ресурсе
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
                if fix.get('resource_id'):
                    target_rid = fix['resource_id']
                    for (oid, rid), pres in presence_map.items():
                        if oid == op_id and rid != target_rid:
                            model.Add(pres == 0)
                    for (oid, rid), pres in presence_map.items():
                        if oid == op_id and rid == target_rid:
                            model.Add(pres == 1)
                            break

    # Целевая функция
    tardiness_vars = []
    for order in orders:
        if not order.ops:
            continue
        last_op = order.ops[-1]
        if last_op.id not in op_starts:
            continue
        due_minutes = int((order.due_date - current_time).total_seconds() / 60)
        if due_minutes < 0:
            due_minutes = 0
        tard = model.NewIntVar(0, horizon, f'tard_{order.id}')
        model.AddMaxEquality(tard, [0, op_ends[last_op.id] - due_minutes])
        weight = int(order.priority_weight * 100)
        tardiness_vars.append(tard * weight)
    if tardiness_vars:
        model.Minimize(sum(tardiness_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = {}
        for order in orders:
            for op in order.ops:
                if op.id not in op_starts:
                    continue
                start = solver.Value(op_starts[op.id])
                end = solver.Value(op_ends[op.id])
                res_id = all_resource_ids[0]  # резерв
                for (oid, rid), pres in presence_map.items():
                    if oid == op.id and solver.Value(pres) == 1:
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