# benchmark_runner.py
"""
Скрипт для тестирования MILP-решателя на классической задаче FJSP Brandimarte MK01.
Данные встроены, парсятся и передаются в build_milp_schedule с точными временами.
"""
from datetime import datetime, timedelta
from datamodels import Order, Operation, Resource
from optimizer_milp import build_milp_schedule

# ---------- Встроенные данные Brandimarte MK01 (формат SchedulingLab) ----------
MK01_DATA = """10 6
6 2 6 2 1 5 3 4 3 5 3 3 5 2 1 2 3 4 6 2 3 6 5 2 6 1 1 1 3 1 3 6 6 3 6 4 3
6 2 1 6 2 2 4 5 2 3 5 5 1 6 2 4 3 6 3 1 1 6 1 2 1 1 5 3 1
5 2 6 1 6 2 1 3 4 3 5 3 3 5 2 1 1 6 1 5 2 3 3 4 4
5 1 1 6 2 2 4 5 2 3 5 5 1 6 2 6 6 6 1 2 1 1 5 3 1
5 2 6 1 6 2 1 3 4 3 5 3 3 5 2 1 2 3 4 6 2 6 5 6 2 5 5 1
5 1 5 6 2 2 4 5 2 3 5 5 1 6 1 6 2 2 6 1 2 1 1 5 3 1
5 1 4 6 2 2 4 5 2 3 5 5 1 6 1 5 6 5 2 6 2 1 1 5 3 1
5 2 1 2 2 4 2 1 6 3 6 1 6 2 6 6 2 1 1 1 5 1 2 1 1 3 3
5 2 2 6 2 3 3 5 3 3 5 2 1 5 1 6 2 2 3 2 5 2 1 3 3 4
5 2 2 4 5 2 3 5 5 1 6 1 1 1 6 2 3 6 3 2 4 6 1 5 5 1 6
"""

BEST_KNOWN_MAKESPAN = 40

def parse_mk01(data_str):
    """Парсит встроенные данные Brandimarte MK01 и возвращает orders, resources."""
    lines = [l.strip() for l in data_str.strip().split('\n') if l.strip()]
    header = lines[0].split()
    n_jobs = int(header[0])
    n_machines = int(header[1])

    resources = [Resource(id=f'M{i}', name=f'Станок {i}') for i in range(n_machines)]
    orders = []
    current_time = datetime(2026, 4, 27, 8, 0, 0)

    for job_idx in range(n_jobs):
        parts = list(map(int, lines[1 + job_idx].split()))
        if not parts:
            continue
        n_ops = parts[0]
        order = Order(id=f'Job_{job_idx+1}', due_date=current_time + timedelta(days=30),
                      priority_weight=1.0, name=f'Job {job_idx+1}')
        idx = 1
        for op_num in range(1, n_ops + 1):
            if idx >= len(parts):
                break
            n_machines_op = parts[idx]
            idx += 1
            machine_options = []
            for _ in range(n_machines_op):
                if idx + 1 >= len(parts):
                    break
                machine_id = f'M{parts[idx]}'   # номер станка из бенчмарка
                proc_time = float(parts[idx + 1])
                machine_options.append((machine_id, proc_time))
                idx += 2
            # В качестве номинальной длительности возьмём минимальное время среди доступных станков
            min_duration = min(t for _, t in machine_options) if machine_options else 0
            op = Operation(
                id=f'Job_{job_idx+1}_op{op_num}',
                order_id=order.id,
                item=f'Op{op_num}',
                op_number=op_num,
                norm_duration=min_duration,   # будет переопределено в MILP
                resource_id=''
            )
            # Сохраняем опции для MILP
            op.machine_options = machine_options
            order.ops.append(op)
        orders.append(order)

    return orders, resources

def main():
    print("Парсинг встроенных данных Brandimarte MK01...")
    orders, resources = parse_mk01(MK01_DATA)
    print(f"Заказов: {len(orders)}, Ресурсов: {len(resources)}")
    print(f"Лучший известный makespan: {BEST_KNOWN_MAKESPAN}")

    schedule = build_milp_schedule(orders, resources, datetime(2026, 4, 27, 8, 0, 0))
    if schedule:
        all_ends = [info['end'] for info in schedule.values()]
        all_starts = [info['start'] for info in schedule.values()]
        makespan = max(all_ends) - min(all_starts)
        makespan_minutes = makespan.total_seconds() / 60.0
        print(f"MILP makespan: {makespan_minutes:.2f} мин")
        deviation = makespan_minutes - BEST_KNOWN_MAKESPAN
        print(f"Отклонение от эталона: {deviation:+.2f} мин")
    else:
        print("MILP не нашёл решение.")

if __name__ == "__main__":
    main()