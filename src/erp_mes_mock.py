# erp_mes_mock.py
import json
import random
from datetime import datetime, timedelta

def generate(erp_file, mes_file, resources):
    # Ресурсы для случайного выбора
    res_ids = [r.id for r in resources]
    items = [f"Деталь_{i}" for i in range(1,6)]

    # --- ERP: заказы ---
    orders = []
    base_time = datetime(2026, 4, 27, 8, 0, 0)
    for i in range(1, 8):
        due = base_time + timedelta(hours=random.randint(10, 80))
        order = {
            "id": f"Заказ_ERP_{i:03d}",
            "due_date": due.isoformat(),
            "priority_weight": round(random.uniform(0.5, 2.0), 2),
            "operations": []
        }
        n_ops = random.randint(2, 3)
        prev = None
        for j in range(1, n_ops+1):
            op_id = f"{order['id']}_оп{j}"
            res = random.choice(res_ids)
            dur = round(random.uniform(20, 120), 1)
            preds = [prev] if prev else []
            order["operations"].append({
                "id": op_id,
                "item": random.choice(items),
                "op_number": j,
                "resource_id": res,
                "norm_duration": dur,
                "predecessors": preds
            })
            prev = op_id
        orders.append(order)

    with open(erp_file, 'w', encoding='utf-8') as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)

    # --- MES: события ---
    events = []
    # Допустим, первая операция началась через 10 минут после старта
    current = base_time + timedelta(minutes=10)
    for order in orders[:3]:  # только первые 3 заказа
        for op in order['operations']:
            start = current
            dur = timedelta(minutes=op['norm_duration'])
            end = start + dur
            events.append({
                "timestamp": start.isoformat(),
                "resource_id": op['resource_id'],
                "operation_id": op['id'],
                "event_type": "start"
            })
            events.append({
                "timestamp": end.isoformat(),
                "resource_id": op['resource_id'],
                "operation_id": op['id'],
                "event_type": "complete"
            })
            current = end + timedelta(minutes=random.randint(5, 15))

    with open(mes_file, 'w', encoding='utf-8') as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    from datamodels import Resource
    resources = [Resource(id='R1', name='Станок 1'), Resource(id='R2', name='Станок 2'), Resource(id='R3', name='Станок 3')]
    generate('erp_orders.json', 'mes_events.json', resources)
    print("Тестовые ERP и MES файлы созданы.")