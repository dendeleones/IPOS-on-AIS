# core.py
import os, json, threading, pickle
import numpy as np
from datetime import datetime, timedelta
from datamodels import Order, Operation, Resource
from predictor import DurationPredictor
from optimizer_milp import build_milp_schedule
from aps_core import HybridAPS
from database import (
    init_db, save_orders_to_db, load_orders_from_db,
    update_schedule, get_fixed_operations, save_resources, load_resources,
    save_downtime, load_downtimes, delete_resource, delete_downtime,
    get_state_types, add_state_type, delete_state_type,
    get_planned_operations, apply_accepted_changes,
    get_operation_types, add_operation_type, delete_operation_type,
    set_resource_operation_types, get_resource_operation_types,
    add_task_to_queue, update_task_status, get_tasks_for_resource, delete_task_from_queue,
    delete_order_from_db, delete_operation_from_db,
    get_setting, set_setting, get_all_settings,
    log_resource_state, log_operation_state, get_execution_data
)
from PySide6.QtCore import QObject, Signal

class BCAgent:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    def select_action(self, state, op_feat=None, mask=None):
        probs = self.model.predict_proba([state])[0]
        probs[~np.array(mask)] = -1
        return np.argmax(probs)

class APSCore(QObject):
    dashboard_changed = Signal()
    orders_changed = Signal()
    message_signal = Signal(str, str)

    def __init__(self):
        super().__init__()
        init_db()
        self.resources = load_resources()
        if not self.resources:
            self.resources = [Resource('R1', 'Станок 1'), Resource('R2', 'Станок 2'), Resource('R3', 'Станок 3')]
            save_resources(self.resources)
        self.resource_map = {r.id: r for r in self.resources}
        self.predictor = DurationPredictor()
        self.aps_engine = HybridAPS(self.resources, predictor=self.predictor)
        self.rl_agent = None
        self.use_rl = False
        self.orders = []
        self.current_schedule = {}
        self.approved_schedule = {}
        self.downtimes = load_downtimes()
        self.state_types = get_state_types()
        self.selected_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        self.loaded_model_name = ""   # <-- новое поле

        self.settings = get_all_settings()
        if not self.settings:
            self.settings = {
                'bc_model_path': 'behavior_cloning.pkl',
                'dqn_episodes': '500', 'dqn_lr': '0.0003',
                'erp_file': 'erp_orders.json', 'mes_file': 'mes_events.json',
                'gantt_slot_minutes': '120'
            }
            for k, v in self.settings.items():
                set_setting(k, v)

        saved_orders = load_orders_from_db(self.resources)
        if saved_orders:
            self.orders = saved_orders
            self.aps_engine.load_orders(self.orders)
        self.load_schedules_from_db()
        if not self.current_schedule and not self.approved_schedule and self.orders:
            self.run_milp()

    def get_loaded_model_info(self):
        return self.loaded_model_name if self.loaded_model_name else "Нет"

    def load_settings(self):
        self.settings = get_all_settings()

    def save_settings(self, settings_dict):
        for k, v in settings_dict.items():
            set_setting(k, str(v))
        self.settings = get_all_settings()

    def load_schedules_from_db(self):
        fixed = get_fixed_operations()
        self.approved_schedule = {op_id: {'start': v['start'], 'end': v['end'], 'resource_id': v['resource_id']}
                                  for op_id, v in fixed.items()}
        planned = get_planned_operations()
        self.current_schedule = {op_id: {'start': v['start'], 'end': v['end'], 'resource_id': v['resource_id']}
                                 for op_id, v in planned.items() if v['start'] is not None}

    def create_order(self, due_date, priority, name, order_number, operations):
        if order_number:
            order_id = f"Заказ_{order_number}"
        else:
            order_id = f"Заказ_{np.random.randint(100, 999)}"
        order = Order(order_id, due_date, priority, name=name)
        for i, op_data in enumerate(operations, start=1):
            op = Operation(
                f"{order_id}_оп{i}", order_id, op_data.get('item', ''),
                i, op_data.get('duration', 60),
                type_id=op_data.get('type_id')
            )
            if 'quantity' in op_data:
                op.quantity = op_data['quantity']
            if 'predecessors' in op_data:
                op.predecessors = op_data['predecessors']
            order.ops.append(op)
        self.orders.append(order)
        save_orders_to_db([order])
        self.aps_engine.load_orders([order])
        # self.run_incremental_milp()   - обучение милп
        self.orders_changed.emit()
        self.dashboard_changed.emit()
        return order

    def delete_order(self, order_id):
        order = next((o for o in self.orders if o.id == order_id), None)
        if not order:
            return
        for op in order.ops:
            self._remove_from_schedules(op.id)
            delete_task_from_queue(op.id)
        self.orders.remove(order)
        delete_order_from_db(order_id)
        self.orders_changed.emit()
        self.dashboard_changed.emit()

    def delete_completed_task(self, op_id):
        self._remove_from_schedules(op_id)
        delete_task_from_queue(op_id)
        for order in self.orders:
            for op in order.ops:
                if op.id == op_id:
                    order.ops.remove(op)
                    delete_operation_from_db(op_id)
                    self.dashboard_changed.emit()
                    return

    def generate_random_orders(self, count=10):
        # Защита от некорректного count
        if not isinstance(count, int) or count <= 0:
            count = 10

        types = list(get_operation_types().keys())
        if not types:
            self.message_signal.emit("warning", "Сначала добавьте типы операций на вкладке «Оборудование → Типы».")
            return

        base = datetime(2026, 4, 27, 8, 0, 0)
        generated = []
        for _ in range(count):
            due = base + timedelta(hours=np.random.randint(10, 100))
            order = Order(
                f"Заказ_{np.random.randint(1000, 9999)}",
                due,
                round(np.random.uniform(0.5, 2.0), 2),
                name="Случайный"
            )
            prev = None
            for j in range(1, np.random.randint(2, 4) + 1):
                op_id = f"{order.id}_оп{j}"
                type_id = np.random.choice(types)
                op = Operation(
                    op_id, order.id, f"Деталь_{np.random.randint(1, 5)}", j,
                    round(np.random.uniform(20, 120), 1)
                )
                op.type_id = type_id
                op.quantity = np.random.randint(1, 10)
                if prev:
                    op.predecessors = [prev]
                order.ops.append(op)
                prev = op_id
            generated.append(order)

        self.orders.extend(generated)
        save_orders_to_db(generated)
        for o in generated:
            self.aps_engine.load_orders([o])

        # self.run_incremental_milp() - если нужно закинуть обучение после генерации
        self.orders_changed.emit()
        self.dashboard_changed.emit()
        self.message_signal.emit("info", f"Создано {count} случайных заказов. Перейдите на вкладку «Дашборд».")

    def load_erp_orders(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            erp_data = json.load(f)
        loaded = []
        for od in erp_data:
            order = Order(od['id'], datetime.fromisoformat(od['due_date']), od.get('priority_weight', 1.0), name=od.get('name', ''))
            for op_d in od['operations']:
                op = Operation(op_d['id'], order.id, op_d['item'], op_d['op_number'],
                               op_d['norm_duration'])
                op.predecessors = op_d.get('predecessors', [])
                op.type_id = op_d.get('type_id', '')
                op.quantity = op_d.get('quantity', 1)
                order.ops.append(op)
            loaded.append(order)
        self.orders.extend(loaded)
        save_orders_to_db(loaded)
        for o in loaded:
            self.aps_engine.load_orders([o])
        self.run_incremental_milp()
        self.orders_changed.emit()
        self.dashboard_changed.emit()

    def run_milp(self):
        if not self.orders:
            return
        if not any(order.ops for order in self.orders):
            return
        self.aps_engine.run_milp()
        self.current_schedule = self.aps_engine.current_schedule
        if self.current_schedule:
            min_start = min(info['start'] for info in self.current_schedule.values())
            self.selected_date = min_start.replace(hour=0, minute=0, second=0, microsecond=0)
            self._sync_schedule_to_task_queue()
        self.dashboard_changed.emit()

    def _sync_schedule_to_task_queue(self):
        for op_id, info in self.current_schedule.items():
            if info['resource_id'] not in self.resource_map:
                continue
            delete_task_from_queue(op_id)
            add_task_to_queue(info['resource_id'], op_id)
        # Больше не активируем задачи автоматически – они остаются в статусе 'pending'

    def run_fixed_milp(self):
        self.aps_engine.run_milp_with_fixed()
        self.current_schedule = self.aps_engine.current_schedule
        self._sync_schedule_to_task_queue()
        self.dashboard_changed.emit()

    def run_incremental_milp(self):
        self.aps_engine.run_milp_incremental()
        self.current_schedule = self.aps_engine.current_schedule
        self._sync_schedule_to_task_queue()
        self.dashboard_changed.emit()

    def approve_plan(self):
        if not self.current_schedule:
            self.message_signal.emit("info", "Нет рабочего плана для утверждения.")
            return
        for op_id, info in self.current_schedule.items():
            update_schedule(op_id, info['resource_id'], info['start'], info['end'], fixed=1, status='fixed')
            self.approved_schedule[op_id] = info
        self.current_schedule.clear()
        self.dashboard_changed.emit()
        self.message_signal.emit("info", "План утверждён.")

    def move_task_to_queue(self, op_id, resource_id):
        # Удаляем операцию из старых планов и очередей
        self._remove_from_schedules(op_id)
        delete_task_from_queue(op_id)
        add_task_to_queue(resource_id, op_id)
        log_operation_state(op_id, 'pending', resource_id)
        tasks = get_tasks_for_resource(resource_id)
        if not any(t['status'] == 'active' for t in tasks):
            pending = [t for t in tasks if t['status'] == 'pending']
            if pending:
                self.move_task_to_production(pending[0]['operation_id'], resource_id)
                return
        self.dashboard_changed.emit()
        self.message_signal.emit("info", f"Заказ {op_id} помещён в очередь {resource_id}")

    def move_task_to_production(self, op_id, resource_id):
        tasks = get_tasks_for_resource(resource_id)
        for t in tasks:
            if t['status'] == 'active':
                update_task_status(t['operation_id'], 'completed')
                log_operation_state(t['operation_id'], 'completed')
        update_task_status(op_id, 'active')
        log_operation_state(op_id, 'active', resource_id)
        self.dashboard_changed.emit()
        self.message_signal.emit("info", f"Заказ {op_id} запущен в производство на {resource_id}")

    def complete_task(self, op_id):
        update_task_status(op_id, 'completed')
        log_operation_state(op_id, 'completed')
        for res in self.resources:
            tasks = get_tasks_for_resource(res.id)
            for t in tasks:
                if t['operation_id'] == op_id:
                    pending = [t for t in tasks if t['status'] == 'pending']
                    if pending:
                        next_op = pending[0]['operation_id']
                        self.move_task_to_production(next_op, res.id)
                    else:
                        self.dashboard_changed.emit()
                    return
        self.dashboard_changed.emit()

    def get_completed_tasks(self):
        completed = []
        for r in self.resources:
            tasks = get_tasks_for_resource(r.id)
            for t in tasks:
                if t['status'] == 'completed':
                    op = self._get_operation_by_id(t['operation_id'])
                    if op:
                        order = self._get_order_by_op(t['operation_id'])
                        completed.append({
                            'operation_id': t['operation_id'],
                            'order_id': order.id if order else '',
                            'item': op.item,
                            'resource_id': r.id,
                            'resource_name': r.name
                        })
        return completed

    def add_resource(self, rid, name):
        if rid in self.resource_map:
            self.message_signal.emit("error", "Ресурс уже существует.")
            return
        r = Resource(rid, name)
        self.resources.append(r)
        self.resource_map[rid] = r
        save_resources(self.resources)
        self.aps_engine.resources[rid] = r
        self.dashboard_changed.emit()

    def delete_resource(self, rid):
        for order in self.orders:
            for op in order.ops:
                if op.resource_id == rid:
                    self.message_signal.emit("error", "На ресурсе есть операции.")
                    return
        self.resources = [r for r in self.resources if r.id != rid]
        del self.resource_map[rid]
        delete_resource(rid)
        save_resources(self.resources)
        self.dashboard_changed.emit()

    def add_operation_type_core(self, name):
        tid = add_operation_type(name)
        return tid

    def delete_operation_type_core(self, tid):
        delete_operation_type(tid)

    def set_resource_types_core(self, rid, type_ids):
        set_resource_operation_types(rid, type_ids)

    def add_state_type_core(self, name, color):
        sid = add_state_type(name, color)
        self.state_types[sid] = (name, color)
        return sid

    def delete_state_type_core(self, sid):
        delete_state_type(sid)
        if sid in self.state_types:
            del self.state_types[sid]

    def add_downtime_core(self, rid, start, end, state_id):
        self.downtimes.setdefault(rid, []).append((start, end, state_id))
        save_downtime(rid, start, end, state_id)

    def clear_downtimes_core(self, rid):
        if rid in self.downtimes:
            for item in self.downtimes[rid]:
                delete_downtime(rid, item[0], item[1])
            del self.downtimes[rid]

    def train_predictor(self):
        try:
            self.aps_engine.train_predictor()
            self.message_signal.emit("info", "Предсказатель обновлён.")
        except Exception as e:
            self.message_signal.emit("error", str(e))

    def _get_op_duration(self, op_id):
        for order in self.orders:
            for op in order.ops:
                if op.id == op_id:
                    return timedelta(minutes=op.norm_duration)
        return None

    def _remove_from_schedules(self, op_id):
        self.current_schedule.pop(op_id, None)
        self.approved_schedule.pop(op_id, None)

    def _get_operation_by_id(self, op_id):
        for order in self.orders:
            for op in order.ops:
                if op.id == op_id:
                    return op
        return None

    def _get_order_by_op(self, op_id):
        for order in self.orders:
            for op in order.ops:
                if op.id == op_id:
                    return order
        return None