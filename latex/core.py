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
    log_resource_state, log_operation_state, get_execution_data, set_order_status
)
from PySide6.QtCore import QObject, Signal
from training import ResourceOpPointerNet, SimpleDispatcher
import torch

class BCAgent:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    def select_action(self, state, op_feat=None, mask=None):
        probs = self.model.predict_proba([state])[0]
        probs[~np.array(mask)] = -1
        return np.argmax(probs)

class PointerNetAgent:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Размерности, использованные при обучении
        self.d_res = 7
        self.d_op = 6
        self.model = ResourceOpPointerNet(d_res=self.d_res, d_op=self.d_op).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def select_action(self, res_feat, op_feat, mask):
        """
        res_feat: (R, d_res)
        op_feat:  (O, d_op)
        mask:     (O, R) bool
        Возвращает индекс выбранной операции (0..O-1).
        """
        with torch.no_grad():
            r = torch.tensor(res_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            o = torch.tensor(op_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            probs = self.model(r, o, m)          # (1, O, R)
            # Выбираем операцию с наибольшей вероятностью хотя бы на одном ресурсе
            values, _ = probs.max(dim=-1)       # (1, O)
            action = values.argmax(dim=-1).item()
            return action, None

class SimpleAgent:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        state = torch.load(model_path, map_location=self.device)

        # Автоматически находим первый и последний Linear слои среди ключей state_dict
        linear_weights = [k for k in state.keys() if k.endswith('.weight') and 'net.' in k]
        if not linear_weights:
            raise RuntimeError("В загруженной модели не найдено полносвязных слоёв (net.*.weight)")

        first_weight_key = sorted(linear_weights)[0]   # например, 'net.0.weight'
        last_weight_key  = sorted(linear_weights)[-1]  # например, 'net.6.weight'

        input_dim = state[first_weight_key].shape[1]   # количество входных признаков
        output_dim = state[last_weight_key].shape[0]   # количество классов (ресурсов)

        # Создаём модель с теми же размерами
        self.model = SimpleDispatcher(input_dim, output_dim).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def select_action(self, X):
        """X – плоский вектор признаков (36,) → возвращает индекс ресурса."""
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x)          # (1, output_dim)
            return logits.argmax(dim=1).item()

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

        # Попытка загрузить RL-модель, если путь сохранён в настройках
        self.load_rl_model_if_available()

        saved_orders = load_orders_from_db(self.resources)
        if saved_orders:
            self.orders = saved_orders
            # Приведение к "1 заказ – 1 операция" для уже существующих в БД записей
            for order in self.orders:
                if len(order.ops) > 1:
                    # Удаляем лишние операции из БД
                    for op in order.ops[1:]:
                        delete_operation_from_db(op.id)
                    order.ops = order.ops[:1]
                    # Пересохраняем заказ с одной операцией
                    save_orders_to_db([order])
            # Удаляем из очереди все операции, которых нет среди активных заказов
            valid_op_ids = set()
            for order in self.orders:
                for op in order.ops:
                    valid_op_ids.add(op.id)
            import sqlite3
            conn = sqlite3.connect("aps.db")
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT operation_id FROM task_queue")
            for row in cur.fetchall():
                if row[0] not in valid_op_ids:
                    cur.execute("DELETE FROM task_queue WHERE operation_id=?", (row[0],))
            conn.commit()
            conn.close()
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

    def _build_pointer_input(self, env, ready_ops):
        type_map = get_operation_types()  # id → имя
        type_list = ['фрезеровка', 'сборка', 'сварка']  # порядок, использованный в датасете
        type_to_idx = {name: i for i, name in enumerate(type_list)}
        res_types = get_resource_operation_types()  # rid → [type_id]

        R = len(env.resource_ids)
        d_res = 7
        res_feat = np.zeros((R, d_res), dtype=np.float32)
        for i, rid in enumerate(env.resource_ids):
            res = self.resource_map[rid]
            status_active = 1.0 if res.status == 'Работает' else 0.0
            load_norm = res.load_minutes / 240.0
            work_hours = float(res.work_hours)
            type_onehot = np.zeros(3)
            for tid in res_types.get(rid, []):
                tname = type_map.get(tid, '')
                if tname in type_to_idx:
                    type_onehot[type_to_idx[tname]] = 1.0
            repair = float(res.repair)
            res_feat[i, 0] = status_active
            res_feat[i, 1] = load_norm
            res_feat[i, 2] = work_hours
            res_feat[i, 3:6] = type_onehot
            res_feat[i, 6] = repair

        O = len(ready_ops)
        d_op = 6
        op_feat = np.zeros((O, d_op), dtype=np.float32)
        mask = np.zeros((O, R), dtype=bool)
        for idx, op_info in enumerate(ready_ops):
            op_id = op_info['op_id']
            op = self._get_operation_by_id(op_id)
            type_onehot = np.zeros(3)
            if op and op.type_id:
                tname = type_map.get(op.type_id, '')
                if tname in type_to_idx:
                    type_onehot[type_to_idx[tname]] = 1.0
            op_feat[idx, :3] = type_onehot
            op_feat[idx, 3] = op.norm_duration / 120.0 if op else 0.5
            op_feat[idx, 4] = op.urgency if op else 1.0
            op_feat[idx, 5] = (op.slack_hours if op else 0) / 48.0

            # Допустимый ресурс из ready_ops
            allowed_rid = op_info['resource_id']
            if allowed_rid in env.resource_ids:
                mask[idx, env.resource_ids.index(allowed_rid)] = True
            else:
                mask[idx, :] = True

        return res_feat, op_feat, mask

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
            order_id = f"Заказ_{np.random.randint(1, 99)}"
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
            # Единственная операция
            op_id = f"{order.id}_оп1"
            type_id = np.random.choice(types)
            op = Operation(
                op_id, order.id, f"Деталь_{np.random.randint(1, 5)}", 1,
                round(np.random.uniform(20, 120), 1)
            )
            op.type_id = type_id
            op.quantity = int(np.random.randint(1, 10))
            order.ops.append(op)
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
            order = Order(...)
            # Берём только первую операцию
            op_d = od['operations'][0] if od.get('operations') else {}
            if op_d:
                op = Operation(op_d['id'], order.id, op_d['item'], 1, op_d['norm_duration'])
                op.predecessors = []  # больше нет зависимостей
                op.type_id = op_d.get('type_id', '')
                op.quantity = int(op_d.get('quantity', 1))
                order.ops.append(op)
            loaded.append(order)
        self.orders.extend(loaded)
        save_orders_to_db(loaded)
        for o in loaded:
            self.aps_engine.load_orders([o])
        self.run_incremental_milp()
        self.orders_changed.emit()
        self.dashboard_changed.emit()

    def _build_heuristic_schedule(self, only_unscheduled=True):
        """
        Быстрый эвристический планировщик: EDD + Least Workload.
        Возвращает словарь current_schedule.
        """
        from datetime import datetime
        work_until = {r.id: datetime.now() for r in self.resources}
        schedule = {}
        ops = []
        for order in self.orders:
            for op in order.ops:
                if only_unscheduled and (op.id in self.current_schedule or op.id in self.approved_schedule):
                    continue
                ops.append((order, op))

        # Сортируем по дате сдачи (Earliest Due Date)
        ops.sort(key=lambda x: x[0].due_date)

        # Если у ресурса не указаны типы, он универсален
        if not hasattr(self, 'cached_res_op_types'):
            self.cached_res_op_types = get_resource_operation_types()

        for order, op in ops:
            # Допустимые ресурсы: если у операции есть тип, берём те, что с ним связаны
            if op.type_id:
                allowed_rids = self.cached_res_op_types.get(op.type_id, [])
            else:
                allowed_rids = list(self.resource_map.keys())
            if not allowed_rids:
                allowed_rids = list(self.resource_map.keys())

            # Выбираем ресурс, который освободится раньше всех (Least Workload)
            best_rid = min(allowed_rids, key=lambda rid: work_until.get(rid, datetime.max))
            start = max(work_until[best_rid], datetime.now())
            dur = timedelta(minutes=op.norm_duration)
            end = start + dur
            schedule[op.id] = {'resource_id': best_rid, 'start': start, 'end': end}
            work_until[best_rid] = end

        return schedule

    def _try_start_next_on_resource(self, resource_id):
        res = self.resource_map.get(resource_id)
        if res and (getattr(res, 'repair', 0) or getattr(res, 'reliability', 1.0) <= 0):
            return
        tasks = get_tasks_for_resource(resource_id)
        if any(t['status'] == 'active' for t in tasks):
            return
        pending = [t for t in tasks if t['status'] == 'pending']
        if pending:
            self.move_task_to_production(pending[0]['operation_id'], resource_id)

    def start_all_queues(self):
        """Запускает первую ожидающую задачу на каждом ресурсе, если он свободен и исправен."""
        for res in self.resources:
            if getattr(res, 'repair', 0) or getattr(res, 'reliability', 1.0) <= 0:
                continue
            self._try_start_next_on_resource(res.id)
        self.dashboard_changed.emit()
        self.message_signal.emit("info", "Очереди запущены.")

    def run_milp(self):
        """Быстрое эвристическое планирование (замена точного MILP)."""
        if not self.orders:
            return
        self.current_schedule = self._build_heuristic_schedule(only_unscheduled=True)
        self._sync_schedule_to_task_queue()
        self.dashboard_changed.emit()
        self.message_signal.emit("info", "Эвристический план построен.")

    def reset_all_queues(self):
        """Перемещает все задачи из очередей и планов в нераспределённые."""
        # Удаляем из очереди и расписаний все операции
        for res in self.resources:
            tasks = get_tasks_for_resource(res.id)
            for t in tasks:
                self._remove_from_schedules(t['operation_id'])
                delete_task_from_queue(t['operation_id'])
        # Очищаем текущий план
        self.current_schedule.clear()
        # Обновляем дашборд
        self.dashboard_changed.emit()
        self.message_signal.emit("info", "Все задачи возвращены в нераспределённые.")

    def _build_state_for_agent(self, env, ready_ops):
        state = env._get_state()
        busy = np.array([1.0 if env.resource_remaining[rid] > 0 else 0.0 for rid in env.resource_ids])
        ext_state = np.concatenate([state, busy])
        op_feat = np.zeros((env.max_actions, 5))
        mask = np.zeros(env.max_actions, dtype=bool)
        for i, op_info in enumerate(ready_ops):
            if i >= env.max_actions:
                break
            op_id = op_info['op_id']
            rid = op_info['resource_id']
            due_min = 0;
            rem_time = 0;
            weight = 0;
            progress = 0
            for order in env.orders:
                for op in order.ops:
                    if op.id == op_id:
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
        return ext_state, op_feat, mask

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
        # Удаляем операцию из всех планов и других очередей
        self._remove_from_schedules(op_id)
        delete_task_from_queue(op_id)
        # Добавляем в очередь выбранного ресурса
        add_task_to_queue(resource_id, op_id)
        log_operation_state(op_id, 'pending', resource_id)
        # Пытаемся сразу запустить, если ресурс свободен
        self._try_start_next_on_resource(resource_id)
        # Обновляем интерфейс
        self.dashboard_changed.emit()
        self.message_signal.emit("info", f"Заказ {op_id} помещён в очередь {resource_id}")

    def move_task_to_production(self, op_id, resource_id):
        # Делаем операцию активной на указанном ресурсе
        update_task_status(op_id, 'active')
        log_operation_state(op_id, 'active', resource_id)
        # Оповещаем GUI
        self.dashboard_changed.emit()
        self.message_signal.emit("info", f"Заказ {op_id} запущен в производство на {resource_id}")

    def complete_task(self, op_id):
        # Ищем ресурс, на котором операция сейчас (active/pending)
        found_resource = None
        for res in self.resources:
            tasks = get_tasks_for_resource(res.id)
            if any(t['operation_id'] == op_id for t in tasks):
                found_resource = res.id
                break

        if not found_resource:
            self.dashboard_changed.emit()
            return

        # Завершаем операцию
        update_task_status(op_id, 'completed')
        log_operation_state(op_id, 'completed', resource_id=found_resource)
        self._remove_from_schedules(op_id)
        delete_task_from_queue(op_id)

        # Находим заказ (теперь операция одна на заказ)
        order = self._get_order_by_op(op_id)
        if order:
            # Помечаем заказ завершённым в БД
            set_order_status(order.id, 'completed')
            # Убираем из активного списка (чтобы не отображался в backlog и на дашборде)
            self.orders = [o for o in self.orders if o.id != order.id]
            # Оповещаем GUI
            self.orders_changed.emit()
            self.dashboard_changed.emit()
        else:
            self.dashboard_changed.emit()
            return

        # Пробуем запустить следующую ожидающую задачу на освободившемся ресурсе
        self._try_start_next_on_resource(found_resource)

    def plan_with_rl(self):
        if not self.rl_agent:
            self.message_signal.emit("error", "Модель не загружена.")
            return

        # Собираем неразмещённые операции, сортируем по EDD
        unscheduled = []
        for order in self.orders:
            for op in order.ops:
                if op.id not in self.current_schedule and op.id not in self.approved_schedule:
                    unscheduled.append((order, op))
        if not unscheduled:
            return
        unscheduled.sort(key=lambda x: x[0].due_date)

        # Кэш типов
        if not hasattr(self, 'cached_res_op_types'):
            self.cached_res_op_types = get_resource_operation_types()
        type_map = get_operation_types()
        types_list = ['фрезеровка', 'сборка', 'сварка']

        # Подготавливаем базовые признаки ресурсов
        R = len(self.resources)
        # Для модели мы можем подкорректировать repair и reliability, чтобы не выходить за рамки обучения
        model_repair = np.zeros(R, dtype=np.float32)
        model_reliability = np.zeros(R, dtype=np.float32)
        for i, res in enumerate(self.resources):
            rel = res.reliability
            rep = res.repair
            # Приводим к допустимому диапазону (как в обучающей выборке)
            if rel < 0.25:
                rep = 1  # низкая готовность интерпретируется как ремонт
                rel = max(rel, 0.2)  # минимальная надёжность 0.2
            model_repair[i] = rep
            model_reliability[i] = rel

        work_until = {r.id: self.selected_date for r in self.resources}
        schedule = {}

        for order, op in unscheduled:
            # Признаки операции (6 чисел)
            type_onehot = np.zeros(3)
            if op.type_id and op.type_id in type_map:
                tname = type_map[op.type_id]
                if tname in types_list:
                    type_onehot[types_list.index(tname)] = 1.0
            op_feat = np.array([
                *type_onehot,
                op.norm_duration / 120.0,
                order.priority_weight,
                (op.slack_hours if hasattr(op, 'slack_hours') else 0) / 48.0
            ])

            # Допустимые ресурсы по типам
            allowed = self.cached_res_op_types.get(op.type_id, []) if op.type_id else list(self.resource_map.keys())
            if not allowed:
                allowed = list(self.resource_map.keys())

            # Строим полный вектор признаков (6 + R*6)
            full = op_feat.copy()
            for i, res in enumerate(self.resources):
                # Текущая загрузка (в минутах работы, нормированная)
                cur_load = (work_until[res.id] - self.selected_date).total_seconds() / 60.0 / 240.0
                compat = 1.0 if res.id in allowed else 0.0
                res_vec = np.array([
                    1.0 if res.status == 'Работает' else 0.0,  # ready
                    cur_load,
                    compat,
                    res.load_minutes / 240.0,  # avg load
                    model_repair[i],  # repair (скорректированный)
                    model_reliability[i]  # reliability (скорректированная)
                ])
                full = np.concatenate([full, res_vec])

            # --- Получаем логиты от модели ---
            with torch.no_grad():
                x = torch.tensor(full, dtype=torch.float32).unsqueeze(0)
                logits = self.rl_agent.model(x).squeeze(0)  # (R,)

            # --- Штраф за низкую готовность (постобработка) ---
            # Используем скорректированные значения, чтобы штраф был согласован с моделью
            effective = model_reliability * (1.0 - 0.5 * model_repair)
            adjusted = logits + torch.tensor(np.log(effective + 1e-6))
            best_idx = adjusted.argmax().item()
            best_rid = self.resources[best_idx].id

            # Планируем
            start = max(work_until[best_rid], self.selected_date)
            dur = timedelta(minutes=op.norm_duration)
            end = start + dur
            schedule[op.id] = {'resource_id': best_rid, 'start': start, 'end': end}
            work_until[best_rid] = end

        self.current_schedule = schedule
        self._sync_schedule_to_task_queue()
        self.dashboard_changed.emit()
        self.message_signal.emit("info", "План построен нейросетью с учётом готовности.")

    def plan_with_reliability(self):
        """
        Планирование с учётом готовности ресурсов.
        Использует эвристику EDD + эффективная загрузка (workload / reliability).
        """
        unscheduled = []
        for order in self.orders:
            for op in order.ops:
                if op.id not in self.current_schedule and op.id not in self.approved_schedule:
                    unscheduled.append((order, op))
        if not unscheduled:
            return

        unscheduled.sort(key=lambda x: x[0].due_date)

        if not hasattr(self, 'cached_res_op_types'):
            self.cached_res_op_types = get_resource_operation_types()

        work_until = {r.id: self.selected_date for r in self.resources}
        schedule = {}

        for order, op in unscheduled:
            if op.type_id:
                allowed_rids = self.cached_res_op_types.get(op.type_id, [])
            else:
                allowed_rids = list(self.resource_map.keys())
            if not allowed_rids:
                allowed_rids = list(self.resource_map.keys())

            def effective_load(rid):
                workload = (work_until[rid] - self.selected_date).total_seconds() / 60.0
                rel = getattr(self.resource_map[rid], 'reliability', 1.0)
                if rel <= 0.0:
                    return float('inf')
                if getattr(self.resource_map[rid], 'repair', 0):
                    rel *= 0.5
                return workload / rel

            best_rid = min(allowed_rids, key=effective_load)
            start = work_until[best_rid]
            dur = timedelta(minutes=op.norm_duration)
            end = start + dur
            schedule[op.id] = {'resource_id': best_rid, 'start': start, 'end': end}
            work_until[best_rid] = end

        self.current_schedule = schedule
        self._sync_schedule_to_task_queue()
        self.dashboard_changed.emit()
        self.message_signal.emit("info", "План построен с учётом готовности.")

    def add_resource(self, rid, name):
        if rid in self.resource_map:
            self.message_signal.emit("error", "Ресурс уже существует.")
            return
        r = Resource(rid, name, reliability=1.0, repair=0)
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

    def load_rl_model_if_available(self):
        path = self.settings.get('model_path')
        if not path or not os.path.exists(path):
            return False
        try:
            if path.endswith('.pkl'):
                self.rl_agent = BCAgent(path)
            elif path.endswith('.pth'):
                self.rl_agent = SimpleAgent(path)  # <-- теперь используем простую модель
            else:
                return False
            self.loaded_model_name = os.path.splitext(os.path.basename(path))[0]
            return True
        except Exception as e:
            self.message_signal.emit("error", f"Ошибка загрузки модели: {e}")
            return False