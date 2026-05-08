# aps_core.py
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from datamodels import Order, Operation, Resource
from optimizer_milp import build_milp_schedule
from predictor import DurationPredictor
from database import (get_fixed_operations, get_planned_operations,
                      add_proposed_change, get_proposed_changes, accept_change,
                      apply_accepted_changes, log_operation_state, get_execution_data,
                      update_schedule)

class HybridAPS:
    def __init__(self, resources, predictor=None, rl_agent=None, use_rl=False):
        self.resources = {r.id: r for r in resources}
        self.predictor = predictor or DurationPredictor()
        self.rl_agent = rl_agent
        self.use_rl = use_rl and rl_agent is not None
        self.orders = []
        self.current_schedule = {}
        self.current_time = datetime.now()

    def load_orders(self, orders):
        self.orders = orders

    def run_milp(self):
        if not self.orders:
            return
        schedule = build_milp_schedule(self.orders, list(self.resources.values()), self.current_time)
        if schedule:
            self._save_schedule(schedule, fixed=False, status='planned')

    def run_milp_with_fixed(self):
        if not self.orders:
            return
        fixed = get_fixed_operations()
        schedule = build_milp_schedule(self.orders, list(self.resources.values()), self.current_time, fixed_ops=fixed)
        if schedule:
            for op_id, info in schedule.items():
                if op_id in fixed:
                    update_schedule(op_id, info['resource_id'], info['start'], info['end'], fixed=1, status='fixed')
                else:
                    update_schedule(op_id, info['resource_id'], info['start'], info['end'], fixed=0, status='planned')
            self.current_schedule = schedule

    def run_milp_incremental(self, new_orders=None):
        all_orders = self.orders.copy()
        if new_orders:
            all_orders.extend(new_orders)
        fixed = get_fixed_operations()
        planned = get_planned_operations()
        valid_op_ids = {op.id for order in all_orders for op in order.ops}
        filtered_fixed = {op_id: info for op_id, info in fixed.items() if op_id in valid_op_ids}

        schedule = build_milp_schedule(all_orders, list(self.resources.values()), self.current_time, fixed_ops=filtered_fixed)
        if not schedule:
            return
        for op_id, info in schedule.items():
            old = planned.get(op_id)
            if old and (old['resource_id'] != info['resource_id'] or old['start'] != info['start'] or old['end'] != info['end']):
                add_proposed_change(op_id, info['resource_id'], info['start'], info['end'])
            elif not old and op_id not in filtered_fixed:
                update_schedule(op_id, info['resource_id'], info['start'], info['end'], fixed=0, status='planned')
        self.current_schedule = schedule

    def apply_changes(self):
        apply_accepted_changes()
        fixed = get_fixed_operations()
        self.current_schedule = build_milp_schedule(self.orders, list(self.resources.values()), self.current_time, fixed_ops=fixed)

    def handle_mes_event(self, event):
        if event.event_type == 'complete':
            log_execution(event.operation_id, event.timestamp, event.timestamp + timedelta(minutes=event.actual_duration),
                          event.resource_id, '')
            self._operational_dispatch(event.resource_id)

    def _operational_dispatch(self, resource_id):
        """Оперативная диспетчеризация: CR или RL (PPO/BC)"""
        ready = self._get_ready_ops(resource_id)
        if not ready:
            return
        if self.use_rl and self.rl_agent:
            # Собираем состояние среды и признаки операций (упрощённо)
            from job_shop_env import JobShopEnv
            env = JobShopEnv(self.orders, list(self.resources.values()))
            env.reset()
            # ... синхронизация состояния опущена для краткости ...
            # Здесь нужно актуализировать среду по текущему состоянию, после чего вызвать
            # agent.select_action(state, op_feat, mask)
            # Пока заглушка
            best_op = ready[0]
        else:
            best_op = min(ready, key=lambda op: self._critical_ratio(op))
        # Отправляем команду MES (заглушка)

    def _critical_ratio(self, op):
        order = self._get_order_by_op(op.id)
        dd = order.due_date
        now = self.current_time
        remaining = sum(o.norm_duration for o in order.ops if o.op_number >= op.op_number)
        delay = max(0, (now - (dd - timedelta(minutes=remaining))).total_seconds() / 60)
        k_ut = 1 + delay / max(1, (dd - now).total_seconds() / 60)
        cr = (dd - now).total_seconds() / 60 / (remaining * k_ut) if remaining > 0 else 0
        return cr

    def train_predictor(self):
        data = get_execution_data()
        features, targets = [], []
        for entry in data:
            op = self._get_operation_by_id(entry['operation_id'])
            if op:
                f = self.predictor.featurize(op, self.resources[entry['resource_id']], datetime.now())
                features.append(f)
                targets.append(entry['actual_duration'])
        if features:
            self.predictor.train({'features': features, 'actual_duration': targets})
            self.predictor.save('predictor.pkl')

    def _save_schedule(self, schedule, fixed=False, status='planned'):
        for op_id, info in schedule.items():
            update_schedule(op_id, info['resource_id'], info['start'], info['end'],
                            fixed=1 if fixed else 0, status=status)
        self.current_schedule = schedule

    def _get_ready_ops(self, resource_id):
        # заглушка
        return []

    def _get_order_by_op(self, op_id):
        for order in self.orders:
            for op in order.ops:
                if op.id == op_id:
                    return order
        return None

    def _get_operation_by_id(self, op_id):
        for order in self.orders:
            for op in order.ops:
                if op.id == op_id:
                    return op
        return None