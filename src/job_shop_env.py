# job_shop_env.py — полный файл
import numpy as np
from typing import List, Dict, Tuple
from datamodels import Order, Resource, Operation
import copy
from datetime import datetime, timedelta

class JobShopEnv:
    def __init__(self, orders: List[Order], resources: List[Resource],
                 max_queue_per_resource: int = 5, predictor=None, gamma=0.95):
        self.original_orders = orders
        self.resources = {r.id: r for r in resources}
        self.resource_ids = list(self.resources.keys())
        self.max_queue_per_resource = max_queue_per_resource
        self.predictor = predictor
        self.gamma = gamma
        self.state_dim = 1 + len(self.resource_ids) * max_queue_per_resource * 4
        self.max_actions = sum(len(order.ops) for order in orders)
        self.reset()

    def reset(self) -> np.ndarray:
        self.orders = copy.deepcopy(self.original_orders)
        self.start_datetime = datetime(2026, 4, 27, 8, 0, 0)
        self.current_time = 0.0
        self.op_status = {}
        self.op_remaining = {}
        self.resource_remaining = {r_id: 0.0 for r_id in self.resource_ids}
        self.resource_current_op = {r_id: None for r_id in self.resource_ids}
        for order in self.orders:
            for op in order.ops:
                self.op_status[op.id] = 'pending'
                self.op_remaining[op.id] = self._get_remaining_time(op, self.current_time)
        self._update_ready_status()
        self.done = False
        self.episode_reward = 0.0
        return self._get_state()

    def step(self, op_id: str) -> Tuple[np.ndarray, float, bool, Dict]:
        resource_id = self._get_resource_of_op(op_id)
        if resource_id is None or self.op_status[op_id] != 'ready' or self.resource_remaining[resource_id] > 0:
            raise ValueError(f"Невозможно запустить {op_id}")

        self.op_status[op_id] = 'running'
        self.resource_remaining[resource_id] = self.op_remaining[op_id]
        self.resource_current_op[resource_id] = op_id

        self._advance_until_resource_free()

        all_completed = all(status == 'completed' for status in self.op_status.values())
        self.done = all_completed

        reward = 0.0
        if self.done:
            total_tardiness = 0.0
            for order in self.orders:
                due_minutes = self._order_due_minutes(order)
                tardiness = max(0.0, self.current_time - due_minutes)
                total_tardiness += order.priority_weight * tardiness
            reward = -total_tardiness
        self.episode_reward += reward
        return self._get_state(), reward, self.done, {}

    def _advance_until_resource_free(self):
        while True:
            busy = [(rid, t) for rid, t in self.resource_remaining.items() if t > 0]
            if not busy:
                break
            min_time = min(t for _, t in busy)
            self.current_time += min_time
            for rid in self.resource_ids:
                if self.resource_remaining[rid] > 0:
                    self.resource_remaining[rid] -= min_time
                    if self.resource_remaining[rid] < 1e-6:
                        self.resource_remaining[rid] = 0.0
                        finished_op = self.resource_current_op[rid]
                        self.op_status[finished_op] = 'completed'
                        self.resource_current_op[rid] = None
            self._update_ready_status()
            if any(self.resource_remaining[rid] == 0.0 and self.resource_current_op[rid] is None and
                   self._get_ready_queue_for_resource(rid) for rid in self.resource_ids):
                break

    def _update_ready_status(self):
        for order in self.orders:
            for op in order.ops:
                if self.op_status[op.id] != 'pending':
                    continue
                if all(self.op_status[pred] == 'completed' for pred in op.predecessors):
                    self.op_status[op.id] = 'ready'
                    self.op_remaining[op.id] = self._get_remaining_time(op, self.current_time)

    def _get_state(self) -> np.ndarray:
        state = [self.current_time / 10000.0]
        for r_id in self.resource_ids:
            queue = self._get_ready_queue_for_resource(r_id)
            for i in range(self.max_queue_per_resource):
                if i < len(queue):
                    op_info = queue[i]
                    slack = (op_info['due_minutes'] - self.current_time) / 1440.0
                    rem_proc = op_info['remaining_time'] / 1440.0
                    weight = op_info['weight'] / 10.0
                    progress = op_info['progress']
                    state.extend([slack, rem_proc, weight, progress])
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0])
        return np.array(state, dtype=np.float32)

    def _get_ready_queue_for_resource(self, resource_id: str) -> List[Dict]:
        queue = []
        for order in self.orders:
            for op in order.ops:
                if op.resource_id == resource_id and self.op_status[op.id] == 'ready':
                    due_minutes = self._order_due_minutes(order)
                    queue.append({
                        'op_id': op.id,
                        'due_minutes': due_minutes,
                        'remaining_time': self.op_remaining[op.id],
                        'weight': order.priority_weight,
                        'progress': sum(1 for o in order.ops if self.op_status[o.id] == 'completed') / len(order.ops)
                    })
        return queue

    def get_ready_ops_list(self) -> List[Dict]:
        ready = []
        for order in self.orders:
            for op in order.ops:
                if self.op_status[op.id] == 'ready':
                    # Если ресурс не указан или пустой, считаем, что он свободен (0)
                    if op.resource_id and op.resource_id in self.resource_remaining:
                        if self.resource_remaining[op.resource_id] == 0.0:
                            ready.append({'op_id': op.id, 'resource_id': op.resource_id})
                    else:
                        # Если ресурс не назначен, операция доступна для любого
                        ready.append({'op_id': op.id, 'resource_id': self.resource_ids[0] if self.resource_ids else ''})
        return ready

    def _get_resource_of_op(self, op_id: str):
        for order in self.orders:
            for op in order.ops:
                if op.id == op_id:
                    return op.resource_id if op.resource_id else self.resource_ids[0]
        return None

    def _order_due_minutes(self, order: Order) -> float:
        delta = order.due_date - self.start_datetime
        return delta.total_seconds() / 60.0

    def _get_remaining_time(self, op: Operation, current_time: float) -> float:
        if self.predictor and hasattr(self.predictor, 'trained') and self.predictor.trained:
            actual_dt = self.start_datetime + timedelta(minutes=current_time)
            return self.predictor.predict(op, self.resources[op.resource_id] if op.resource_id else self.resource_ids[0], actual_dt)
        return op.norm_duration