# aps_core.py
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from datamodels import Order, Operation, Resource, MESEvent
from optimizer_milp import build_milp_schedule
from predictor import DurationPredictor
from dqn_agent import DQNAgent
from database import get_fixed_operations, update_schedule

class HybridAPS:
    def __init__(self, resources: List[Resource], predictor: Optional[DurationPredictor] = None,
                 rl_agent: Optional[DQNAgent] = None, use_rl: bool = False):
        self.resources = {r.id: r for r in resources}
        self.predictor = predictor or DurationPredictor()
        self.rl_agent = rl_agent
        self.use_rl = use_rl and rl_agent is not None
        self.orders: List[Order] = []
        self.current_schedule: Dict[str, dict] = {}
        self.current_time = datetime.now()
        self.milp_interval = timedelta(hours=4)
        self.last_milp_time = None

    def load_orders(self, orders: List[Order]):
        self.orders = orders
        # Не запускаем MILP автоматически, чтобы дать пользователю возможность изменить фиксированные операции
        # self.run_milp()  # можно вызвать вручную из GUI

    def run_milp(self):
        """Обычный запуск MILP без фиксаций."""
        if not self.orders:
            return
        # Обновляем прогноз для длительностей (пока просто норматив)
        for order in self.orders:
            for op in order.ops:
                op.norm_duration = self.predictor.predict(op, self.resources[op.resource_id], self.current_time)
        schedule = build_milp_schedule(self.orders, list(self.resources.values()), self.current_time)
        if schedule:
            self.current_schedule = schedule
            self.last_milp_time = self.current_time
            # Сохраняем расписание в БД (как нефиксированное)
            for op_id, info in schedule.items():
                update_schedule(op_id, info['resource_id'], info['start'], info['end'], fixed=0)

    def run_milp_with_fixed(self):
        """Запуск MILP с учётом фиксированных операций из БД."""
        if not self.orders:
            return
        # Получаем фиксированные операции из БД
        fixed = get_fixed_operations()
        # Перед запуском MILP применяем изменения ресурсов для фиксированных операций.
        # Ищем операцию в self.orders и меняем ей resource_id, если он изменился.
        for order in self.orders:
            for op in order.ops:
                if op.id in fixed:
                    new_res = fixed[op.id]['resource_id']
                    if new_res and new_res != op.resource_id:
                        op.resource_id = new_res  # переназначаем
        # Теперь вызываем MILP с передачей fixed_ops
        schedule = build_milp_schedule(self.orders, list(self.resources.values()), self.current_time, fixed_ops=fixed)
        if schedule:
            self.current_schedule = schedule
            self.last_milp_time = self.current_time
            # Обновляем БД для всех операций (сбрасываем fixed=0 для незафиксированных)
            for op_id, info in schedule.items():
                update_schedule(op_id, info['resource_id'], info['start'], info['end'],
                                fixed=1 if op_id in fixed else 0)

    def handle_event(self, event: MESEvent):
        """Заглушка для интеграции с MES."""
        if event.event_type == 'complete':
            self._dispatch_resource(event.resource_id)
        if self.last_milp_time and (self.current_time - self.last_milp_time) > self.milp_interval:
            self.run_milp_with_fixed()  # перепланирование с сохранением фиксаций

    def _dispatch_resource(self, resource_id: str):
        # Заглушка – реальная диспетчеризация вызывается по необходимости
        pass