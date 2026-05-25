# datamodels.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class Resource:
    id: str
    name: str
    section: str = "Основной участок"
    status: str = "Работает"
    operator_name: str = ""
    load_minutes: float = 0.0
    downtime_minutes: float = 0.0
    work_hours: int = 24
    reliability: float = 1.0
    repair: int = 0

@dataclass
class Operation:
    id: str
    order_id: str
    item: str
    op_number: int
    norm_duration: float
    resource_id: str = ""
    predecessors: List[str] = field(default_factory=list)
    type_id: Optional[str] = None
    quantity: int = 1

@dataclass
class Order:
    id: str
    due_date: datetime
    priority_weight: float = 1.0
    name: str = ""
    ops: List[Operation] = field(default_factory=list)

@dataclass
class MESEvent:
    timestamp: datetime
    resource_id: str
    operation_id: Optional[str] = None
    event_type: str = ""
    new_state: Optional[str] = None
    duration: Optional[float] = None