# datamodels.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class Resource:
    id: str
    name: str
    calendar_id: str = "24/7"

@dataclass
class Operation:
    id: str
    order_id: str
    item: str
    op_number: int
    resource_id: str
    norm_duration: float        # минуты
    predecessors: List[str] = field(default_factory=list)

@dataclass
class Order:
    id: str
    due_date: datetime
    priority_weight: float = 1.0
    ops: List[Operation] = field(default_factory=list)

@dataclass
class MESEvent:
    timestamp: datetime
    resource_id: str
    operation_id: str
    event_type: str             # 'start','complete','breakdown','repair'
    actual_duration: Optional[float] = None