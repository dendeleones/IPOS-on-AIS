from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import engine, SessionLocal, load_resources, load_orders_from_db, update_schedule
from aps_core import HybridAPS
from pydantic import BaseModel
from datetime import datetime

class TaskMove(BaseModel):
    resource_id: str
    start: str = None
    end: str = None

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

resources = load_resources()
aps = HybridAPS(resources)
orders = load_orders_from_db(resources)
aps.load_orders(orders)

@app.get("/api/orders")
def list_orders():
    return [{"id": o.id, "due_date": o.due_date.isoformat(), "priority": o.priority_weight} for o in orders]

@app.get("/api/schedule")
def get_schedule():
    return aps.current_schedule

@app.post("/api/tasks/{op_id}/move")
def move_task(op_id: str, move: TaskMove):
    if move.start:
        start = datetime.fromisoformat(move.start)
        end = datetime.fromisoformat(move.end) if move.end else start + timedelta(minutes=...)
    else:
        # default
        start = datetime.now().replace(second=0,microsecond=0)
        # find duration...
    aps._move_task_to_queue(op_id, move.resource_id)
    return {"status": "ok"}