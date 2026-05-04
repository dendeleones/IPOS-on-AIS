# database.py
import sqlite3
from datetime import datetime

DB_NAME = "aps.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # Создание таблиц, если их нет
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            due_date TEXT NOT NULL,
            priority_weight REAL NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS operations (
            id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            item TEXT,
            op_number INTEGER,
            resource_id TEXT,
            norm_duration REAL,
            FOREIGN KEY (order_id) REFERENCES orders (id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS resources (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schedule (
            operation_id TEXT PRIMARY KEY,
            resource_id TEXT,
            start TEXT,
            end TEXT,
            fixed INTEGER DEFAULT 0,
            FOREIGN KEY (operation_id) REFERENCES operations (id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            resource_id TEXT,
            operation_id TEXT,
            event_type TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS downtimes (
            resource_id TEXT,
            start TEXT,
            end TEXT,
            state_id TEXT,
            FOREIGN KEY (resource_id) REFERENCES resources (id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS state_types (
            id TEXT PRIMARY KEY,
            name TEXT,
            color TEXT
        )
    """)

    # Проверка и добавление колонки state_id, если её ещё нет
    cur.execute("PRAGMA table_info(downtimes)")
    columns = [col[1] for col in cur.fetchall()]
    if 'state_id' not in columns:
        cur.execute("ALTER TABLE downtimes ADD COLUMN state_id TEXT")

    conn.commit()
    conn.close()

def save_orders_to_db(orders):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    for order in orders:
        cur.execute("INSERT OR REPLACE INTO orders (id, due_date, priority_weight) VALUES (?, ?, ?)",
                    (order.id, order.due_date.isoformat(), order.priority_weight))
        for op in order.ops:
            cur.execute("INSERT OR REPLACE INTO operations (id, order_id, item, op_number, resource_id, norm_duration) VALUES (?, ?, ?, ?, ?, ?)",
                        (op.id, order.id, op.item, op.op_number, op.resource_id, op.norm_duration))
    conn.commit()
    conn.close()

def load_orders_from_db(resources):
    from datamodels import Order, Operation
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT * FROM orders")
    orders = []
    for row in cur:
        order = Order(id=row[0], due_date=datetime.fromisoformat(row[1]), priority_weight=row[2])
        cur2 = conn.execute("SELECT * FROM operations WHERE order_id=?", (order.id,))
        for op_row in cur2:
            op = Operation(id=op_row[0], order_id=op_row[1], item=op_row[2],
                           op_number=op_row[3], resource_id=op_row[4],
                           norm_duration=op_row[5])
            order.ops.append(op)
        orders.append(order)
    conn.close()
    return orders

def update_schedule(operation_id, resource_id, start, end, fixed=1):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    if resource_id is None:
        cur.execute("UPDATE schedule SET fixed=0 WHERE operation_id=?", (operation_id,))
    else:
        cur.execute("INSERT OR REPLACE INTO schedule (operation_id, resource_id, start, end, fixed) VALUES (?, ?, ?, ?, ?)",
                    (operation_id, resource_id, start.isoformat(), end.isoformat(), fixed))
    conn.commit()
    conn.close()

def get_fixed_operations():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT operation_id, resource_id, start, end FROM schedule WHERE fixed=1")
    fixed = {}
    for row in cur:
        fixed[row[0]] = {
            'resource_id': row[1],
            'start': datetime.fromisoformat(row[2]),
            'end': datetime.fromisoformat(row[3])
        }
    conn.close()
    return fixed

def save_resources(resources):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM resources")
    for r in resources:
        cur.execute("INSERT INTO resources (id, name) VALUES (?, ?)", (r.id, r.name))
    conn.commit()
    conn.close()

def load_resources():
    from datamodels import Resource
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM resources")
    result = [Resource(id=row[0], name=row[1]) for row in cur]
    conn.close()
    return result

def delete_resource(resource_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM resources WHERE id=?", (resource_id,))
    conn.commit()
    conn.close()

def save_downtime(resource_id, start, end, state_id=""):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO downtimes (resource_id, start, end, state_id) VALUES (?, ?, ?, ?)",
                (resource_id, start.isoformat(), end.isoformat(), state_id))
    conn.commit()
    conn.close()

def load_downtimes():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT resource_id, start, end, state_id FROM downtimes")
    downtimes = {}
    for row in cur:
        res_id, start_str, end_str, state_id = row
        if res_id not in downtimes:
            downtimes[res_id] = []
        downtimes[res_id].append((datetime.fromisoformat(start_str), datetime.fromisoformat(end_str), state_id))
    conn.close()
    return downtimes

def delete_downtime(resource_id, start, end):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM downtimes WHERE resource_id=? AND start=? AND end=?",
                (resource_id, start.isoformat(), end.isoformat()))
    conn.commit()
    conn.close()

def get_state_types():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT id, name, color FROM state_types")
    result = {}
    for row in cur:
        result[row[0]] = (row[1], row[2])
    conn.close()
    return result

def add_state_type(name, color):
    import uuid
    sid = str(uuid.uuid4())[:8]
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO state_types (id, name, color) VALUES (?, ?, ?)", (sid, name, color))
    conn.commit()
    conn.close()
    return sid

def delete_state_type(sid):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM state_types WHERE id=?", (sid,))
    conn.commit()
    conn.close()