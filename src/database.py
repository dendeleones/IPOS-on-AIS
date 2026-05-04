# database.py
import sqlite3
from datetime import datetime

DB_NAME = "aps.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
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
            type_id TEXT DEFAULT '',
            predecessors TEXT DEFAULT '',
            FOREIGN KEY (order_id) REFERENCES orders (id)
        )
    """)
    cur.execute("PRAGMA table_info(operations)")
    cols = [c[1] for c in cur.fetchall()]
    if 'predecessors' not in cols:
        cur.execute("ALTER TABLE operations ADD COLUMN predecessors TEXT DEFAULT ''")
    if 'type_id' not in cols:
        cur.execute("ALTER TABLE operations ADD COLUMN type_id TEXT DEFAULT ''")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS resources (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS resource_types (
            resource_id TEXT,
            type_id TEXT,
            PRIMARY KEY (resource_id, type_id),
            FOREIGN KEY (resource_id) REFERENCES resources(id),
            FOREIGN KEY (type_id) REFERENCES operation_types(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS operation_types (
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
            status TEXT DEFAULT 'planned',
            FOREIGN KEY (operation_id) REFERENCES operations (id)
        )
    """)
    cur.execute("PRAGMA table_info(schedule)")
    cols = [c[1] for c in cur.fetchall()]
    if 'status' not in cols:
        cur.execute("ALTER TABLE schedule ADD COLUMN status TEXT DEFAULT 'planned'")
        cur.execute("UPDATE schedule SET status='fixed' WHERE fixed=1")
        cur.execute("UPDATE schedule SET status='planned' WHERE fixed=0")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS proposed_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_id TEXT,
            new_resource_id TEXT,
            new_start TEXT,
            new_end TEXT,
            accepted INTEGER DEFAULT 0,
            FOREIGN KEY (operation_id) REFERENCES operations (id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS execution_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_id TEXT,
            actual_start TEXT,
            actual_end TEXT,
            resource_id TEXT,
            state_id TEXT,
            FOREIGN KEY (operation_id) REFERENCES operations (id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS state_types (
            id TEXT PRIMARY KEY,
            name TEXT,
            color TEXT
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
    conn.commit()
    conn.close()

# ----- Orders -----
def save_orders_to_db(orders):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    for order in orders:
        cur.execute("INSERT OR REPLACE INTO orders (id, due_date, priority_weight) VALUES (?, ?, ?)",
                    (order.id, order.due_date.isoformat(), order.priority_weight))
        for op in order.ops:
            preds = ",".join(op.predecessors) if op.predecessors else ""
            type_id = getattr(op, 'type_id', '')
            cur.execute("INSERT OR REPLACE INTO operations (id, order_id, item, op_number, resource_id, norm_duration, type_id, predecessors) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (op.id, order.id, op.item, op.op_number, op.resource_id, op.norm_duration, type_id, preds))
    conn.commit()
    conn.close()

def load_orders_from_db(resources):
    from datamodels import Order, Operation
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(operations)")
    cols = [c[1] for c in cur.fetchall()]
    has_preds = 'predecessors' in cols
    has_type = 'type_id' in cols
    cur.execute("SELECT * FROM orders")
    orders = []
    for row in cur:
        order = Order(id=row[0], due_date=datetime.fromisoformat(row[1]), priority_weight=row[2])
        cur2 = conn.execute("SELECT * FROM operations WHERE order_id=?", (order.id,))
        for op_row in cur2:
            preds = []
            type_id = ''
            if has_preds and len(op_row) >= 7:
                preds_str = op_row[6] if op_row[6] else ''
                preds = preds_str.split(",") if preds_str else []
            if has_type and len(op_row) >= 8:
                type_id = op_row[7] if op_row[7] else ''
            op = Operation(
                id=op_row[0], order_id=op_row[1], item=op_row[2],
                op_number=op_row[3], resource_id=op_row[4],
                norm_duration=op_row[5], predecessors=preds
            )
            if type_id:
                op.type_id = type_id
            order.ops.append(op)
        orders.append(order)
    conn.close()
    return orders

# ----- Resources -----
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

# ----- Resource Types -----
def add_operation_type(name):
    import uuid
    tid = str(uuid.uuid4())[:8]
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO operation_types (id, name) VALUES (?, ?)", (tid, name))
    conn.commit()
    conn.close()
    return tid

def get_operation_types():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM operation_types")
    types = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return types

def delete_operation_type(type_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM operation_types WHERE id=?", (type_id,))
    conn.commit()
    conn.close()

def set_resource_types(resource_id, type_ids):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM resource_types WHERE resource_id=?", (resource_id,))
    for tid in type_ids:
        cur.execute("INSERT INTO resource_types (resource_id, type_id) VALUES (?, ?)", (resource_id, tid))
    conn.commit()
    conn.close()

def get_resource_types():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT resource_id, type_id FROM resource_types")
    res_types = {}
    for row in cur:
        res_id, tid = row
        res_types.setdefault(res_id, []).append(tid)
    conn.close()
    return res_types

# ----- Schedule -----
def update_schedule(operation_id, resource_id, start, end, fixed=1, status='fixed'):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    if resource_id is None:
        cur.execute("DELETE FROM schedule WHERE operation_id=?", (operation_id,))
    else:
        cur.execute("INSERT OR REPLACE INTO schedule (operation_id, resource_id, start, end, fixed, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (operation_id, resource_id, start.isoformat(), end.isoformat(), fixed, status))
    conn.commit()
    conn.close()

def get_fixed_operations():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT operation_id, resource_id, start, end FROM schedule WHERE status='fixed'")
    fixed = {}
    for row in cur:
        fixed[row[0]] = {'resource_id': row[1], 'start': datetime.fromisoformat(row[2]), 'end': datetime.fromisoformat(row[3])}
    conn.close()
    return fixed

def get_planned_operations():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT operation_id, resource_id, start, end FROM schedule WHERE status='planned'")
    planned = {}
    for row in cur:
        planned[row[0]] = {'resource_id': row[1], 'start': datetime.fromisoformat(row[2]) if row[2] else None,
                           'end': datetime.fromisoformat(row[3]) if row[3] else None}
    conn.close()
    return planned

# ----- proposed changes -----
def add_proposed_change(operation_id, new_resource_id, new_start, new_end):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO proposed_changes (operation_id, new_resource_id, new_start, new_end) VALUES (?, ?, ?, ?)",
                (operation_id, new_resource_id, new_start.isoformat(), new_end.isoformat()))
    conn.commit()
    conn.close()

def get_proposed_changes():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT id, operation_id, new_resource_id, new_start, new_end FROM proposed_changes WHERE accepted=0")
    changes = []
    for row in cur:
        changes.append({
            'id': row[0], 'operation_id': row[1], 'new_resource_id': row[2],
            'new_start': datetime.fromisoformat(row[3]), 'new_end': datetime.fromisoformat(row[4])
        })
    conn.close()
    return changes

def accept_change(change_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("UPDATE proposed_changes SET accepted=1 WHERE id=?", (change_id,))
    conn.commit()
    conn.close()

def apply_accepted_changes():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT operation_id, new_resource_id, new_start, new_end FROM proposed_changes WHERE accepted=1")
    for row in cur:
        op_id, res, start, end = row
        cur.execute("UPDATE schedule SET resource_id=?, start=?, end=?, status='fixed' WHERE operation_id=?",
                    (res, start, end, op_id))
    conn.commit()
    conn.close()

# ----- Execution log -----
def log_execution(operation_id, actual_start, actual_end, resource_id, state_id=""):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO execution_log (operation_id, actual_start, actual_end, resource_id, state_id) VALUES (?, ?, ?, ?, ?)",
                (operation_id, actual_start.isoformat(), actual_end.isoformat(), resource_id, state_id))
    conn.commit()
    conn.close()

def get_execution_data():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        SELECT e.operation_id, e.resource_id, o.norm_duration,
               (julianday(e.actual_end) - julianday(e.actual_start)) * 1440
        FROM execution_log e
        JOIN operations o ON e.operation_id = o.id
    """)
    data = []
    for row in cur:
        data.append({'operation_id': row[0], 'resource_id': row[1], 'norm_duration': row[2], 'actual_duration': row[3]})
    conn.close()
    return data

# ----- Downtimes & States -----
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
        downtimes.setdefault(res_id, []).append((datetime.fromisoformat(start_str), datetime.fromisoformat(end_str), state_id))
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
    result = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
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