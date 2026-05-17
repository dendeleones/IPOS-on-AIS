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
            priority_weight REAL NOT NULL,
            name TEXT DEFAULT ''
        )
    """)
    cur.execute("PRAGMA table_info(orders)")
    cols = [c[1] for c in cur.fetchall()]
    if 'name' not in cols:
        cur.execute("ALTER TABLE orders ADD COLUMN name TEXT DEFAULT ''")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS operations (
            id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            item TEXT,
            op_number INTEGER,
            norm_duration REAL,
            type_id TEXT DEFAULT '',
            predecessors TEXT DEFAULT '',
            quantity INTEGER DEFAULT 1,
            urgency REAL DEFAULT 1.0,
            slack_hours REAL DEFAULT 0,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        )
    """)
    cur.execute("PRAGMA table_info(operations)")
    cols = [c[1] for c in cur.fetchall()]
    if 'predecessors' not in cols:
        cur.execute("ALTER TABLE operations ADD COLUMN predecessors TEXT DEFAULT ''")
    if 'type_id' not in cols:
        cur.execute("ALTER TABLE operations ADD COLUMN type_id TEXT DEFAULT ''")
    if 'quantity' not in cols:
        cur.execute("ALTER TABLE operations ADD COLUMN quantity INTEGER DEFAULT 1")
    if 'urgency' not in cols:
        cur.execute("ALTER TABLE operations ADD COLUMN urgency REAL DEFAULT 1.0")
    if 'slack_hours' not in cols:
        cur.execute("ALTER TABLE operations ADD COLUMN slack_hours REAL DEFAULT 0")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS resources (
            id TEXT PRIMARY KEY,
            name TEXT,
            section TEXT DEFAULT 'Основной участок',
            status TEXT DEFAULT 'Работает',
            operator_name TEXT DEFAULT '',
            load_minutes REAL DEFAULT 0,
            downtime_minutes REAL DEFAULT 0,
            work_hours INTEGER DEFAULT 24,
            repair INTEGER DEFAULT 0,
            reliability REAL DEFAULT 1.0
        )
    """)
    cur.execute("PRAGMA table_info(resources)")
    cols = [c[1] for c in cur.fetchall()]
    if 'section' not in cols:
        cur.execute("ALTER TABLE resources ADD COLUMN section TEXT DEFAULT 'Основной участок'")
    if 'status' not in cols:
        cur.execute("ALTER TABLE resources ADD COLUMN status TEXT DEFAULT 'Работает'")
    if 'operator_name' not in cols:
        cur.execute("ALTER TABLE resources ADD COLUMN operator_name TEXT DEFAULT ''")
    if 'load_minutes' not in cols:
        cur.execute("ALTER TABLE resources ADD COLUMN load_minutes REAL DEFAULT 0")
    if 'downtime_minutes' not in cols:
        cur.execute("ALTER TABLE resources ADD COLUMN downtime_minutes REAL DEFAULT 0")
    if 'work_hours' not in cols:
        cur.execute("ALTER TABLE resources ADD COLUMN work_hours INTEGER DEFAULT 24")
    if 'repair' not in cols:
        cur.execute("ALTER TABLE resources ADD COLUMN repair INTEGER DEFAULT 0")
    if 'reliability' not in cols:
        cur.execute("ALTER TABLE resources ADD COLUMN reliability REAL DEFAULT 1.0")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS operation_types (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS resource_operation_types (
            resource_id TEXT NOT NULL,
            operation_type_id TEXT NOT NULL,
            PRIMARY KEY (resource_id, operation_type_id),
            FOREIGN KEY (resource_id) REFERENCES resources(id),
            FOREIGN KEY (operation_type_id) REFERENCES operation_types(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS schedule (
            operation_id TEXT PRIMARY KEY,
            resource_id TEXT NOT NULL,
            start TEXT,
            end TEXT,
            fixed INTEGER DEFAULT 0,
            status TEXT DEFAULT 'planned',
            FOREIGN KEY (operation_id) REFERENCES operations(id),
            FOREIGN KEY (resource_id) REFERENCES resources(id)
        )
    """)
    cur.execute("PRAGMA table_info(schedule)")
    cols = [c[1] for c in cur.fetchall()]
    if 'status' not in cols:
        cur.execute("ALTER TABLE schedule ADD COLUMN status TEXT DEFAULT 'planned'")
        cur.execute("UPDATE schedule SET status='fixed' WHERE fixed=1")
        cur.execute("UPDATE schedule SET status='planned' WHERE fixed=0")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS task_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resource_id TEXT NOT NULL,
            operation_id TEXT UNIQUE NOT NULL,
            position INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','active','completed')),
            FOREIGN KEY (resource_id) REFERENCES resources(id),
            FOREIGN KEY (operation_id) REFERENCES operations(id)
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
            resource_id TEXT NOT NULL,
            start TEXT NOT NULL,
            end TEXT,
            state_id TEXT,
            FOREIGN KEY (resource_id) REFERENCES resources(id),
            FOREIGN KEY (state_id) REFERENCES state_types(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS resource_state_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resource_id TEXT NOT NULL,
            state_id TEXT NOT NULL,
            start TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            end TIMESTAMP,
            source TEXT DEFAULT 'manual' CHECK(source IN ('manual','mes')),
            FOREIGN KEY (resource_id) REFERENCES resources(id),
            FOREIGN KEY (state_id) REFERENCES state_types(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS operation_state_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','active','completed','interrupted')),
            resource_id TEXT,
            state_id TEXT,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            source TEXT DEFAULT 'manual' CHECK(source IN ('manual','mes')),
            FOREIGN KEY (operation_id) REFERENCES operations(id),
            FOREIGN KEY (resource_id) REFERENCES resources(id),
            FOREIGN KEY (state_id) REFERENCES state_types(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS proposed_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_id TEXT NOT NULL,
            new_resource_id TEXT NOT NULL,
            new_start TEXT NOT NULL,
            new_end TEXT NOT NULL,
            accepted INTEGER DEFAULT 0,
            FOREIGN KEY (operation_id) REFERENCES operations(id),
            FOREIGN KEY (new_resource_id) REFERENCES resources(id)
        )
    """)

    # столбец status у заказов
    cur.execute("PRAGMA table_info(orders)")
    cols = [c[1] for c in cur.fetchall()]
    if 'status' not in cols:
        cur.execute("ALTER TABLE orders ADD COLUMN status TEXT DEFAULT 'active'")
        cur.execute("UPDATE orders SET status='active' WHERE status IS NULL")

    conn.commit()
    conn.close()

# ----- Orders -----
def save_orders_to_db(orders):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    for order in orders:
        cur.execute("INSERT OR REPLACE INTO orders (id, due_date, priority_weight, name, status) VALUES (?, ?, ?, ?, 'active')",
                    (order.id, order.due_date.isoformat(), order.priority_weight, order.name))
        for op in order.ops:
            preds = ",".join(op.predecessors) if op.predecessors else ""
            type_id = getattr(op, 'type_id', '')
            quantity = getattr(op, 'quantity', 1)
            urgency = getattr(op, 'urgency', 1.0)
            slack = getattr(op, 'slack_hours', 0.0)
            cur.execute("INSERT OR REPLACE INTO operations (id, order_id, item, op_number, norm_duration, type_id, predecessors, quantity, urgency, slack_hours) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (op.id, order.id, op.item, op.op_number, op.norm_duration, type_id, preds, quantity, urgency, slack))
    conn.commit()
    conn.close()

def load_orders_from_db(resources=None):
    from datamodels import Order, Operation
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(operations)")
    cols = [c[1] for c in cur.fetchall()]
    has_preds = 'predecessors' in cols
    has_type = 'type_id' in cols
    has_quantity = 'quantity' in cols
    has_urgency = 'urgency' in cols
    has_slack = 'slack_hours' in cols
    # Загружаем только активные заказы
    cur.execute("SELECT * FROM orders WHERE status='active'")
    orders = []
    for row in cur:
        order = Order(id=row[0], due_date=datetime.fromisoformat(row[1]), priority_weight=row[2])
        if len(row) > 3:
            order.name = row[3] if row[3] else ""
        cur2 = conn.execute("SELECT * FROM operations WHERE order_id=?", (order.id,))
        for op_row in cur2:
            preds = []
            type_id = ''
            quantity = 1
            urgency = 1.0
            slack = 0.0
            if has_preds and len(op_row) >= 7:
                preds_str = op_row[6] if op_row[6] else ''
                preds = preds_str.split(",") if preds_str else []
            if has_type and len(op_row) >= 8:
                type_id = op_row[7] if op_row[7] else ''
            if has_quantity and len(op_row) >= 9:
                try:
                    quantity = int(float(op_row[8])) if op_row[8] else 1
                except (ValueError, TypeError):
                    quantity = 1
            if has_urgency and len(op_row) >= 10:
                urgency = op_row[9] if op_row[9] else 1.0
            if has_slack and len(op_row) >= 11:
                slack = op_row[10] if op_row[10] else 0.0
            op = Operation(
                id=op_row[0], order_id=op_row[1], item=op_row[2],
                op_number=op_row[3], norm_duration=op_row[4],
                predecessors=preds
            )
            if type_id:
                op.type_id = type_id
            op.quantity = quantity
            op.urgency = urgency
            op.slack_hours = slack
            order.ops.append(op)
        orders.append(order)
    conn.close()
    return orders

def delete_order_from_db(order_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM orders WHERE id=?", (order_id,))
    cur.execute("DELETE FROM operations WHERE order_id=?", (order_id,))
    conn.commit()
    conn.close()

def delete_operation_from_db(op_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM operations WHERE id=?", (op_id,))
    conn.commit()
    conn.close()

def set_order_status(order_id, status):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("UPDATE orders SET status=? WHERE id=?", (status, order_id))
    conn.commit()
    conn.close()

# ----- Resources -----
def save_resources(resources):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM resources")
    for r in resources:
        cur.execute("""
            INSERT INTO resources (id, name, section, status, operator_name,
                                   load_minutes, downtime_minutes, work_hours, repair, reliability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (r.id, r.name, r.section, r.status, r.operator_name,
              r.load_minutes, r.downtime_minutes,
              getattr(r, 'work_hours', 24),
              int(getattr(r, 'repair', 0)),
              float(getattr(r, 'reliability', 1.0))))
    conn.commit()
    conn.close()

def load_resources():
    from datamodels import Resource
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # Гарантируем наличие нужных столбцов (если база старая)
    cur.execute("PRAGMA table_info(resources)")
    cols = [c[1] for c in cur.fetchall()]
    if 'repair' not in cols:
        cur.execute("ALTER TABLE resources ADD COLUMN repair INTEGER DEFAULT 0")
    if 'reliability' not in cols:
        cur.execute("ALTER TABLE resources ADD COLUMN reliability REAL DEFAULT 1.0")
    conn.commit()

    cur.execute("""
        SELECT id, name, section, status, operator_name,
               load_minutes, downtime_minutes, work_hours, repair, reliability
        FROM resources
    """)
    resources = []
    for row in cur:
        r = Resource(id=row[0], name=row[1])
        r.section = row[2] if row[2] else "Основной участок"
        r.status = row[3] if row[3] else "Работает"
        r.operator_name = row[4] if row[4] else ""
        r.load_minutes = float(row[5]) if row[5] else 0.0
        r.downtime_minutes = float(row[6]) if row[6] else 0.0
        r.work_hours = int(row[7]) if row[7] else 24
        r.repair = int(row[8]) if row[8] is not None else 0
        r.reliability = float(row[9]) if row[9] is not None else 1.0
        resources.append(r)
    conn.close()
    return resources

# ----- Остальные функции (сохранены из предыдущей версии) -----
def delete_resource(resource_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM resources WHERE id=?", (resource_id,))
    conn.commit()
    conn.close()

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

def set_resource_operation_types(resource_id, type_ids):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM resource_operation_types WHERE resource_id=?", (resource_id,))
    for tid in type_ids:
        cur.execute("INSERT INTO resource_operation_types (resource_id, operation_type_id) VALUES (?, ?)", (resource_id, tid))
    conn.commit()
    conn.close()

def get_resource_operation_types():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT resource_id, operation_type_id FROM resource_operation_types")
    res_types = {}
    for row in cur:
        res_id, tid = row
        res_types.setdefault(res_id, []).append(tid)
    conn.close()
    return res_types

# алиас для совместимости
get_resource_types = get_resource_operation_types

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

def add_task_to_queue(resource_id, operation_id, position=None):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    if position is None:
        cur.execute("SELECT COALESCE(MAX(position),0)+1 FROM task_queue WHERE resource_id=? AND status='pending'", (resource_id,))
        position = cur.fetchone()[0]
    cur.execute("INSERT OR REPLACE INTO task_queue (resource_id, operation_id, position, status) VALUES (?, ?, ?, 'pending')",
                (resource_id, operation_id, position))
    conn.commit()
    conn.close()

def update_task_status(operation_id, new_status):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("UPDATE task_queue SET status=? WHERE operation_id=?", (new_status, operation_id))
    conn.commit()
    conn.close()

def get_completed_tasks():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            o.operation_id, op.order_id, op.item, op.type_id, o.resource_id, r.name as resource_name
        FROM operation_state_log o
        JOIN operations op ON o.operation_id = op.id
        JOIN resources r ON o.resource_id = r.id
        WHERE o.status = 'completed'
    """)
    tasks = []
    for row in cur:
        tasks.append({
            'operation_id': row[0],
            'order_id': row[1],
            'item': row[2],
            'type_id': row[3],
            'resource_id': row[4],
            'resource_name': row[5]
        })
    conn.close()
    return tasks

def get_tasks_for_resource(resource_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    # Возвращаем только задачи в статусе 'pending' и 'active'
    cur.execute("SELECT operation_id, position, status FROM task_queue WHERE resource_id=? AND status IN ('pending', 'active') ORDER BY position", (resource_id,))
    tasks = [{'operation_id': row[0], 'position': row[1], 'status': row[2]} for row in cur.fetchall()]
    conn.close()
    return tasks

def delete_task_from_queue(operation_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM task_queue WHERE operation_id=?", (operation_id,))
    conn.commit()
    conn.close()

def add_state_type(name, color):
    import uuid
    sid = str(uuid.uuid4())[:8]
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO state_types (id, name, color) VALUES (?, ?, ?)", (sid, name, color))
    conn.commit()
    conn.close()
    return sid

def get_state_types():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT id, name, color FROM state_types")
    return {row[0]: (row[1], row[2]) for row in cur.fetchall()}

def delete_state_type(sid):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM state_types WHERE id=?", (sid,))
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

def get_setting(key, default=None):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT value FROM app_settings WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else default

def set_setting(key, value):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)", (key, str(value)))
    conn.commit()
    conn.close()

def get_all_settings():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM app_settings")
    settings = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return settings

def log_resource_state(resource_id, state_id, source='manual'):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("UPDATE resource_state_log SET end=CURRENT_TIMESTAMP WHERE resource_id=? AND end IS NULL", (resource_id,))
    cur.execute("INSERT INTO resource_state_log (resource_id, state_id, source) VALUES (?, ?, ?)", (resource_id, state_id, source))
    conn.commit()
    conn.close()

def log_operation_state(operation_id, status, resource_id=None, state_id=None, source='manual'):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO operation_state_log (operation_id, status, resource_id, state_id, source) VALUES (?, ?, ?, ?, ?)",
                (operation_id, status, resource_id, state_id, source))
    conn.commit()
    conn.close()

def get_execution_data():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        SELECT o.operation_id, o.resource_id, op.norm_duration,
               (julianday(o.end) - julianday(o.start)) * 1440
        FROM (
            SELECT operation_id, resource_id,
                   MIN(timestamp) as start,
                   MAX(timestamp) as end
            FROM operation_state_log
            WHERE status IN ('active','completed')
            GROUP BY operation_id
        ) o
        JOIN operations op ON o.operation_id = op.id
    """)
    data = []
    for row in cur:
        data.append({'operation_id': row[0], 'resource_id': row[1], 'norm_duration': row[2], 'actual_duration': row[3]})
    conn.close()
    return data

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