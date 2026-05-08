# teacher_window.py
import pickle
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QScrollArea, QFrame, QGridLayout, QMessageBox,
    QStyleFactory
)
from PySide6.QtCore import Qt, QMimeData, QPoint, Signal
from PySide6.QtGui import QFont, QColor, QDrag, QPixmap, QPainter, QPen, QBrush

from database import get_operation_types, get_tasks_for_resource

STYLE_SHEET = """
QMainWindow {
    background-color: #f5f5f5;
    color: #1e1e2e;
}
QLabel {
    color: #1e1e2e;
}
QPushButton {
    background-color: #0078d4;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #005a9e;
}
QFrame#TaskCard {
    background-color: #ffffff;
    border: 1px solid #aaa;
    border-radius: 8px;
}
QFrame#ResourceLine {
    background-color: #89b4fa;
    border: none;
    min-height: 3px;
    max-height: 3px;
}
"""

class TeacherTaskCard(QFrame):
    def __init__(self, op_id, text, parent=None):
        super().__init__(parent)
        self.op_id = op_id
        self.drag_start_pos = QPoint()
        self.setObjectName("TaskCard")
        self.setStyleSheet("TeacherTaskCard { background-color: #ffffff; border: 1px solid #45475a; border-radius: 8px; }")
        self.setMinimumSize(200, 80)
        self.setMaximumSize(200, 80)
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: black; font-weight: bold; background: transparent;")
        layout.addWidget(label)
        self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if (event.position().toPoint() - self.drag_start_pos).manhattanLength() < 10:
                return
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.op_id)
            drag.setMimeData(mime)
            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)
            drag.setHotSpot(event.position().toPoint())
            drag.exec(Qt.MoveAction)

class TeacherDropZone(QWidget):
    action_proposed = Signal(str, str)  # op_id, resource_id

    def __init__(self, resource_id, resource_name, parent=None):
        super().__init__(parent)
        self.resource_id = resource_id
        self.setAcceptDrops(True)
        self.setMinimumSize(220, 200)
        layout = QVBoxLayout(self)
        label = QLabel(f"Очередь: {resource_name}")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-weight: bold; color: #1e1e2e;")
        layout.addWidget(label)
        self.setStyleSheet("TeacherDropZone { border: 2px dashed #aaa; background-color: #f0f0f0; border-radius: 8px; }")

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        op_id = event.mimeData().text()
        self.action_proposed.emit(op_id, self.resource_id)
        event.acceptProposedAction()

class TeacherWindow(QMainWindow):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.setWindowTitle("Обучение с учителем")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet(STYLE_SHEET)

        self.proposals = []  # (ext_state, op_feat, mask, chosen_action)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        btn_train = QPushButton("Обучить агента на моих действиях")
        btn_train.clicked.connect(self.train_agent)
        layout.addWidget(btn_train)

        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.grid_layout = QGridLayout(self.scroll_widget)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        self.rebuild_dashboard()

    def rebuild_dashboard(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        headers = ["Нераспределённые задания"]
        for res in self.core.resources:
            headers.append(f"Очередь {res.name}")
        for col, title in enumerate(headers):
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold;")
            self.grid_layout.addWidget(lbl, 0, col)

        scheduled_ids = set()
        for r in self.core.resources:
            tasks = get_tasks_for_resource(r.id)
            for t in tasks:
                scheduled_ids.add(t['operation_id'])

        type_map = get_operation_types()
        row = 1
        for order in self.core.orders:
            for op in order.ops:
                if op.id not in scheduled_ids:
                    type_name = type_map.get(op.type_id, "—")
                    text = f"Заказ №{order.id.split('_')[-1]}\n{order.name}\n{op.item}\n{op.norm_duration} мин | {type_name}\nКол-во: {op.quantity}"
                    card = TeacherTaskCard(op.id, text)
                    self.grid_layout.addWidget(card, row, 0)
                    row += 1

        for idx, res in enumerate(self.core.resources):
            drop_zone = TeacherDropZone(res.id, res.name)
            drop_zone.action_proposed.connect(self.handle_drop)
            self.grid_layout.addWidget(drop_zone, 1, idx + 1, max(row, 2), 1)

    def handle_drop(self, op_id, resource_id):
        env = self._get_env()
        ready_ops = env.get_ready_ops_list()
        if not ready_ops:
            return
        ext_state, op_feat, mask = self._build_state(env, ready_ops)
        try:
            chosen_idx = next(i for i, op in enumerate(ready_ops) if op['op_id'] == op_id)
        except StopIteration:
            return
        self.proposals.append((ext_state, op_feat, mask, chosen_idx))
        QMessageBox.information(self, "Записано", f"Действие: {op_id} → {resource_id} запомнено.")

    def train_agent(self):
        if not self.proposals:
            QMessageBox.warning(self, "Нет данных", "Сначала перетащите хотя бы одну операцию.")
            return
        from ppo_agent import PPO
        env = self._get_env()
        state_dim = env.state_dim + len(self.core.resources)
        op_feat_dim = 5
        action_dim = env.max_actions
        agent = PPO(state_dim, op_feat_dim, action_dim, device='cpu')
        for (ext_state, op_feat, mask, action) in self.proposals:
            agent.update_single(ext_state, op_feat, mask, action)
        agent.save("ppo_teacher_trained.pth")
        self.proposals.clear()
        QMessageBox.information(self, "Обучение", "Агент обучен и сохранён.")

    def _get_env(self):
        from job_shop_env import JobShopEnv
        return JobShopEnv(self.core.orders, self.core.resources)

    def _build_state(self, env, ready_ops):
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
            due_min = 0; rem_time = 0; weight = 0; progress = 0
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