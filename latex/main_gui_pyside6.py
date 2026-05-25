# main_gui_pyside6.py
import sys, os, json, random, threading, pickle
import numpy as np
from datetime import datetime, timedelta
import torch
from weights import set_model_path, get_model_path, get_last_model_name

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QLineEdit, QFormLayout, QGroupBox, QRadioButton,
    QScrollArea, QFrame, QGridLayout, QSizePolicy, QStyleFactory,
    QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit, QGraphicsView, QGraphicsScene,
    QDialog,  QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal, QThread, QMimeData, QPoint
from PySide6.QtGui import QFont, QColor, QDrag, QPixmap, QPainter, QPen, QBrush, QIcon
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

from PySide6.QtWidgets import QFileDialog   # если ещё нет
from training import TrainingSignals, Trainer
from train_slim import SelfLabelingTrainer

from core import APSCore, PointerNetAgent, SimpleAgent
from database import (
    get_operation_types, get_state_types, get_tasks_for_resource,
    delete_task_from_queue, get_resource_operation_types, get_operation_types, get_completed_tasks
)

STYLE_SHEET = """
QMainWindow {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    background-color: #1e1e2e;
    border-radius: 10px;
}
QTabBar::tab {
    background: #313244;
    color: #cdd6f4;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background: #45475a;
    border-bottom: 2px solid #89b4fa;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
    border: none;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton:pressed {
    background-color: #74c7ec;
}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit {
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #45475a;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 16px;
    color: #cdd6f4;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #89b4fa;
}
QTableWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    gridline-color: #45475a;
    border-radius: 8px;
}
QHeaderView::section {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    padding: 4px;
}
QLabel {
    color: #cdd6f4;
}
QRadioButton {
    color: #cdd6f4;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QFrame#ResourceLine {
    background-color: #89b4fa;
    border: none;
    min-height: 3px;
    max-height: 3px;
}
"""

class PPOTrainThread(QThread):
    progress = Signal(int, float)
    finished = Signal(str)

    def __init__(self, core, episodes, lr, n_orders, n_res, stop_event):
        super().__init__()
        self.core = core
        self.episodes = episodes
        self.lr = lr
        self.n_orders = n_orders
        self.n_res = n_res
        self.stop_event = stop_event          # ← сохраняем

    def run(self):
        try:
            from train_ppo_dispatcher import train_ppo as run_ppo
            run_ppo(episodes=self.episodes, lr=self.lr,
                    n_orders=self.n_orders, n_resources=self.n_res,
                    progress_callback=lambda ep, loss: self.progress.emit(ep, loss),
                    stop_event=self.stop_event)          # ← передаём
            self.finished.emit("ppo_dispatcher.pth")
        except Exception as e:
            self.core.message_signal.emit("error", f"Ошибка PPO: {e}")
            self.finished.emit("")

class TaskCard(QFrame):
    def __init__(self, op_id, text, bg_color="#ffffff", top_strip_color="#cccccc", status="Новый", text_color="black", parent=None):
        super().__init__(parent)
        self.op_id = op_id
        self.drag_start_pos = QPoint()
        self.setObjectName("TaskCard")
        self.setStyleSheet(f"TaskCard {{ background-color: {bg_color}; border: 1px solid #45475a; border-radius: 8px; }}")
        self.setMinimumSize(240, 100)
        self.setMaximumSize(240, 100)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        strip = QFrame()
        strip.setFixedHeight(6)
        strip.setStyleSheet(f"background-color: {top_strip_color}; border: none;")
        layout.addWidget(strip)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {text_color}; font-weight: bold; background: transparent; padding: 4px 6px;")
        layout.addWidget(label)
        self.setCursor(Qt.OpenHandCursor)


    def mouseDoubleClickEvent(self, event):
        main_win = self.window()
        if main_win and hasattr(main_win, 'open_order_editor'):
            main_win.open_order_editor(self.op_id)
        super().mouseDoubleClickEvent(event)

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

class DropZone(QWidget):
    def __init__(self, zone_type, resource_id=None, parent=None):
        super().__init__(parent)
        self.zone_type = zone_type
        self.resource_id = resource_id
        self.setAcceptDrops(True)
        self.setMinimumSize(180, 60)
        self.setLayout(QVBoxLayout())
        self.setStyleSheet("DropZone { border: 1px dashed #89b4fa; background-color: #2a2a3a; }")

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        op_id = event.mimeData().text()
        main_win = self.window()
        if main_win and hasattr(main_win, 'core'):
            if self.zone_type == 'queue' and self.resource_id:
                main_win.core.move_task_to_queue(op_id, self.resource_id)
            elif self.zone_type == 'backlog':
                main_win.core._remove_from_schedules(op_id)
                delete_task_from_queue(op_id)
                main_win.core.dashboard_changed.emit()
        event.acceptProposedAction()

class HorizontalLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("ResourceLine")

class MILPWorker(QThread):
    finished = Signal()
    error = Signal(str)
    def __init__(self, core, fixed=False):
        super().__init__()
        self.core = core
        self.fixed = fixed
    def run(self):
        try:
            if self.fixed:
                self.core.run_fixed_milp()
            else:
                self.core.run_milp()
        except Exception as e:
            self.error.emit(str(e))
        self.finished.emit()

class APSApp(QMainWindow):
    ppo_progress = Signal(int, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("APS Планировщик производства")
        self.setWindowIcon(QIcon("logo.png"))
        self.setGeometry(50, 50, 1400, 800)
        self.setStyleSheet(STYLE_SHEET)

        self.ppo_progress.connect(self.update_ppo_chart)

        self.core = APSCore()
        self.core.dashboard_changed.connect(self.refresh_dashboard)
        self.core.orders_changed.connect(self.refresh_order_table)
        self.core.message_signal.connect(self.show_message)
        self.core.dashboard_changed.connect(self.refresh_completed_table)


        # Восстановление последней модели
        last_model = get_last_model_name()
        if last_model:
            path = get_model_path(last_model)
            if path and os.path.exists(path):
                self.core.loaded_model_name = last_model
                self.core.settings['model_path'] = path

        self.init_ui()
        self.show_dashboard()



        saved_count = len(self.core.orders)
        if saved_count > 0:
            self.statusBar().showMessage(f"Загружено {saved_count} заказов из БД", 5000)

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(self.create_dashboard_tab(), "Дашборд")
        self.tabs.addTab(self.create_orders_tab(), "Заказы")
        self.tabs.addTab(self.create_planning_tab(), "Планирование")
        self.tabs.addTab(self.create_equip_tab(), "Оборудование")
        self.tabs.addTab(self.create_states_tab(), "Состояния")
        self.tabs.addTab(self.create_train_tab(), "Обучение")
        self.tabs.addTab(self.create_settings_tab(), "Настройки")
        self.tabs.addTab(self.create_completed_tab(), "Завершённые")



    # ---------- ДАШБОРД ----------
    def create_dashboard_tab(self):
        dash_tab = QWidget()
        layout = QVBoxLayout(dash_tab)
        top_frame = QHBoxLayout()
        top_frame.addWidget(QLabel("Дата планирования:"))
        self.day_combo = QComboBox()
        dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        self.day_combo.addItems(dates)
        self.day_combo.setCurrentText(self.core.selected_date.strftime("%Y-%m-%d"))
        self.day_combo.currentTextChanged.connect(self.on_date_changed)
        top_frame.addWidget(self.day_combo)
        top_frame.addStretch()
        start_btn = QPushButton("В работу")
        start_btn.clicked.connect(self.core.start_all_queues)
        top_frame.addWidget(start_btn)
        approve_btn = QPushButton("Утвердить текущий план")
        approve_btn.clicked.connect(self.core.approve_plan)
        top_frame.addWidget(approve_btn)
        advise_btn = QPushButton("RL-совет")
        advise_btn.clicked.connect(self.show_rl_advice)
        top_frame.addWidget(advise_btn)
        reset_btn = QPushButton("Сбросить в нераспределённые")
        reset_btn.clicked.connect(self.core.reset_all_queues)
        top_frame.addWidget(reset_btn)
        layout.addLayout(top_frame)

        main_hbox = QHBoxLayout()
        layout.addLayout(main_hbox)

        # Левая панель
        left_frame = QFrame()
        left_frame.setFixedWidth(280)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(5, 5, 5, 5)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.backlog_zone = DropZone('backlog')
        self.backlog_zone.layout().setAlignment(Qt.AlignTop)
        left_scroll.setWidget(self.backlog_zone)
        left_layout.addWidget(left_scroll)
        main_hbox.addWidget(left_frame)

        # Правая панель
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        self.right_widget = QWidget()
        self.grid_layout = QGridLayout(self.right_widget)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        right_scroll.setWidget(self.right_widget)
        right_layout.addWidget(right_scroll)
        main_hbox.addWidget(right_frame, 1)
        self.right_scroll = right_scroll

        return dash_tab

    def refresh_dashboard(self):
        # Очистка backlog
        backlog_layout = self.backlog_zone.layout()
        while backlog_layout.count():
            item = backlog_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Очистка правой сетки
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Заголовки правых колонок
        headers = ["Рабочие центры", "Производство", "Очередь"]
        for col, title in enumerate(headers):
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa; background: transparent; padding: 4px;")
            self.grid_layout.addWidget(lbl, 0, col)

        # Получаем ID операций, которые находятся в очереди или в производстве
        scheduled_ids = set()
        for r in self.core.resources:
            tasks = get_tasks_for_resource(r.id)
            for t in tasks:
                scheduled_ids.add(t['operation_id'])

        # Также добавляем ID операций, которые запланированы (в scheduled), но не назначены на ресурс
        for op_id, info in self.core.current_schedule.items():
            if info['resource_id'] not in self.core.resource_map:
                scheduled_ids.add(op_id)

        type_map = get_operation_types()

        # Заполнение левой панели
        for order in self.core.orders:
            for op in order.ops:
                if op.id not in scheduled_ids:
                    type_name = type_map.get(op.type_id, "Без типа") if op.type_id else "Без типа"
                    text = (f"Заказ №{order.id.split('_')[-1]}\n"
                            f"{op.item}\n"
                            f"Тип: {type_name}\n"
                            f"Срок: {order.due_date.strftime('%d.%m.%y %H:%M')}\n"
                            f"Длит: {op.norm_duration} мин\n"
                            f"Кол-во: {int(op.quantity)}")
                    card = TaskCard(op.id, text, bg_color="#ffffff", top_strip_color="#F9E2AF", status="Новый", text_color="black")
                    backlog_layout.addWidget(card)

        # Переменная для расчёта ширины очереди
        max_queue_width = 250

        # Ресурсы (правая сетка)
        for idx, res in enumerate(self.core.resources):
            grid_row = 1 + idx * 2
            if idx > 0:
                line = HorizontalLine()
                self.grid_layout.addWidget(line, grid_row - 1, 0, 1, 3)

            # Рабочий центр
            # rc_card = QFrame()
            # rc_card.setStyleSheet("background-color: #ffffff; border: 1px solid #45475a; border-radius: 6px;")
            # rc_card.setFixedSize(220, 80)
            # rc_layout = QVBoxLayout(rc_card)
            # rc_label = QLabel(f"{res.name}\n{res.status}")
            # rc_label.setStyleSheet("color: black; font-weight: bold; font-size: 12px;")
            # rc_layout.addWidget(rc_label)

            tasks = get_tasks_for_resource(res.id)
            active_task = next((t for t in tasks if t['status'] == 'active'), None)
            pending_tasks = [t for t in tasks if t['status'] == 'pending']

            # Вместо отдельной кнопки – контейнер с обработкой клика
            # Строим кликабельную карточку ресурса с индикаторами
            rc_container = QFrame()
            rc_container.setStyleSheet(
                "QFrame { background-color: #ffffff; border: 1px solid #45475a; border-radius: 6px; }"
                "QFrame:hover { background-color: #f0f0f0; }"
            )
            rc_container.setFixedSize(220, 120)
            # Двойной клик по карточке открывает информацию о ресурсе
            rc_container.mouseDoubleClickEvent = lambda event, r=res: self.open_resource_info(r)

            rc_layout = QVBoxLayout(rc_container)
            rc_layout.setContentsMargins(8, 8, 8, 8)

            # Название и статус
            rc_name = QLabel(f"<b>{res.name}</b><br>{res.status}")
            rc_name.setStyleSheet("color: black; font-size: 12px;")
            rc_layout.addWidget(rc_name)

            # Индикаторы
            pending_count = len(pending_tasks) if pending_tasks else 0
            has_active = active_task is not None
            reliability = getattr(res, 'reliability', 1.0)

            indic = QLabel(
                f"Готовность: {reliability:.2f}\n"
                f"В очереди: {pending_count}\n"
                f"Активен: {'да' if has_active else 'нет'}"
            )
            indic.setStyleSheet("color: black; font-size: 10px;")
            rc_layout.addWidget(indic)

            self.grid_layout.addWidget(rc_container, grid_row, 0, alignment=Qt.AlignTop)



            # Производство
            prod_zone = QWidget()
            prod_layout = QVBoxLayout(prod_zone)
            if active_task:
                op_id = active_task['operation_id']
                op = self.core._get_operation_by_id(op_id)
                if op:
                    order = self.core._get_order_by_op(op_id)
                    type_name = type_map.get(op.type_id, "Без типа") if op.type_id else "Без типа"
                    text = (f"Заказ №{order.id.split('_')[-1]}\n"
                            f"{op.item}\n"
                            f"Тип: {type_name}\n"
                            f"Срок: {order.due_date.strftime('%d.%m.%y %H:%M')}\n"
                            f"Длит: {op.norm_duration} мин\n"
                            f"Кол-во: {int(op.quantity)}")
                    card = TaskCard(op_id, text, bg_color="#ffffff", top_strip_color="#A6E3A1", status="В работе", text_color="black")
                    btn = QPushButton("Завершить")
                    btn.clicked.connect(lambda checked=False, oid=op_id: self.core.complete_task(oid))
                    prod_layout.addWidget(card)
                    prod_layout.addWidget(btn)
            self.grid_layout.addWidget(prod_zone, grid_row, 1, alignment=Qt.AlignTop)

            # Очередь – горизонтальный контейнер
            # Для каждого ресурса
            queue_zone = DropZone('queue', resource_id=res.id)
            # Удаляем старый layout, если был
            if queue_zone.layout():
                QWidget().setLayout(queue_zone.layout())
            # Создаём новый горизонтальный layout
            hor_layout = QHBoxLayout(queue_zone)
            hor_layout.setAlignment(Qt.AlignLeft)
            hor_layout.setContentsMargins(5, 5, 5, 5)
            hor_layout.setSpacing(8)

            for pt in pending_tasks:
                op_id = pt['operation_id']
                op = self.core._get_operation_by_id(op_id)
                if op:
                    order = self.core._get_order_by_op(op_id)
                    type_name = type_map.get(op.type_id, "Без типа") if op.type_id else "Без типа"
                    text = (f"Заказ №{order.id.split('_')[-1]}\n"
                            f"{op.item}\n"
                            f"Тип: {type_name}\n"
                            f"Срок: {order.due_date.strftime('%d.%m.%y %H:%M')}\n"
                            f"Длит: {op.norm_duration} мин\n"
                            f"Кол-во: {int(op.quantity)}")
                    card = TaskCard(op_id, text, bg_color="#ffffff", top_strip_color="#89B4FA", status="Запущен", text_color="black")
                    hor_layout.addWidget(card)

            # Вычисляем ширину очереди
            queue_width = max(250, len(pending_tasks) * (240 + 8) + 10)
            if queue_width > max_queue_width:
                max_queue_width = queue_width

            self.grid_layout.addWidget(queue_zone, grid_row, 2, alignment=Qt.AlignTop)

        # Нижняя линия
        if self.core.resources:
            last_data_row = 1 + (len(self.core.resources) - 1) * 2
            line_bottom = HorizontalLine()
            self.grid_layout.addWidget(line_bottom, last_data_row + 1, 0, 1, 3)

        self.grid_layout.setRowStretch(self.grid_layout.rowCount(), 1)

        # Устанавливаем минимальную ширину правой панели
        min_width = 220 + 220 + max_queue_width + 40
        self.right_widget.setMinimumWidth(min_width)

    def open_resource_info(self, resource):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Ресурс: {resource.name}")
        dlg.resize(350, 250)
        layout = QFormLayout(dlg)

        layout.addRow("ID:", QLabel(resource.id))
        layout.addRow("Статус:", QLabel(resource.status))

        reliability_edit = QDoubleSpinBox()
        reliability_edit.setRange(0.0, 1.0)
        reliability_edit.setSingleStep(0.05)
        reliability_edit.setValue(getattr(resource, 'reliability', 1.0))
        layout.addRow("Готовность (0-1):", reliability_edit)

        repair_check = QCheckBox("Ресурс в ремонте")
        repair_check.setChecked(getattr(resource, 'repair', 0) == 1)
        layout.addRow("Ремонт:", repair_check)

        tasks = get_tasks_for_resource(resource.id)
        active = next((t for t in tasks if t['status'] == 'active'), None)
        pending_count = sum(1 for t in tasks if t['status'] == 'pending')
        layout.addRow("Заказов в очереди:", QLabel(str(pending_count)))
        layout.addRow("Активный заказ:", QLabel(active['operation_id'] if active else "нет"))

        # Кнопки Сохранить / Отмена
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        cancel_btn = QPushButton("Отмена")
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(btn_layout)

        def save_changes():
            resource.reliability = reliability_edit.value()
            resource.repair = 1 if repair_check.isChecked() else 0
            from database import save_resources
            save_resources(self.core.resources)  # пишем в БД
            self.core.dashboard_changed.emit()  # обновим индикаторы на дашборде
            self.statusBar().showMessage(f"Ресурс {resource.name} сохранён", 3000)
            dlg.accept()

        save_btn.clicked.connect(save_changes)
        cancel_btn.clicked.connect(dlg.reject)

        dlg.exec()

    def show_dashboard(self):
        self.tabs.setCurrentIndex(0)
        self.refresh_dashboard()

    def on_date_changed(self, text):
        try:
            dt = datetime.strptime(text, "%Y-%m-%d")
            self.core.selected_date = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            self.refresh_dashboard()
        except:
            pass

    def run_rl_plan(self):
        self.core.plan_with_reliability()
        self.core.start_all_queues()  # <-- сразу запускаем очереди
        self.statusBar().showMessage("План построен", 5000)
        if self.core.current_schedule:
            self.draw_milp_plan()
        self.show_dashboard()

    # ---------- RL-совет ----------
    def show_rl_advice(self):
        # Если агент не загружен – эвристический совет
        if not self.core.rl_agent:
            self._heuristic_advice()
            return

        # Находим первую неразмещённую операцию
        unscheduled = []
        for order in self.core.orders:
            for op in order.ops:
                if op.id not in self.core.current_schedule and op.id not in self.core.approved_schedule:
                    unscheduled.append((order, op))
                    break
            if unscheduled:
                break
        if not unscheduled:
            QMessageBox.information(self, "RL‑совет", "Нет неразмещённых операций.")
            return

        order, op = unscheduled[0]

        # Допустимые ресурсы для операции
        if not hasattr(self.core, 'cached_res_op_types'):
            self.core.cached_res_op_types = get_resource_operation_types()
        allowed_rids = self.core.cached_res_op_types.get(op.type_id, []) if op.type_id else list(
            self.core.resource_map.keys())
        if not allowed_rids:
            allowed_rids = list(self.core.resource_map.keys())

        try:
            if isinstance(self.core.rl_agent, PointerNetAgent):
                # Используем старый метод для PointerNetAgent (три аргумента)
                res_feat, op_feat, mask = self.core._build_pointer_input(None,
                                                                         [{'op_id': op.id, 'resource_id': r} for r in
                                                                          allowed_rids])
                action, _ = self.core.rl_agent.select_action(res_feat, op_feat, mask)
                chosen_res = allowed_rids[action] if action < len(allowed_rids) else allowed_rids[0]
            elif isinstance(self.core.rl_agent, SimpleAgent):
                # Для SimpleAgent строим плоский вектор (36 признаков) и вызываем select_action(X)
                full_vec = self._build_flat_state_for_advice(order, op, allowed_rids)
                action_idx = self.core.rl_agent.select_action(full_vec)
                chosen_res = self.core.resources[action_idx].id if action_idx < len(self.core.resources) else \
                allowed_rids[0]
            else:
                self._heuristic_advice()
                return
        except Exception as e:
            QMessageBox.warning(self, "Ошибка совета", str(e))
            self._heuristic_advice()
            return

        QMessageBox.information(self, "RL‑совет",
                                f"Операция: {op.id} (заказ {order.id})\n"
                                f"Нейросеть рекомендует ресурс: {chosen_res} "
                                f"(готовность {self.core.resource_map[chosen_res].reliability:.2f})")

    def _build_flat_state_for_advice(self, order, op, allowed_rids):
        """Строит плоский вектор признаков (36 чисел) для SimpleAgent."""
        import numpy as np
        type_map = get_operation_types()
        type_list = ['фрезеровка', 'сборка', 'сварка']
        type_to_idx = {name: i for i, name in enumerate(type_list)}

        # Признаки операции (6)
        type_onehot = np.zeros(3)
        if op.type_id:
            tname = type_map.get(op.type_id, '')
            if tname in type_to_idx:
                type_onehot[type_to_idx[tname]] = 1.0
        op_feat = np.array([
            *type_onehot,
            op.norm_duration / 120.0,
            order.priority_weight,
            (op.slack_hours if hasattr(op, 'slack_hours') else 0) / 48.0
        ])

        # Признаки ресурсов (R * 6)
        R = len(self.core.resources)
        res_feats = []
        for i, res in enumerate(self.core.resources):
            cur_load = 0.0  # упрощённо, т.к. это совет
            compat = 1.0 if res.id in allowed_rids else 0.0
            res_vec = np.array([
                1.0 if res.status == 'Работает' else 0.0,
                cur_load,
                compat,
                res.load_minutes / 240.0,
                float(res.repair),
                res.reliability
            ])
            res_feats.append(res_vec)
        return np.concatenate([op_feat] + res_feats)

    def _heuristic_advice(self):
        """Эвристический совет, когда модель не загружена."""
        unscheduled = []
        for order in self.core.orders:
            for op in order.ops:
                if op.id not in self.core.current_schedule and op.id not in self.core.approved_schedule:
                    unscheduled.append((order, op))
                    break
            if unscheduled:
                break
        if not unscheduled:
            QMessageBox.information(self, "Совет", "Нет неразмещённых операций.")
            return

        if not hasattr(self.core, 'cached_res_op_types'):
            self.core.cached_res_op_types = get_resource_operation_types()

        order, op = unscheduled[0]
        if op.type_id:
            allowed_rids = self.core.cached_res_op_types.get(op.type_id, [])
        else:
            allowed_rids = list(self.core.resource_map.keys())
        if not allowed_rids:
            allowed_rids = list(self.core.resource_map.keys())

        now = datetime.now()
        work_until = {r.id: now for r in self.core.resources}

        def effective_load(rid):
            rel = self.core.resource_map[rid].reliability
            if rel <= 0:
                return float('inf')
            if self.core.resource_map[rid].repair:
                rel *= 0.5
            workload = (work_until[rid] - now).total_seconds() / 60.0
            return workload / rel

        best_rid = min(allowed_rids, key=effective_load)
        QMessageBox.information(self, "Совет",
                                f"Для операции {op.id} (заказ {order.id})\n"
                                f"Рекомендуется ресурс: {best_rid} "
                                f"(готовность {self.core.resource_map[best_rid].reliability:.2f})")


    # ---------- ЗАКАЗЫ ----------
    def create_orders_tab(self):
        orders_tab = QWidget()
        layout = QVBoxLayout(orders_tab)
        form_group = QGroupBox("Новый заказ")
        form_layout = QFormLayout()
        self.order_name_edit = QLineEdit()
        form_layout.addRow("Название заказа:", self.order_name_edit)
        self.order_num_edit = QLineEdit()
        form_layout.addRow("Номер заказа (необязательно):", self.order_num_edit)
        self.due_edit = QLineEdit("2026-05-10 08:00")
        form_layout.addRow("Дата сдачи:", self.due_edit)
        self.prio_spin = QDoubleSpinBox();
        self.prio_spin.setRange(0.5, 2.0);
        self.prio_spin.setValue(1.0)
        form_layout.addRow("Приоритет:", self.prio_spin)

        # Фиксированная единственная операция
        types = get_operation_types()
        self.op_type_combo = QComboBox()
        self.op_type_combo.addItems(types.values() if types else ["Нет типов"])
        form_layout.addRow("Тип операции:", self.op_type_combo)

        self.op_dur_edit = QLineEdit("60")
        form_layout.addRow("Длительность (мин):", self.op_dur_edit)

        self.op_item_edit = QLineEdit("Деталь_1")
        form_layout.addRow("Изделие:", self.op_item_edit)

        self.op_qty_edit = QLineEdit("1")
        form_layout.addRow("Кол-во:", self.op_qty_edit)

        # Удаляем все старые динамические строки (ops_layout, add_operation_row) – больше не нужны

        save_btn = QPushButton("Сохранить заказ")
        save_btn.clicked.connect(self.save_order)
        form_layout.addRow(save_btn)
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        gen_btn = QPushButton("Сгенерировать 10 случайных заказов")
        gen_btn.clicked.connect(lambda: self.core.generate_random_orders(10))
        layout.addWidget(gen_btn)
        load_erp_btn = QPushButton("Загрузить из ERP (JSON)")
        load_erp_btn.clicked.connect(
            lambda: self.core.load_erp_orders(self.core.settings.get('erp_file', 'erp_orders.json')))
        layout.addWidget(load_erp_btn)

        self.order_table = QTableWidget()
        self.order_table.setColumnCount(6)
        self.order_table.setHorizontalHeaderLabels(["Заказ", "Название", "Срок", "Операций", "Приоритет", "Действия"])
        self.order_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.order_table)
        self.refresh_order_table()
        return orders_tab

    def add_operation_row(self):
        row = QHBoxLayout()
        types = get_operation_types()
        type_combo = QComboBox()
        type_combo.addItems(types.values() if types else ["Нет типов"])
        row.addWidget(QLabel("Тип:")); row.addWidget(type_combo)
        dur_edit = QLineEdit("60"); dur_edit.setFixedWidth(80)
        row.addWidget(QLabel("Длит.(мин):")); row.addWidget(dur_edit)
        item_edit = QLineEdit("Деталь_1"); item_edit.setFixedWidth(120)
        row.addWidget(QLabel("Изделие:")); row.addWidget(item_edit)
        qty_edit = QLineEdit("1"); qty_edit.setFixedWidth(50)
        row.addWidget(QLabel("Кол-во:")); row.addWidget(qty_edit)
        if not hasattr(self, 'ops_widgets'): self.ops_widgets = []
        self.ops_widgets.append((type_combo, dur_edit, item_edit, qty_edit))
        self.ops_layout.addLayout(row)

    def open_order_editor(self, op_id):
        order = self.core._get_order_by_op(op_id)
        if not order:
            return
        dlg = OrderEditDialog(order, self.core, self)
        if dlg.exec():
            self.core.orders_changed.emit()
            self.core.dashboard_changed.emit()

    def save_order(self):
        try:
            name = self.order_name_edit.text().strip()
            order_num = self.order_num_edit.text().strip()
            due = datetime.strptime(self.due_edit.text(), "%Y-%m-%d %H:%M")
            prio = self.prio_spin.value()
            type_map = get_operation_types()
            if not type_map:
                QMessageBox.critical(self, "Ошибка", "Сначала добавьте типы операций на вкладке «Оборудование → Типы».")
                return
            type_name = self.op_type_combo.currentText()
            type_id = next((tid for tid, tn in type_map.items() if tn == type_name), None)
            if type_id is None:
                QMessageBox.critical(self, "Ошибка", "Выберите корректный тип операции.")
                return
            ops = [{
                'item': self.op_item_edit.text(),
                'duration': float(self.op_dur_edit.text()),
                'type_id': type_id,
                'quantity': int(self.op_qty_edit.text()) if self.op_qty_edit.text() else 1
            }]
            self.core.create_order(due, prio, name, order_num, ops)
            self.refresh_dashboard()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def refresh_order_table(self):
        self.order_table.setRowCount(len(self.core.orders))
        for i, order in enumerate(self.core.orders):
            self.order_table.setItem(i, 0, QTableWidgetItem(order.id))
            self.order_table.setItem(i, 1, QTableWidgetItem(order.name))
            self.order_table.setItem(i, 2, QTableWidgetItem(order.due_date.strftime("%Y-%m-%d %H:%M")))
            self.order_table.setItem(i, 3, QTableWidgetItem(str(len(order.ops))))
            self.order_table.setItem(i, 4, QTableWidgetItem(f"{order.priority_weight:.2f}"))
            btn_delete = QPushButton("Удалить")
            btn_delete.clicked.connect(lambda checked=False, oid=order.id: self.core.delete_order(oid))
            self.order_table.setCellWidget(i, 5, btn_delete)

    # ---------- ПЛАНИРОВАНИЕ ----------
    def create_planning_tab(self):
        planning_tab = QWidget()
        layout = QVBoxLayout(planning_tab)
        btn_layout = QHBoxLayout()

        # Основная быстрая кнопка – RL‑план
        rl_plan_btn = QPushButton("Построить план (RL)")
        rl_plan_btn.clicked.connect(self.run_rl_plan)
        btn_layout.addWidget(rl_plan_btn)

        # Кнопка точного MILP (медленного)
        milp_btn = QPushButton("Точный MILP‑план (медленно)")
        milp_btn.clicked.connect(self.run_milp)
        btn_layout.addWidget(milp_btn)

        # Существующая кнопка «Применить фиксированные и перестроить»
        fixed_btn = QPushButton("Применить фиксированные и перестроить")
        fixed_btn.clicked.connect(self.run_fixed_milp)
        btn_layout.addWidget(fixed_btn)

        layout.addLayout(btn_layout)

        self.planning_scene = QGraphicsScene()
        self.planning_view = QGraphicsView(self.planning_scene)
        self.planning_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.planning_view)
        return planning_tab

    def run_milp(self):
        if not self.core.orders:
            QMessageBox.critical(self, "Ошибка", "Сначала создайте заказы.")
            return
        self.core.plan_with_reliability()
        self.statusBar().showMessage("План построен", 5000)
        if self.core.current_schedule:
            self.draw_milp_plan()
        self.show_dashboard()

    def run_fixed_milp(self):
        self.statusBar().showMessage("Перестроение с учётом фиксированных...")
        self.milp_thread = MILPWorker(self.core, fixed=True)
        self.milp_thread.finished.connect(self.on_milp_finished)
        self.milp_thread.error.connect(lambda msg: QMessageBox.critical(self, "Ошибка MILP", msg))
        self.milp_thread.start()

    def on_milp_finished(self):
        self.statusBar().showMessage("План построен", 5000)
        if self.core.current_schedule:
            self.day_combo.setCurrentText(self.core.selected_date.strftime("%Y-%m-%d"))
            self.draw_milp_plan()
        self.show_dashboard()

    def draw_milp_plan(self):
        self.planning_scene.clear()
        if not self.core.current_schedule:
            return
        valid_sched = {op_id: info for op_id, info in self.core.current_schedule.items()
                       if info['resource_id'] in self.core.resource_map}
        if not valid_sched:
            return

        type_map = get_operation_types()
        # Цвета для разных типов
        type_colors = {
            'фрезеровка': QColor("#A6E3A1"),  # светло-зелёный
            'сборка': QColor("#89B4FA"),  # голубой
            'сварка': QColor("#FAB387")  # оранжевый
        }
        default_color = QColor("#CBA6F7")  # фиолетовый для неопознанных

        # Определяем временной диапазон
        min_time = min(v['start'] for v in valid_sched.values())
        max_time = max(v['end'] for v in valid_sched.values())
        total_seconds = (max_time - min_time).total_seconds()
        if total_seconds <= 0:
            return

        width = 800
        y_step = 40
        used_rids = sorted(set(info['resource_id'] for info in valid_sched.values()))
        resource_y = {rid: 20 + i * y_step for i, rid in enumerate(used_rids)}

        # Названия ресурсов (слева)
        for rid, y in resource_y.items():
            name = self.core.resource_map[rid].name
            text = self.planning_scene.addText(name)
            text.setPos(5, y)
            text.setDefaultTextColor(QColor("#cdd6f4"))
            font = text.font()
            font.setPointSize(9)
            text.setFont(font)

        # Временная шкала (упрощённая)
        for i in range(0, int(total_seconds / 3600) + 1, 2):  # каждые 2 часа
            x = (i * 3600) / total_seconds * (width - 150) + 150
            tick = self.planning_scene.addLine(x, 5, x, 5 + len(used_rids) * y_step, QPen(QColor("#45475a")))
            time_label = self.planning_scene.addText(f"{int(min_time.hour) + i}:00")
            time_label.setPos(x - 15, 0)
            time_label.setDefaultTextColor(QColor("#bac2de"))

        # Блоки операций
        for op_id, info in valid_sched.items():
            op = self.core._get_operation_by_id(op_id)
            if not op:
                continue
            x1 = (info['start'] - min_time).total_seconds() / total_seconds * (width - 150) + 150
            x2 = (info['end'] - min_time).total_seconds() / total_seconds * (width - 150) + 150
            y = resource_y[info['resource_id']] + 15

            # Цвет по типу операции
            tname = type_map.get(op.type_id, '')
            color = type_colors.get(tname, default_color)

            rect = self.planning_scene.addRect(x1, y - 10, x2 - x1, 20, QPen(Qt.black), QBrush(color))
            # Подпись (op_id без префикса заказа)
            short_id = op.id.split('_')[-1]  # например, "оп1"
            lbl = self.planning_scene.addText(short_id)
            lbl.setPos(x1 + 2, y - 10)
            lbl.setDefaultTextColor(QColor("#1e1e2e"))
            font = lbl.font()
            font.setPointSize(7)
            lbl.setFont(font)

        # Легенда
        legend_y = 5 + len(used_rids) * y_step + 20
        legend_x = 150
        for i, (tname, color) in enumerate(type_colors.items()):
            self.planning_scene.addRect(legend_x + i * 80, legend_y, 15, 15, QPen(Qt.black), QBrush(color))
            t = self.planning_scene.addText(tname)
            t.setPos(legend_x + i * 80 + 18, legend_y - 2)
            t.setDefaultTextColor(QColor("#cdd6f4"))

    # ---------- ОБОРУДОВАНИЕ ----------
    def create_equip_tab(self):
        equip_tab = QWidget()
        layout = QVBoxLayout(equip_tab)
        sub_tabs = QTabWidget()
        res_tab = QWidget(); types_tab = QWidget(); links_tab = QWidget()
        sub_tabs.addTab(res_tab, "Ресурсы"); sub_tabs.addTab(types_tab, "Типы"); sub_tabs.addTab(links_tab, "Связи")
        layout.addWidget(sub_tabs)

        # Ресурсы
        rl = QVBoxLayout(res_tab)
        self.res_list = QTextEdit(); rl.addWidget(self.res_list)
        hr = QHBoxLayout()
        self.res_id_edit = QLineEdit(); self.res_id_edit.setPlaceholderText("ID ресурса")
        self.res_name_edit = QLineEdit(); self.res_name_edit.setPlaceholderText("Название")
        btn_add = QPushButton("Добавить"); btn_add.clicked.connect(self.add_resource)
        hr.addWidget(self.res_id_edit); hr.addWidget(self.res_name_edit); hr.addWidget(btn_add)
        rl.addLayout(hr)
        hr2 = QHBoxLayout()
        self.del_res_combo = QComboBox(); self.del_res_combo.addItems([r.id for r in self.core.resources])
        btn_del = QPushButton("Удалить"); btn_del.clicked.connect(self.delete_resource)
        hr2.addWidget(QLabel("Выберите ID:")); hr2.addWidget(self.del_res_combo); hr2.addWidget(btn_del)
        rl.addLayout(hr2)
        self.refresh_res_list()

        # Типы
        tl = QVBoxLayout(types_tab)
        self.types_list = QTextEdit(); tl.addWidget(self.types_list)
        ht = QHBoxLayout()
        self.type_name_edit = QLineEdit(); self.type_name_edit.setPlaceholderText("Название типа")
        btn_tadd = QPushButton("Добавить"); btn_tadd.clicked.connect(self.add_type)
        ht.addWidget(self.type_name_edit); ht.addWidget(btn_tadd)
        tl.addLayout(ht)
        ht2 = QHBoxLayout()
        self.del_type_combo = QComboBox(); self.del_type_combo.addItems(list(get_operation_types().keys()))
        btn_tdel = QPushButton("Удалить"); btn_tdel.clicked.connect(self.delete_type)
        ht2.addWidget(QLabel("Удалить тип:")); ht2.addWidget(self.del_type_combo); ht2.addWidget(btn_tdel)
        tl.addLayout(ht2)
        self.refresh_types_list()

        # Связи
        links_tab = QWidget()
        ll = QVBoxLayout(links_tab)

        # Верхняя панель: выбор ресурса и типа + кнопка "Добавить"
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Ресурс:"))
        self.link_res_combo = QComboBox()
        self.link_res_combo.addItems([r.id for r in self.core.resources])
        top_row.addWidget(self.link_res_combo)

        top_row.addWidget(QLabel("Тип:"))
        self.link_type_combo = QComboBox()
        self.link_type_combo.addItems(list(get_operation_types().values()))
        top_row.addWidget(self.link_type_combo)

        btn_add_link = QPushButton("Добавить связь")
        btn_add_link.clicked.connect(self.add_resource_type_link)
        top_row.addWidget(btn_add_link)
        ll.addLayout(top_row)

        # Таблица текущих связей
        self.links_table = QTableWidget()
        self.links_table.setColumnCount(3)
        self.links_table.setHorizontalHeaderLabels(["Ресурс", "Тип операции", "Действия"])
        self.links_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        ll.addWidget(self.links_table)
        self.refresh_links_table()
        return equip_tab

    def add_resource_type_link(self):
        rid = self.link_res_combo.currentText()
        type_name = self.link_type_combo.currentText()
        type_map = get_operation_types()
        type_id = next((tid for tid, tname in type_map.items() if tname == type_name), None)
        if not type_id:
            QMessageBox.critical(self, "Ошибка", "Выберите тип операции.")
            return
        # Получаем текущие связи для ресурса
        current = get_resource_operation_types().get(rid, [])
        if type_id in current:
            QMessageBox.information(self, "Инфо", "Связь уже существует.")
            return
        current.append(type_id)
        self.core.set_resource_types_core(rid, current)
        self.refresh_links_table()
        QMessageBox.information(self, "Успех", f"Связь {rid} → {type_name} добавлена.")

    def refresh_links_table(self):
        res_types = get_resource_operation_types()
        type_map = get_operation_types()
        rows = []
        for rid, tid_list in res_types.items():
            for tid in tid_list:
                tname = type_map.get(tid, tid)
                rows.append((rid, tname, tid))
        self.links_table.setRowCount(len(rows))
        for i, (rid, tname, tid) in enumerate(rows):
            self.links_table.setItem(i, 0, QTableWidgetItem(rid))
            self.links_table.setItem(i, 1, QTableWidgetItem(tname))
            btn_del = QPushButton("Удалить")
            btn_del.clicked.connect(lambda checked=False, r=rid, t=tid: self.delete_resource_type_link(r, t))
            self.links_table.setCellWidget(i, 2, btn_del)

    def delete_resource_type_link(self, rid, tid):
        current = get_resource_operation_types().get(rid, [])
        if tid in current:
            current.remove(tid)
            self.core.set_resource_types_core(rid, current)
            self.refresh_links_table()
            QMessageBox.information(self, "Успех", "Связь удалена.")

    def refresh_res_list(self):
        self.res_list.clear()
        for r in self.core.resources:
            self.res_list.append(f"{r.id}: {r.name}")

    def refresh_types_list(self):
        self.types_list.clear()
        for tid, tn in get_operation_types().items():
            self.types_list.append(f"{tid}: {tn}")

    def add_resource(self):
        rid = self.res_id_edit.text().strip(); name = self.res_name_edit.text().strip()
        if not rid or not name: QMessageBox.critical(self, "Ошибка", "Введите ID и название."); return
        self.core.add_resource(rid, name)
        self.refresh_res_list()
        self.del_res_combo.addItem(rid)
        QMessageBox.information(self, "Успех", f"Ресурс {name} добавлен.")

    def delete_resource(self):
        rid = self.del_res_combo.currentText()
        if rid not in self.core.resource_map: QMessageBox.critical(self, "Ошибка", "Ресурс не найден."); return
        self.core.delete_resource(rid)
        self.refresh_res_list()
        self.del_res_combo.removeItem(self.del_res_combo.findText(rid))
        QMessageBox.information(self, "Успех", f"Ресурс {rid} удалён.")

    def add_type(self):
        name = self.type_name_edit.text().strip()
        if not name: QMessageBox.critical(self, "Ошибка", "Введите название типа."); return
        tid = self.core.add_operation_type_core(name)
        self.refresh_types_list()
        self.del_type_combo.addItem(tid)
        QMessageBox.information(self, "Успех", f"Тип '{name}' добавлен.")

    def delete_type(self):
        tid = self.del_type_combo.currentText()
        if not tid: return
        self.core.delete_operation_type_core(tid)
        self.refresh_types_list()
        self.del_type_combo.removeItem(self.del_type_combo.findText(tid))
        QMessageBox.information(self, "Успех", "Тип удалён.")

    def save_resource_types(self):
        rid = self.link_res_combo.currentText()
        selected = [tid for tid, cb in self.link_type_checkboxes.items() if cb.isChecked()]
        self.core.set_resource_types_core(rid, selected)
        QMessageBox.information(self, "Успех", f"Типы для ресурса {rid} сохранены.")

    # ---------- СОСТОЯНИЯ ----------
    def create_states_tab(self):
        states_tab = QWidget()
        layout = QVBoxLayout(states_tab)
        sub_tabs = QTabWidget()
        st_tab = QWidget(); dt_tab = QWidget()
        sub_tabs.addTab(st_tab, "Типы состояний"); sub_tabs.addTab(dt_tab, "Простои")
        layout.addWidget(sub_tabs)

        sl = QVBoxLayout(st_tab)
        self.state_list = QTextEdit(); sl.addWidget(self.state_list)
        hst = QHBoxLayout()
        self.state_name_edit = QLineEdit(); self.state_name_edit.setPlaceholderText("Название")
        self.state_color_edit = QLineEdit(); self.state_color_edit.setPlaceholderText("Цвет (hex)")
        btn_sadd = QPushButton("Добавить"); btn_sadd.clicked.connect(self.add_state_type)
        hst.addWidget(self.state_name_edit); hst.addWidget(self.state_color_edit); hst.addWidget(btn_sadd)
        sl.addLayout(hst)
        hst2 = QHBoxLayout()
        self.del_state_combo = QComboBox(); self.del_state_combo.addItems(list(self.core.state_types.keys()))
        btn_sdel = QPushButton("Удалить"); btn_sdel.clicked.connect(self.del_state_type)
        hst2.addWidget(QLabel("Удалить состояние:")); hst2.addWidget(self.del_state_combo); hst2.addWidget(btn_sdel)
        sl.addLayout(hst2)
        self.refresh_state_list()

        dl = QVBoxLayout(dt_tab)
        form = QFormLayout()
        self.dt_res_combo = QComboBox(); self.dt_res_combo.addItems([r.id for r in self.core.resources])
        form.addRow("Ресурс:", self.dt_res_combo)
        self.dt_state_combo = QComboBox(); self.dt_state_combo.addItems(list(self.core.state_types.keys()))
        form.addRow("Тип состояния:", self.dt_state_combo)
        self.dt_start_edit = QLineEdit("2026-04-27 10:00")
        form.addRow("Начало:", self.dt_start_edit)
        self.dt_end_edit = QLineEdit("2026-04-27 12:00")
        form.addRow("Конец:", self.dt_end_edit)
        btn_dadd = QPushButton("Добавить простой"); btn_dadd.clicked.connect(self.add_downtime)
        form.addRow(btn_dadd)
        dl.addLayout(form)
        self.dt_list = QTextEdit(); dl.addWidget(self.dt_list)
        btn_clr = QPushButton("Удалить все простои для ресурса"); btn_clr.clicked.connect(self.clear_downtimes)
        dl.addWidget(btn_clr)
        self.refresh_downtime_list()
        return states_tab

    def refresh_state_list(self):
        self.state_list.clear()
        for sid, (name, color) in self.core.state_types.items():
            self.state_list.append(f"{sid}: {name} ({color})")

    def add_state_type(self):
        name = self.state_name_edit.text().strip(); color = self.state_color_edit.text().strip()
        if not name or not color: QMessageBox.critical(self, "Ошибка", "Введите название и цвет."); return
        sid = self.core.add_state_type_core(name, color)
        self.refresh_state_list()
        self.del_state_combo.addItem(sid)
        self.dt_state_combo.addItem(sid)
        QMessageBox.information(self, "Успех", f"Состояние '{name}' добавлено.")

    def del_state_type(self):
        sid = self.del_state_combo.currentText()
        if not sid: return
        self.core.delete_state_type_core(sid)
        self.refresh_state_list()
        self.del_state_combo.removeItem(self.del_state_combo.findText(sid))
        self.dt_state_combo.removeItem(self.dt_state_combo.findText(sid))
        QMessageBox.information(self, "Успех", "Состояние удалено.")

    def add_downtime(self):
        rid = self.dt_res_combo.currentText(); state_id = self.dt_state_combo.currentText()
        try:
            start = datetime.strptime(self.dt_start_edit.text(), "%Y-%m-%d %H:%M")
            end = datetime.strptime(self.dt_end_edit.text(), "%Y-%m-%d %H:%M")
            if end <= start: raise ValueError("Конец должен быть позже начала.")
        except Exception as e: QMessageBox.critical(self, "Ошибка", str(e)); return
        self.core.add_downtime_core(rid, start, end, state_id)
        self.refresh_downtime_list()
        QMessageBox.information(self, "Успех", f"Простой для {rid} добавлен.")

    def clear_downtimes(self):
        rid = self.dt_res_combo.currentText()
        self.core.clear_downtimes_core(rid)
        self.refresh_downtime_list()
        QMessageBox.information(self, "Успех", f"Все простои для {rid} удалены.")

    def refresh_downtime_list(self):
        self.dt_list.clear()
        for rid, intervals in self.core.downtimes.items():
            name = self.core.resource_map.get(rid, None)
            for s, e, sid in intervals:
                state_name = self.core.state_types.get(sid, ('?',''))[0]
                self.dt_list.append(f"{name.name if name else rid}: {s.strftime('%d.%m.%y %H:%M')} - {e.strftime('%d.%m.%y %H:%M')} [{state_name}]")

    # ---------- ОБУЧЕНИЕ ----------
    def create_train_tab(self):
        train_tab = QWidget()
        layout = QVBoxLayout(train_tab)
        sub_tabs = QTabWidget()

        # --- Подвкладка "Текущая НС" ---
        current_tab = QWidget()
        current_layout = QVBoxLayout(current_tab)
        self.model_status_label = QLabel(
            f"Загружена модель: {self.core.loaded_model_name if self.core.loaded_model_name else 'нет'}")
        self.model_status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        current_layout.addWidget(self.model_status_label)

        load_btn = QPushButton("Загрузить модель")
        load_btn.clicked.connect(self.load_model)
        current_layout.addWidget(load_btn)
        sub_tabs.addTab(current_tab, "Текущая НС")

        # --- Подвкладка "Обучение НС" ---
        training_tab = QWidget()
        train_layout = QVBoxLayout(training_tab)

        # Общие настройки обучения
        settings_group = QGroupBox("Параметры обучения")
        settings_form = QFormLayout()
        self.train_epoch_spin = QSpinBox()
        self.train_epoch_spin.setRange(1, 5000)
        self.train_epoch_spin.setValue(50)
        settings_form.addRow("Эпох:", self.train_epoch_spin)

        self.train_lr_spin = QDoubleSpinBox()
        self.train_lr_spin.setRange(0.00001, 0.1)
        self.train_lr_spin.setDecimals(5)
        self.train_lr_spin.setValue(0.001)
        settings_form.addRow("Скорость обучения:", self.train_lr_spin)

        settings_group.setLayout(settings_form)
        train_layout.addWidget(settings_group)

        # Горизонтальное расположение: график слева, лог справа
        h_layout = QHBoxLayout()

        # Левая часть – график
        left_frame = QVBoxLayout()
        self.train_chart = QChart()
        self.train_series = QLineSeries()
        self.train_chart.addSeries(self.train_series)
        self.train_axisX = QValueAxis()
        self.train_axisX.setTitleText("Эпоха")
        self.train_chart.addAxis(self.train_axisX, Qt.AlignBottom)
        self.train_axisY = QValueAxis()
        self.train_axisY.setTitleText("Ошибка")
        self.train_chart.addAxis(self.train_axisY, Qt.AlignLeft)
        self.train_series.attachAxis(self.train_axisX)
        self.train_series.attachAxis(self.train_axisY)
        self.train_chart_view = QChartView(self.train_chart)
        left_frame.addWidget(self.train_chart_view)

        # Кнопки обучения
        btn_layout = QHBoxLayout()
        btn_bc = QPushButton("Обучить PointerNet")
        btn_bc.clicked.connect(self.train_pointer_net)
        btn_layout.addWidget(btn_bc)

        btn_slim = QPushButton("Обучить SLIM")
        btn_slim.clicked.connect(self.train_slim)
        btn_layout.addWidget(btn_slim)

        btn_ppo = QPushButton("Обучить PPO")
        btn_ppo.clicked.connect(self.train_ppo_thread)
        btn_layout.addWidget(btn_ppo)

        # Кнопка остановки обучения
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.stop_btn)

        btn_save = QPushButton("Сохранить веса")
        btn_save.clicked.connect(self.save_model_weights)
        btn_layout.addWidget(btn_save)

        btn_load = QPushButton("Загрузить веса")
        btn_load.clicked.connect(self.load_model_weights)
        btn_layout.addWidget(btn_load)

        btn_newdata = QPushButton("Новый датасет")
        btn_newdata.clicked.connect(self.regenerate_dataset)
        btn_layout.addWidget(btn_newdata)

        left_frame.addLayout(btn_layout)
        h_layout.addLayout(left_frame)

        # Правая часть – лог
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        h_layout.addWidget(self.train_log)

        train_layout.addLayout(h_layout)
        sub_tabs.addTab(training_tab, "Обучение НС")

        layout.addWidget(sub_tabs)
        return train_tab

    def stop_training(self):
        """Останавливает текущий поток обучения."""
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
            self.train_log.append("Отправлен сигнал остановки обучения PPO...")
            self.stop_btn.setEnabled(False)
        if hasattr(self, 'trainer') and self.trainer:
            self.trainer.stop()
            self.train_log.append("Отправлен сигнал остановки PointerNet...")
            self.stop_btn.setEnabled(False)

    def regenerate_dataset(self):
        """Помечает, что при следующем обучении нужно перегенерировать датасет."""
        self.force_regen_data = True
        if os.path.exists("milp_dataset.npz"):
            os.remove("milp_dataset.npz")
        self.train_log.append("Датасет будет пересоздан при следующем обучении.")

    def train_slim(self):
        self.train_series.clear()
        self.train_log.clear()
        self.stop_btn.setEnabled(True)
        self.signals = TrainingSignals()
        self.signals.progress.connect(self.update_train_chart)
        self.signals.log.connect(self.train_log.append)
        self.slim_trainer = SelfLabelingTrainer(signals=self.signals)

        def train_and_save():
            model = self.slim_trainer.train_slim(
                epochs=self.train_epoch_spin.value(),
                lr=self.train_lr_spin.value(),
                dataset_path="gpss_dataset.npz",
                unlabeled_path="unlabeled.npz",  # <-- включаем "срезы"
                self_label_interval=30,  # раз в 100 эпох
                confidence_threshold=0.95
            )
            if model:
                torch.save(model.state_dict(), "slim_pointer.pth")
                self.signals.log.emit("Модель SLIM сохранена как slim_pointer.pth")
            self.stop_btn.setEnabled(False)

        self.slim_thread = threading.Thread(target=train_and_save, daemon=True)
        self.slim_thread.start()

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель", "",
                                              "PyTorch (*.pth);;Pickle (*.pkl);;All (*.*)")
        if not path:
            return
        model_name = os.path.splitext(os.path.basename(path))[0]
        set_model_path(model_name, path)  # запоминает в weights.py
        self.core.settings['model_path'] = path
        self.core.save_settings(self.core.settings)  # <-- ДОБАВИТЬ: сохраняем в БД
        self.core.loaded_model_name = model_name
        self.model_status_label.setText(f"Загружена модель: {model_name}")

        # Создаём RL-агента
        success = self.core.load_rl_model_if_available()
        if success:
            self.model_status_label.setText(f"Загружена модель: {self.core.loaded_model_name} (агент активен)")
            self.statusBar().showMessage("RL-агент активирован", 3000)
        else:
            self.model_status_label.setText(f"Модель выбрана, но не удалось создать агента")

    def train_pointer_net(self):
        self.train_series.clear()
        self.train_log.clear()
        self.stop_btn.setEnabled(True)
        self.signals = TrainingSignals()
        self.signals.progress.connect(self.update_train_chart)
        self.signals.log.connect(self.train_log.append)
        self.trainer = Trainer(signals=self.signals)

        def train_and_save():
            load_path = self.core.settings.get('model_path')
            if load_path and not os.path.exists(load_path):
                load_path = None
            model = self.trainer.train_bc(
                epochs=self.train_epoch_spin.value(),
                lr=self.train_lr_spin.value(),
                load_path=load_path,
                force_regen_data=getattr(self, 'force_regen_data', False)  # если была нажата кнопка "Новый датасет"
            )
            if model:
                torch.save(model.state_dict(), "pointer_net.pth")
                set_model_path("pointer_net", "pointer_net.pth")
                self.core.loaded_model_name = "pointer_net"
                self.core.settings['model_path'] = "pointer_net.pth"
                self.model_status_label.setText("Загружена модель: pointer_net")
                self.signals.log.emit("Модель PointerNet сохранена и установлена как текущая.")
            else:
                self.signals.log.emit("Обучение прервано, модель не сохранена.")
            self.stop_btn.setEnabled(False)
            self.force_regen_data = False

        self.train_thread = threading.Thread(target=train_and_save, daemon=True)
        self.train_thread.start()

    def update_train_chart(self, epoch, loss, accuracy=None):
        self.train_series.append(epoch, loss)
        if self.train_series.count() > 1:
            self.train_axisX.setRange(0, epoch + 1)
            y_vals = [p.y() for p in self.train_series.points()]
            self.train_axisY.setRange(min(y_vals) - 0.1, max(y_vals) + 0.1)

    def load_bc_model(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Выберите BC-модель", "", "Pickle Files (*.pkl)")
        if not path:
            return
        try:
            self.core.settings['bc_model_path'] = path
            self.core.save_settings(self.core.settings)
            from core import BCAgent
            self.core.rl_agent = BCAgent(path)
            self.core.aps_engine.use_rl = True
            self.core.aps_engine.rl_agent = self.core.rl_agent
            self.statusBar().showMessage(f"Модель загружена: {path}")
            QMessageBox.information(self, "BC", f"Модель {path} загружена.")
            self.core.loaded_model_name = os.path.basename(path)
            self.model_status_label.setText(f"Загружена модель: {self.core.loaded_model_name}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def save_model_weights(self):
        if not hasattr(self, 'trainer') or self.trainer.model is None:
            QMessageBox.warning(self, "Внимание", "Сначала обучите модель PointerNet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить веса", "pointer_net.pth", "PyTorch (*.pth)")
        if path:
            if self.trainer.save(path):
                self.train_log.append(f"Веса сохранены в {path}")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить веса.")

    def load_model_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить веса", "", "PyTorch (*.pth)")
        if not path:
            return
        # Сохраняем путь для использования при следующем обучении
        self.loaded_weights_path = path
        self.train_log.append(f"Веса будут загружены из {path} при следующем обучении.")


    def train_bc_thread(self): threading.Thread(target=self.train_bc, daemon=True).start()
    def train_bc(self):
        try:
            from train_behavior_cloning import generate_training_data
            from sklearn.neural_network import MLPClassifier
            QMessageBox.information(self, "BC", "Генерация данных и обучение...")
            X, y = generate_training_data(2000)
            model = MLPClassifier(hidden_layer_sizes=(256,256), max_iter=25)
            model.fit(X, y)
            with open(self.core.settings['bc_model_path'], "wb") as f: pickle.dump(model, f)
            QMessageBox.information(self, "BC", "Модель BC обучена и сохранена.")
        except Exception as e: QMessageBox.critical(self, "Ошибка", str(e))

    def update_ppo_chart(self, epoch, loss):
        self.train_series.append(epoch, loss)
        if self.train_series.count() > 1:
            self.train_axisX.setRange(0, epoch + 1)
            y_vals = [p.y() for p in self.train_series.points()]
            self.train_axisY.setRange(min(y_vals) - 0.1, max(y_vals) + 0.1)

    def train_ppo_thread(self):
        self.train_series.clear()
        self.train_log.clear()
        self.stop_btn.setEnabled(True)
        self.stop_event = threading.Event()
        self.ppo_thread = PPOTrainThread(
            self.core,
            self.train_epoch_spin.value(),
            self.train_lr_spin.value(),
            int(self.core.settings.get('ppo_n_orders', 10)),
            5,  # n_resources жестко 5
            self.stop_event
        )
        self.ppo_thread.progress.connect(self.update_train_chart)
        self.ppo_thread.finished.connect(
            lambda name: self.train_log.append(
                f"Модель сохранена: {name}" if name else "Обучение PPO завершено без сохранения")
        )
        self.ppo_thread.finished.connect(lambda: self.stop_btn.setEnabled(False))
        self.ppo_thread.start()
        self.train_thread = self.ppo_thread

    def train_ppo(self, episodes, lr, n_orders, n_res, progress_callback=None):
        try:
            from train_ppo_dispatcher import train_ppo as run_ppo
            run_ppo(episodes=episodes, lr=lr, n_orders=n_orders, n_resources=n_res,
                    progress_callback=progress_callback)
            self.core.message_signal.emit("info", "Обучение PPO завершено. Модель сохранена.")
        except Exception as e:
            self.core.message_signal.emit("error", f"Ошибка PPO: {e}")

    def load_ppo_model(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Выберите PPO-модель", "", "PyTorch Files (*.pth)")
        if not path:
            return
        try:
            self.core.settings['ppo_model_path'] = path
            self.core.save_settings(self.core.settings)
            self.statusBar().showMessage(f"Путь к PPO-модели: {path}")
            QMessageBox.information(self, "PPO", f"Модель {path} выбрана. Будет использована при обучении.")
            self.core.loaded_model_name = os.path.basename(path)
            self.model_status_label.setText(f"Загружена модель: {self.core.loaded_model_name}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    # ---------- НАСТРОЙКИ ----------
    def create_settings_tab(self):
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)
        form = QFormLayout()
        self.dqn_lr_spin = QDoubleSpinBox(); self.dqn_lr_spin.setRange(0.00001,0.1); self.dqn_lr_spin.setDecimals(5); self.dqn_lr_spin.setValue(float(self.core.settings.get('dqn_lr', 3e-4)))
        form.addRow("Скорость обучения DQN:", self.dqn_lr_spin)
        self.dqn_ep_spin = QSpinBox(); self.dqn_ep_spin.setRange(10,5000); self.dqn_ep_spin.setValue(int(self.core.settings.get('dqn_episodes', 500)))
        form.addRow("Количество эпизодов:", self.dqn_ep_spin)
        self.gantt_slot_spin = QSpinBox(); self.gantt_slot_spin.setRange(5,120); self.gantt_slot_spin.setValue(int(self.core.settings.get('gantt_slot_minutes', 120)))
        form.addRow("Интервал сетки (минуты):", self.gantt_slot_spin)
        self.erp_file_edit = QLineEdit(self.core.settings.get('erp_file', 'erp_orders.json'))
        form.addRow("Файл заказов ERP:", self.erp_file_edit)
        self.mes_file_edit = QLineEdit(self.core.settings.get('mes_file', 'mes_events.json'))
        form.addRow("Файл событий MES:", self.mes_file_edit)
        layout.addLayout(form)
        save_btn = QPushButton("Сохранить настройки"); save_btn.clicked.connect(self.apply_settings)
        layout.addWidget(save_btn)
        gen_btn = QPushButton("Сгенерировать тестовые ERP/MES"); gen_btn.clicked.connect(self.generate_mock_erp_mes)
        layout.addWidget(gen_btn)
        return settings_tab

    def apply_settings(self):
        try:
            new_settings = {
                'dqn_lr': str(self.dqn_lr_spin.value()),
                'dqn_episodes': str(self.dqn_ep_spin.value()),
                'gantt_slot_minutes': str(self.gantt_slot_spin.value()),
                'erp_file': self.erp_file_edit.text(),
                'mes_file': self.mes_file_edit.text()
            }
            self.core.save_settings(new_settings)
            QMessageBox.information(self, "Настройки", "Сохранено.")
        except Exception as e: QMessageBox.critical(self, "Ошибка", str(e))

    def generate_mock_erp_mes(self):
        try:
            import erp_mes_mock
            erp_mes_mock.generate(self.core.settings.get('erp_file', 'erp_orders.json'),
                                  self.core.settings.get('mes_file', 'mes_events.json'),
                                  self.core.resources)
            QMessageBox.information(self, "Генерация", "Файлы созданы.")
        except Exception as e: QMessageBox.critical(self, "Ошибка", str(e))

    # ---------- ЗАВЕРШЁННЫЕ ----------
    def create_completed_tab(self):
        completed_tab = QWidget()
        layout = QVBoxLayout(completed_tab)
        self.completed_table = QTableWidget()
        self.completed_table.setColumnCount(4)
        self.completed_table.setHorizontalHeaderLabels(["Операция", "Заказ", "Ресурс", "Действия"])
        self.completed_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.completed_table)
        self.refresh_completed_table()
        return completed_tab

    def refresh_completed_table(self):
        completed = get_completed_tasks()
        self.completed_table.setColumnCount(5)  # теперь 5 колонок
        self.completed_table.setHorizontalHeaderLabels(["Операция", "Тип", "Заказ", "Ресурс", "Действия"])
        self.completed_table.setRowCount(len(completed))
        type_map = get_operation_types()
        for i, task in enumerate(completed):
            self.completed_table.setItem(i, 0, QTableWidgetItem(task['operation_id']))
            type_name = type_map.get(task['type_id'], '—')
            self.completed_table.setItem(i, 1, QTableWidgetItem(type_name))
            self.completed_table.setItem(i, 2, QTableWidgetItem(task['order_id']))
            self.completed_table.setItem(i, 3, QTableWidgetItem(task['resource_name']))
            btn_delete = QPushButton("Удалить")
            btn_delete.clicked.connect(
                lambda checked=False, oid=task['operation_id']: self.core.delete_completed_task(oid))
            self.completed_table.setCellWidget(i, 4, btn_delete)

    def show_message(self, typ, text):
        if typ == 'info':
            self.statusBar().showMessage(text, 5000)
        elif typ == 'warning':
            QMessageBox.warning(self, "Предупреждение", text)
        elif typ == 'error':
            QMessageBox.critical(self, "Ошибка", text)

class OrderEditDialog(QDialog):
    def __init__(self, order, core, parent=None):
        super().__init__(parent)
        self.order = order
        self.core = core
        self.setWindowTitle(f"Заказ: {order.id}")
        self.resize(400, 350)
        layout = QFormLayout(self)

        self.name_edit = QLineEdit(order.name)
        layout.addRow("Название:", self.name_edit)

        self.due_edit = QLineEdit(order.due_date.strftime("%Y-%m-%d %H:%M"))
        layout.addRow("Дата сдачи:", self.due_edit)

        self.prio_spin = QDoubleSpinBox()
        self.prio_spin.setRange(0.5, 2.0)
        self.prio_spin.setValue(order.priority_weight)
        layout.addRow("Приоритет:", self.prio_spin)

        if order.ops:
            op = order.ops[0]
            self.type_combo = QComboBox()
            types = get_operation_types()
            self.type_combo.addItems(types.values())
            if op.type_id and op.type_id in types:
                self.type_combo.setCurrentText(types[op.type_id])
            layout.addRow("Тип операции:", self.type_combo)

            self.dur_edit = QLineEdit(str(op.norm_duration))
            layout.addRow("Длит. (мин):", self.dur_edit)

            self.item_edit = QLineEdit(op.item)
            layout.addRow("Изделие:", self.item_edit)

            self.qty_edit = QLineEdit(str(op.quantity))
            layout.addRow("Кол-во:", self.qty_edit)

        # Кнопки
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save_changes)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def save_changes(self):
        try:
            self.order.name = self.name_edit.text()
            self.order.due_date = datetime.strptime(self.due_edit.text(), "%Y-%m-%d %H:%M")
            self.order.priority_weight = self.prio_spin.value()

            if self.order.ops:
                op = self.order.ops[0]
                type_map = get_operation_types()
                type_name = self.type_combo.currentText()
                op.type_id = next((tid for tid, tn in type_map.items() if tn == type_name), None)
                op.norm_duration = float(self.dur_edit.text())
                op.item = self.item_edit.text()
                op.quantity = int(self.qty_edit.text())

            from database import save_orders_to_db
            save_orders_to_db([self.order])
            self.core.message_signal.emit("info", f"Заказ {self.order.id} обновлён.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("logo.png"))
    app.setStyle(QStyleFactory.create("Fusion"))
    window = APSApp()
    window.show()
    sys.exit(app.exec())