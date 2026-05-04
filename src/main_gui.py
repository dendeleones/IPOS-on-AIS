# main_gui.py
import customtkinter as ctk
from tkinter import messagebox, Canvas, Scrollbar, BOTH, HORIZONTAL, VERTICAL, RIGHT, LEFT, BOTTOM, Y, X
import threading
import pickle
import numpy as np
import random
from datetime import datetime, timedelta
from datamodels import Order, Operation, Resource
from predictor import DurationPredictor
from optimizer_milp import build_milp_schedule
from aps_core import HybridAPS
from dqn_agent import DQNAgent
from database import (init_db, save_orders_to_db, load_orders_from_db,
                      update_schedule, get_fixed_operations,
                      save_resources, load_resources,
                      save_downtime, load_downtimes,
                      delete_resource, delete_downtime,
                      get_state_types, add_state_type, delete_state_type)
import json
import os

# ---------- BC Agent ----------
class BCAgent:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    def select_action(self, state, action_mask):
        probs = self.model.predict_proba([state])[0]
        probs[~np.array(action_mask)] = -1
        return np.argmax(probs)

# ---------- Главное приложение ----------
class APSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("APS Планировщик производства")
        self.geometry("1400x800")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # БД
        init_db()
        self.resources = load_resources()
        if not self.resources:
            self.resources = [Resource(id='R1', name='Станок 1'),
                              Resource(id='R2', name='Станок 2'),
                              Resource(id='R3', name='Станок 3')]
            save_resources(self.resources)
        self.resource_map = {r.id: r for r in self.resources}
        self.predictor = DurationPredictor()
        self.aps = HybridAPS(self.resources, predictor=self.predictor)
        self.rl_agent = None
        self.use_rl = False
        self.orders = []
        self.current_schedule = {}
        self.downtimes = load_downtimes()
        self.state_types = get_state_types()

        self.settings = {
            'bc_model_path': 'behavior_cloning.pkl',
            'dqn_episodes': 500,
            'dqn_lr': 3e-4,
            'erp_file': 'erp_orders.json',
            'mes_file': 'mes_events.json',
            'gantt_slot_minutes': 120
        }
        self.load_settings()

        saved_orders = load_orders_from_db(self.resources)
        if saved_orders:
            self.orders = saved_orders
            self.aps.load_orders(self.orders)
            self.aps.run_milp()
            self.current_schedule = self.aps.current_schedule
            if self.orders:
                messagebox.showinfo("Загрузка", f"Загружено {len(self.orders)} заказов из БД.")

        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.create_sidebar()

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self._drag_data = {"rect_item": None, "text_item": None, "op_id": None, "from_left": False,
                           "start_logical_x": 0, "start_logical_y": 0}
        self.show_dashboard()

    def create_sidebar(self):
        ctk.CTkLabel(self.sidebar, text="APS Система", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20, padx=20, anchor="w")
        ctk.CTkButton(self.sidebar, text="Дашборд", command=self.show_dashboard).pack(pady=5, padx=20, anchor="w")
        ctk.CTkButton(self.sidebar, text="Заказы", command=self.show_orders).pack(pady=5, padx=20, anchor="w")
        ctk.CTkButton(self.sidebar, text="Планирование", command=self.show_planning).pack(pady=5, padx=20, anchor="w")
        ctk.CTkButton(self.sidebar, text="Оборудование", command=self.show_equipment).pack(pady=5, padx=20, anchor="w")
        ctk.CTkButton(self.sidebar, text="Состояния", command=self.show_states).pack(pady=5, padx=20, anchor="w")
        ctk.CTkButton(self.sidebar, text="Обучение", command=self.show_training).pack(pady=5, padx=20, anchor="w")
        ctk.CTkButton(self.sidebar, text="Настройки", command=self.show_settings).pack(pady=5, padx=20, anchor="w")

    def clear_main(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def load_settings(self):
        if os.path.exists('settings.json'):
            with open('settings.json', 'r', encoding='utf-8') as f:
                self.settings.update(json.load(f))

    def save_settings(self):
        with open('settings.json', 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=2)

    # ─── ДАШБОРД ─────────────────────────────────────────────────
    def show_dashboard(self):
        self.clear_main()
        ctk.CTkLabel(self.main_frame, text="Интерактивный график загрузки",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)

        self.dash_canvas = Canvas(self.main_frame, bg='white')
        vsbar = Scrollbar(self.main_frame, orient=VERTICAL, command=self.dash_canvas.yview)
        hsbar = Scrollbar(self.main_frame, orient=HORIZONTAL, command=self.dash_canvas.xview)
        self.dash_canvas.configure(yscrollcommand=vsbar.set, xscrollcommand=hsbar.set)
        vsbar.pack(side=RIGHT, fill=Y)
        hsbar.pack(side=BOTTOM, fill=X)
        self.dash_canvas.pack(side=LEFT, fill=BOTH, expand=True)

        self.LEFT_PANEL_W = 300
        self.GANTT_X0 = 320
        self.dash_canvas.config(scrollregion=(0, 0, 1600, 800))

        self.draw_left_panel()
        self.draw_gantt_interactive()
        self._bind_all_events()

    def _bind_all_events(self):
        self.dash_canvas.tag_bind("draggable", "<ButtonPress-1>", self.on_drag_start)
        self.dash_canvas.tag_bind("draggable", "<B1-Motion>", self.on_drag_motion)
        self.dash_canvas.tag_bind("draggable", "<ButtonRelease-1>", self.on_drag_stop)
        self.dash_canvas.tag_bind("gantt_op", "<ButtonPress-1>", self.on_drag_start)
        self.dash_canvas.tag_bind("gantt_op", "<B1-Motion>", self.on_drag_motion)
        self.dash_canvas.tag_bind("gantt_op", "<ButtonRelease-1>", self.on_drag_stop)

    def draw_left_panel(self):
        self.dash_canvas.delete("left_panel")
        self.dash_canvas.delete("draggable")
        self.dash_canvas.create_rectangle(0, 0, self.LEFT_PANEL_W, 800, fill="#f8f8f8",
                                          outline="gray", tags="left_panel")
        self.dash_canvas.create_text(self.LEFT_PANEL_W/2, 25, text="Задания",
                                     font=("Arial", 14, "bold"), tags="left_panel")
        # Показываем только операции, которых нет в текущем расписании
        scheduled_ids = set(self.current_schedule.keys()) if self.current_schedule else set()
        y = 60
        for order in self.orders:
            for op in order.ops:
                if op.id in scheduled_ids:
                    continue   # уже распределена
                x1, y1, x2, y2 = 10, y, self.LEFT_PANEL_W-10, y+60
                grp_tag = f"grp_{op.id}"
                self.dash_canvas.create_rectangle(x1, y1, x2, y2,
                                                  fill="lightyellow", outline="black",
                                                  tags=("left_panel", "op_rect", grp_tag, "draggable"))
                order_num = order.id.split('_')[-1]
                due_str = order.due_date.strftime("%d.%m %H:%M")
                text = f"Заказ №{order_num}\n{op.item}\nСрок: {due_str}\n{op.id}"
                self.dash_canvas.create_text(x1+5, y1+5, anchor="nw",
                                             text=text, font=("Arial", 8),
                                             tags=("left_panel", "op_text", grp_tag, "draggable"),
                                             width=x2-x1-10)
                y += 65
        self.dash_canvas.configure(scrollregion=(0, 0, 1600, max(800, y+20)))

    def draw_gantt_interactive(self):
        self.dash_canvas.delete("gantt")
        self.dash_canvas.delete("timescale")
        self.dash_canvas.delete("downtime_rect")
        self.dash_canvas.delete("gantt_op")
        if not self.current_schedule:
            self.dash_canvas.create_text(500, 200, text="Нет расписания. Постройте план на вкладке «Планирование».",
                                         tags="gantt", font=("Arial", 12))
            return

        schedule = self.current_schedule
        all_starts = [v['start'] for v in schedule.values()]
        all_ends = [v['end'] for v in schedule.values()]
        if not all_starts:
            return
        min_time = min(all_starts)
        max_time = max(all_ends)
        total_minutes = (max_time - min_time).total_seconds() / 60.0
        if total_minutes <= 0:
            total_minutes = 1

        slot_min = self.settings['gantt_slot_minutes']
        canvas_w = 1600
        x_scale = (canvas_w - self.GANTT_X0 - 40) / total_minutes
        y_step = 100
        resource_ids = [r.id for r in self.resources]
        resource_y = {rid: 100 + i * y_step for i, rid in enumerate(resource_ids)}

        self._draw_time_scale(min_time, max_time, x_scale, slot_min, canvas_w)

        for rid, y in resource_y.items():
            self.dash_canvas.create_line(self.GANTT_X0, y, canvas_w - 20, y, fill="gray", tags="gantt")
            self.dash_canvas.create_text(self.GANTT_X0 - 10, y - 15,
                                         text=self.resource_map[rid].name,
                                         anchor="e", font=("Arial", 10, "bold"), tags="gantt")
            cur = min_time.replace(second=0, microsecond=0)
            while cur <= max_time:
                x = self.GANTT_X0 + (cur - min_time).total_seconds() / 60.0 * x_scale
                self.dash_canvas.create_line(x, y - 20, x, y + 20, fill="lightgray",
                                             dash=(2, 4), tags="gantt")
                cur += timedelta(minutes=slot_min)

        for rid, intervals in self.downtimes.items():
            if rid not in resource_y:
                continue
            y = resource_y[rid]
            for start_dt, end_dt, state_id in intervals:
                if end_dt > min_time and start_dt < max_time:
                    x1 = self.GANTT_X0 + max(0, (start_dt - min_time).total_seconds() / 60.0) * x_scale
                    x2 = self.GANTT_X0 + min(total_minutes, (end_dt - min_time).total_seconds() / 60.0) * x_scale
                    color = 'gray'
                    if state_id in self.state_types:
                        color = self.state_types[state_id][1]
                    self.dash_canvas.create_rectangle(x1, y - 25, x2, y + 25,
                                                      fill=color, stipple='gray50', outline='',
                                                      tags=("gantt", "downtime_rect"))

        op_info_map = {}
        for order in self.orders:
            for op in order.ops:
                op_info_map[op.id] = (order.id, order.due_date, op.item)

        for op_id, info in schedule.items():
            if op_id not in op_info_map:
                continue
            order_id, due, item = op_info_map[op_id]
            res_id = info['resource_id']
            if res_id not in resource_y:
                continue
            y = resource_y[res_id]
            x1 = self.GANTT_X0 + (info['start'] - min_time).total_seconds() / 60.0 * x_scale
            x2 = self.GANTT_X0 + (info['end'] - min_time).total_seconds() / 60.0 * x_scale
            grp_tag = f"grp_{op_id}"
            self.dash_canvas.create_rectangle(
                x1, y - 25, x2, y + 25, fill='lightblue', outline='black',
                tags=("gantt_op", "op_rect", grp_tag)
            )
            order_num = order_id.split('_')[-1]
            due_str = due.strftime("%d.%m %H:%M")
            text = f"Заказ №{order_num}\n{item}\nСрок: {due_str}"
            if x2 - x1 > 40:
                self.dash_canvas.create_text(
                    (x1 + x2) / 2, y, text=text,
                    font=("Arial", 8, "bold"),
                    tags=("gantt_op", "op_text", grp_tag),
                    width=x2 - x1 - 4
                )
            else:
                self.dash_canvas.create_text(
                    (x1 + x2) / 2, y, text=op_id,
                    font=("Arial", 6),
                    tags=("gantt_op", "op_text", grp_tag)
                )

    def _draw_time_scale(self, min_time, max_time, x_scale, slot_min, canvas_w):
        self.dash_canvas.create_rectangle(self.GANTT_X0, 10, canvas_w - 20, 80,
                                          fill='#f0f0f0', outline='', tags="timescale")
        cur = min_time.replace(second=0, microsecond=0)
        prev_date = None
        while cur <= max_time:
            x = self.GANTT_X0 + (cur - min_time).total_seconds() / 60.0 * x_scale
            if cur.minute % slot_min == 0:
                self.dash_canvas.create_line(x, 10, x, 80, fill="lightgray", tags="timescale")
            if cur.minute == 0:
                self.dash_canvas.create_line(x, 10, x, 80, fill="black", tags="timescale")
                time_label = cur.strftime("%H:%M")
                self.dash_canvas.create_text(x, 60, text=time_label, font=("Arial", 8), anchor="n", tags="timescale")
                cur_date = cur.strftime("%d.%m.%Y")
                if cur_date != prev_date:
                    self.dash_canvas.create_text(x, 75, text=cur_date, font=("Arial", 8, "bold"), anchor="n", tags="timescale")
                    prev_date = cur_date
            cur += timedelta(minutes=slot_min)

    # ─── Drag and Drop (улучшенный) ──────────────────────────
    def on_drag_start(self, event):
        x = self.dash_canvas.canvasx(event.x)
        y = self.dash_canvas.canvasy(event.y)
        items = self.dash_canvas.find_overlapping(x, y, x, y)
        op_id = None
        from_left = False
        rect_item = None
        text_item = None
        for item in items:
            tags = self.dash_canvas.gettags(item)
            if "op_rect" in tags:
                for tag in tags:
                    if tag.startswith("grp_"):
                        op_id = tag[4:]
                        rect_item = item
                        break
        if not op_id:
            self._reset_drag()
            return
        # Ищем связанный текст
        for it in self.dash_canvas.find_withtag(f"grp_{op_id}"):
            if "op_text" in self.dash_canvas.gettags(it):
                text_item = it
                break
        # Определяем, из левой панели ли
        coords = self.dash_canvas.coords(rect_item)
        if coords and coords[0] < self.LEFT_PANEL_W:
            from_left = True
        self._drag_data = {
            "rect_item": rect_item,
            "text_item": text_item,
            "op_id": op_id,
            "from_left": from_left,
            "start_logical_x": x,
            "start_logical_y": y
        }

    def on_drag_motion(self, event):
        data = self._drag_data
        if not data["rect_item"]:
            return
        cur_x = self.dash_canvas.canvasx(event.x)
        cur_y = self.dash_canvas.canvasy(event.y)
        dx = cur_x - data["start_logical_x"]
        dy = cur_y - data["start_logical_y"]
        self.dash_canvas.move(data["rect_item"], dx, dy)
        if data["text_item"]:
            self.dash_canvas.move(data["text_item"], dx, dy)
        data["start_logical_x"] = cur_x
        data["start_logical_y"] = cur_y

    def on_drag_stop(self, event):
        data = self._drag_data
        op_id = data.get("op_id")
        rect_item = data.get("rect_item")
        if not op_id or not rect_item:
            self._reset_drag()
            return
        coords = self.dash_canvas.coords(rect_item)
        if not coords or len(coords) < 4:
            self._reset_drag()
            return
        x1, y1, x2, y2 = coords
        from_left = data.get("from_left", False)

        if x1 >= self.GANTT_X0:
            self._handle_drop_on_gantt(op_id, x1, y1, x2, y2)
        elif x1 < self.LEFT_PANEL_W:
            self._handle_drop_on_left(op_id)
        else:
            # Отпустили между панелями – отменяем, возвращаем на место через перерисовку
            pass

        # Перерисовываем дашборд
        self.draw_left_panel()
        self.draw_gantt_interactive()
        self._bind_all_events()   # на всякий случай обновим привязки
        self._reset_drag()

    def _handle_drop_on_gantt(self, op_id, x1, y1, x2, y2):
        # Определяем ресурс по y
        resource_y_dict = {rid: 100 + i * 100 for i, rid in enumerate([r.id for r in self.resources])}
        closest_res = None
        min_dist = float('inf')
        mid_y = (y1 + y2) / 2
        for rid, y in resource_y_dict.items():
            dist = abs(mid_y - y)
            if dist < min_dist:
                min_dist = dist
                closest_res = rid

        schedule = self.current_schedule or {}
        all_starts = [v['start'] for v in schedule.values()] if schedule else [datetime.now()]
        all_ends = [v['end'] for v in schedule.values()] if schedule else [datetime.now() + timedelta(days=1)]
        min_time = min(all_starts)
        max_time = max(all_ends)
        total_minutes = (max_time - min_time).total_seconds() / 60.0
        if total_minutes <= 0:
            total_minutes = 1
        x_scale = (1600 - self.GANTT_X0 - 40) / total_minutes
        new_start_minutes = (x1 - self.GANTT_X0) / x_scale
        new_start = min_time + timedelta(minutes=new_start_minutes)

        duration = None
        for order in self.orders:
            for op in order.ops:
                if op.id == op_id:
                    duration = timedelta(minutes=op.norm_duration)
                    break
        if duration is None:
            return
        new_end = new_start + duration

        # Фиксация в БД
        update_schedule(op_id, closest_res, new_start, new_end, fixed=1)
        for order in self.orders:
            for op in order.ops:
                if op.id == op_id:
                    op.resource_id = closest_res
                    break
        self.aps.run_milp_with_fixed()
        self.current_schedule = self.aps.current_schedule   # обновляем расписание

    def _handle_drop_on_left(self, op_id):
        # Снять фиксацию (уберем из расписания)
        update_schedule(op_id, None, None, None, fixed=0)
        # Перестроим MILP без этой операции
        self.aps.run_milp_with_fixed()
        self.current_schedule = self.aps.current_schedule

    def _reset_drag(self):
        self._drag_data = {"rect_item": None, "text_item": None, "op_id": None, "from_left": False,
                           "start_logical_x": 0, "start_logical_y": 0}

    # ─── ЗАКАЗЫ ─────────────────────────────────────────────────
    def show_orders(self):
        self.clear_main()
        ctk.CTkLabel(self.main_frame, text="Управление заказами",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)

        form_frame = ctk.CTkFrame(self.main_frame)
        form_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(form_frame, text="Новый заказ").pack(anchor="w")
        row1 = ctk.CTkFrame(form_frame)
        row1.pack(fill="x", pady=5)
        ctk.CTkLabel(row1, text="Дата сдачи (ГГГГ-ММ-ДД ЧЧ:ММ):").pack(side="left")
        self.due_entry = ctk.CTkEntry(row1)
        self.due_entry.insert(0, "2026-05-10 08:00")
        self.due_entry.pack(side="left", padx=5)
        ctk.CTkLabel(row1, text="Приоритет (0.5-2):").pack(side="left", padx=5)
        self.prio_entry = ctk.CTkEntry(row1, width=50)
        self.prio_entry.insert(0, "1.0")
        self.prio_entry.pack(side="left")
        self.ops_frame = ctk.CTkFrame(form_frame)
        self.ops_frame.pack(fill="x", pady=10)
        self.ops_widgets = []
        self.add_operation_row()
        ctk.CTkButton(form_frame, text="+ Добавить операцию", command=self.add_operation_row).pack(anchor="w", pady=5)
        ctk.CTkButton(form_frame, text="Сохранить заказ", command=self.save_order).pack(pady=10)
        ctk.CTkButton(self.main_frame, text="Сгенерировать 10 случайных заказов",
                      command=self.generate_random_orders).pack(pady=5)
        ctk.CTkButton(self.main_frame, text="Загрузить из ERP (JSON)",
                      command=self.load_erp_orders).pack(pady=5)
        self.create_order_table()

    def add_operation_row(self):
        row = ctk.CTkFrame(self.ops_frame)
        row.pack(fill="x", pady=2)
        res_menu = ctk.CTkOptionMenu(row, values=[r.name for r in self.resources])
        res_menu.pack(side="left", padx=5)
        dur_entry = ctk.CTkEntry(row, width=80)
        dur_entry.insert(0, "60")
        dur_entry.pack(side="left", padx=5)
        item_entry = ctk.CTkEntry(row, width=100)
        item_entry.insert(0, "Деталь_1")
        item_entry.pack(side="left", padx=5)
        self.ops_widgets.append((res_menu, dur_entry, item_entry))

    def save_order(self):
        try:
            due_str = self.due_entry.get()
            due = datetime.strptime(due_str, "%Y-%m-%d %H:%M")
            prio = float(self.prio_entry.get())
            order_id = f"Заказ_{random.randint(100,999)}"
            order = Order(id=order_id, due_date=due, priority_weight=prio)
            for i, (res_menu, dur_entry, item_entry) in enumerate(self.ops_widgets, start=1):
                res_name = res_menu.get()
                res_id = [r.id for r in self.resources if r.name == res_name][0]
                dur = float(dur_entry.get())
                item = item_entry.get()
                op = Operation(id=f"{order_id}_оп{i}", order_id=order_id, item=item,
                               op_number=i, resource_id=res_id, norm_duration=dur)
                order.ops.append(op)
            self.orders.append(order)
            save_orders_to_db([order])
            self.aps.load_orders([order])
            self.refresh_order_table()
            messagebox.showinfo("Успех", f"Заказ {order_id} сохранён")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def generate_random_orders(self):
        random.seed(None)
        base = datetime(2026, 4, 27, 8, 0, 0)
        generated = []
        for i in range(10):
            due = base + timedelta(hours=random.randint(10, 100))
            order = Order(id=f"Заказ_{random.randint(1000,9999)}", due_date=due,
                          priority_weight=round(random.uniform(0.5, 2.0), 2))
            n_ops = random.randint(2, 3)
            prev = None
            for j in range(1, n_ops+1):
                op_id = f"{order.id}_оп{j}"
                res = random.choice(self.resources)
                order.ops.append(Operation(
                    id=op_id, order_id=order.id, item=f"Деталь_{random.randint(1,5)}",
                    op_number=j, resource_id=res.id,
                    norm_duration=round(random.uniform(20, 120), 1),
                    predecessors=[prev] if prev else []))
                prev = op_id
            generated.append(order)
        self.orders.extend(generated)
        save_orders_to_db(generated)
        for o in generated:
            self.aps.load_orders([o])
        self.refresh_order_table()
        messagebox.showinfo("Генерация", "Создано 10 случайных заказов и сохранено в БД.")

    def load_erp_orders(self):
        try:
            with open(self.settings['erp_file'], 'r', encoding='utf-8') as f:
                erp_data = json.load(f)
            loaded_orders = []
            for ord_data in erp_data:
                order = Order(id=ord_data['id'], due_date=datetime.fromisoformat(ord_data['due_date']),
                              priority_weight=ord_data.get('priority_weight', 1.0))
                for op_d in ord_data['operations']:
                    op = Operation(id=op_d['id'], order_id=order.id, item=op_d['item'],
                                   op_number=op_d['op_number'], resource_id=op_d['resource_id'],
                                   norm_duration=op_d['norm_duration'],
                                   predecessors=op_d.get('predecessors', []))
                    order.ops.append(op)
                loaded_orders.append(order)
            self.orders.extend(loaded_orders)
            save_orders_to_db(loaded_orders)
            for o in loaded_orders:
                self.aps.load_orders([o])
            self.refresh_order_table()
            messagebox.showinfo("ERP", f"Загружено {len(loaded_orders)} заказов.")
        except Exception as e:
            messagebox.showerror("Ошибка ERP", str(e))

    def create_order_table(self):
        from tkinter import ttk
        if hasattr(self, 'tree_frame'):
            self.tree_frame.destroy()
        self.tree_frame = ctk.CTkFrame(self.main_frame)
        self.tree_frame.pack(fill="both", expand=True, padx=20, pady=10)
        columns = ('id','due','ops','priority')
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show='headings')
        self.tree.heading('id', text='Заказ')
        self.tree.heading('due', text='Срок')
        self.tree.heading('ops', text='Операций')
        self.tree.heading('priority', text='Приоритет')
        self.tree.pack(fill="both", expand=True, side="left")
        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=vsb.set)
        self.refresh_order_table()

    def refresh_order_table(self):
        if hasattr(self, 'tree'):
            for i in self.tree.get_children():
                self.tree.delete(i)
            for order in self.orders:
                self.tree.insert('', 'end', values=(order.id,
                                                    order.due_date.strftime("%Y-%m-%d %H:%M"),
                                                    len(order.ops),
                                                    f"{order.priority_weight:.2f}"))

    # ─── ОБОРУДОВАНИЕ ──────────────────────────────────────────
    def show_equipment(self):
        self.clear_main()
        ctk.CTkLabel(self.main_frame, text="Управление оборудованием",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="x", padx=20, pady=10)
        self.equip_listbox = ctk.CTkTextbox(frame, height=150)
        self.equip_listbox.pack(fill="x", pady=5)
        self._refresh_equip_listbox()
        add_frame = ctk.CTkFrame(frame)
        add_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(add_frame, text="ID ресурса:").pack(side="left")
        self.new_res_id_entry = ctk.CTkEntry(add_frame, width=80)
        self.new_res_id_entry.pack(side="left", padx=5)
        ctk.CTkLabel(add_frame, text="Название:").pack(side="left", padx=5)
        self.new_res_name_entry = ctk.CTkEntry(add_frame, width=150)
        self.new_res_name_entry.pack(side="left", padx=5)
        ctk.CTkButton(add_frame, text="Добавить", command=self.add_resource).pack(side="left", padx=10)
        del_frame = ctk.CTkFrame(frame)
        del_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(del_frame, text="Выберите ID для удаления:").pack(side="left")
        self.del_res_combo = ctk.CTkComboBox(del_frame, values=[r.id for r in self.resources])
        self.del_res_combo.pack(side="left", padx=5)
        ctk.CTkButton(del_frame, text="Удалить", command=self.delete_resource).pack(side="left", padx=10)

    def _refresh_equip_listbox(self):
        self.equip_listbox.delete("0.0", "end")
        for r in self.resources:
            self.equip_listbox.insert("end", f"{r.id}: {r.name}\n")

    def add_resource(self):
        rid = self.new_res_id_entry.get().strip()
        name = self.new_res_name_entry.get().strip()
        if not rid or not name:
            messagebox.showerror("Ошибка", "Введите ID и название.")
            return
        if rid in self.resource_map:
            messagebox.showerror("Ошибка", "Ресурс с таким ID уже существует.")
            return
        new_res = Resource(id=rid, name=name)
        self.resources.append(new_res)
        self.resource_map[rid] = new_res
        save_resources(self.resources)
        self.aps.resources[rid] = new_res
        self._refresh_equip_listbox()
        self.del_res_combo.configure(values=[r.id for r in self.resources])
        messagebox.showinfo("Успех", f"Ресурс {name} добавлен.")

    def delete_resource(self):
        rid = self.del_res_combo.get()
        if rid not in self.resource_map:
            messagebox.showerror("Ошибка", "Ресурс не найден.")
            return
        for order in self.orders:
            for op in order.ops:
                if op.resource_id == rid:
                    messagebox.showerror("Ошибка", f"На ресурсе {rid} есть операции. Переназначьте их.")
                    return
        self.resources = [r for r in self.resources if r.id != rid]
        del self.resource_map[rid]
        if rid in self.aps.resources:
            del self.aps.resources[rid]
        delete_resource(rid)
        save_resources(self.resources)
        self._refresh_equip_listbox()
        self.del_res_combo.configure(values=[r.id for r in self.resources])
        messagebox.showinfo("Успех", f"Ресурс {rid} удалён.")

    # ─── СОСТОЯНИЯ И ПРОСТОИ ────────────────────────────────────
    def show_states(self):
        self.clear_main()
        ctk.CTkLabel(self.main_frame, text="Состояния и простои оборудования",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        notebook = ctk.CTkTabview(self.main_frame)
        notebook.pack(fill="both", expand=True, padx=20, pady=10)
        notebook.add("Типы состояний")
        notebook.add("Простои")

        tab1 = notebook.tab("Типы состояний")
        list_frame = ctk.CTkFrame(tab1)
        list_frame.pack(fill="x", pady=5)
        self.state_type_list = ctk.CTkTextbox(list_frame, height=100)
        self.state_type_list.pack(fill="x")
        self._refresh_state_type_list()
        add_frame = ctk.CTkFrame(tab1)
        add_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(add_frame, text="Название:").pack(side="left")
        self.state_name_entry = ctk.CTkEntry(add_frame, width=150)
        self.state_name_entry.pack(side="left", padx=5)
        ctk.CTkLabel(add_frame, text="Цвет (hex):").pack(side="left", padx=5)
        self.state_color_entry = ctk.CTkEntry(add_frame, width=80)
        self.state_color_entry.insert(0, "#FFA500")
        self.state_color_entry.pack(side="left", padx=5)
        ctk.CTkButton(add_frame, text="Добавить", command=self.add_state_type).pack(side="left", padx=10)
        del_frame = ctk.CTkFrame(tab1)
        del_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(del_frame, text="Удалить состояние:").pack(side="left")
        self.state_del_combo = ctk.CTkComboBox(del_frame, values=list(self.state_types.keys()))
        self.state_del_combo.pack(side="left", padx=5)
        ctk.CTkButton(del_frame, text="Удалить", command=self.del_state_type).pack(side="left", padx=10)

        tab2 = notebook.tab("Простои")
        res_sel = ctk.CTkFrame(tab2)
        res_sel.pack(fill="x", pady=5)
        ctk.CTkLabel(res_sel, text="Ресурс:").pack(side="left")
        self.downtime_res_combo = ctk.CTkComboBox(res_sel, values=[r.id for r in self.resources])
        self.downtime_res_combo.pack(side="left", padx=5)
        ctk.CTkLabel(res_sel, text="Тип состояния:").pack(side="left", padx=5)
        self.downtime_type_combo = ctk.CTkComboBox(res_sel, values=list(self.state_types.keys()))
        self.downtime_type_combo.pack(side="left", padx=5)
        add_dt = ctk.CTkFrame(tab2)
        add_dt.pack(fill="x", pady=10)
        ctk.CTkLabel(add_dt, text="Начало (ГГГГ-ММ-ДД ЧЧ:ММ):").pack(side="left")
        self.dt_start_entry = ctk.CTkEntry(add_dt, width=150)
        self.dt_start_entry.insert(0, "2026-04-27 10:00")
        self.dt_start_entry.pack(side="left", padx=5)
        ctk.CTkLabel(add_dt, text="Конец:").pack(side="left", padx=5)
        self.dt_end_entry = ctk.CTkEntry(add_dt, width=150)
        self.dt_end_entry.insert(0, "2026-04-27 12:00")
        self.dt_end_entry.pack(side="left", padx=5)
        ctk.CTkButton(add_dt, text="Добавить простой", command=self.add_downtime).pack(side="left", padx=10)
        self.downtime_list = ctk.CTkTextbox(tab2, height=120)
        self.downtime_list.pack(fill="x", pady=10)
        self._refresh_downtime_list()
        ctk.CTkButton(tab2, text="Удалить все простои для ресурса", command=self.clear_downtimes).pack(pady=5)

    def _refresh_state_type_list(self):
        self.state_type_list.delete("0.0", "end")
        for sid, (name, color) in self.state_types.items():
            self.state_type_list.insert("end", f"{sid}: {name} ({color})\n")

    def add_state_type(self):
        name = self.state_name_entry.get().strip()
        color = self.state_color_entry.get().strip()
        if not name or not color:
            messagebox.showerror("Ошибка", "Введите название и цвет.")
            return
        sid = add_state_type(name, color)
        self.state_types[sid] = (name, color)
        self._refresh_state_type_list()
        self.state_del_combo.configure(values=list(self.state_types.keys()))
        self.downtime_type_combo.configure(values=list(self.state_types.keys()))
        messagebox.showinfo("Успех", f"Состояние '{name}' добавлено.")

    def del_state_type(self):
        sid = self.state_del_combo.get()
        if not sid:
            return
        delete_state_type(sid)
        if sid in self.state_types:
            del self.state_types[sid]
        self._refresh_state_type_list()
        self.state_del_combo.configure(values=list(self.state_types.keys()))
        self.downtime_type_combo.configure(values=list(self.state_types.keys()))
        messagebox.showinfo("Успех", "Состояние удалено.")

    def add_downtime(self):
        rid = self.downtime_res_combo.get()
        state_id = self.downtime_type_combo.get()
        try:
            start = datetime.strptime(self.dt_start_entry.get(), "%Y-%m-%d %H:%M")
            end = datetime.strptime(self.dt_end_entry.get(), "%Y-%m-%d %H:%M")
            if end <= start:
                raise ValueError("Конец должен быть позже начала.")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        if rid not in self.downtimes:
            self.downtimes[rid] = []
        self.downtimes[rid].append((start, end, state_id))
        save_downtime(rid, start, end, state_id)
        self._refresh_downtime_list()
        messagebox.showinfo("Успех", f"Простой для {rid} добавлен.")

    def clear_downtimes(self):
        rid = self.downtime_res_combo.get()
        if rid in self.downtimes:
            for item in self.downtimes[rid]:
                delete_downtime(rid, item[0], item[1])
            del self.downtimes[rid]
        self._refresh_downtime_list()
        messagebox.showinfo("Успех", f"Все простои для {rid} удалены.")

    def _refresh_downtime_list(self):
        self.downtime_list.delete("0.0", "end")
        for rid, intervals in self.downtimes.items():
            name = self.resource_map.get(rid, Resource(rid, "?"))
            for s, e, sid in intervals:
                state_name = self.state_types.get(sid, ('?',''))[0]
                self.downtime_list.insert("end",
                    f"{name.name} ({rid}): {s.strftime('%d.%m.%y %H:%M')} - {e.strftime('%d.%m.%y %H:%M')} [{state_name}]\n")

    # ─── ПЛАНИРОВАНИЕ ──────────────────────────────────────────
    def show_planning(self):
        self.clear_main()
        ctk.CTkLabel(self.main_frame, text="Планирование и MILP",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        frame_ctrl = ctk.CTkFrame(self.main_frame)
        frame_ctrl.pack(fill="x", padx=20, pady=10)
        self.mode_var = ctk.StringVar(value="CR")
        ctk.CTkRadioButton(frame_ctrl, text="CR-диспетчер", variable=self.mode_var, value="CR").pack(side="left", padx=5)
        ctk.CTkRadioButton(frame_ctrl, text="RL/BC-диспетчер", variable=self.mode_var, value="RL").pack(side="left", padx=5)
        ctk.CTkButton(self.main_frame, text="Построить MILP-план", command=self.show_milp).pack(pady=10)
        ctk.CTkButton(self.main_frame, text="Применить фиксированные и перестроить",
                      command=self.run_fixed_milp).pack(pady=5)
        self.gantt_canvas = ctk.CTkCanvas(self.main_frame, bg='white', height=300)
        self.gantt_canvas.pack(fill="both", expand=True, padx=20, pady=10)

    def show_milp(self):
        if not self.orders:
            messagebox.showerror("Ошибка", "Сначала создайте заказы.")
            return
        self.aps.run_milp()
        self.current_schedule = self.aps.current_schedule
        if not self.current_schedule:
            messagebox.showerror("MILP", "Не удалось построить план.")
            return
        self.draw_gantt_static(self.current_schedule)

    def run_fixed_milp(self):
        self.aps.run_milp_with_fixed()
        self.current_schedule = self.aps.current_schedule
        if self.current_schedule:
            self.draw_gantt_static(self.current_schedule)
            messagebox.showinfo("Успех", "План перестроен с учётом фиксированных операций.")
        else:
            messagebox.showerror("Ошибка", "Не удалось построить план с фиксациями.")

    def draw_gantt_static(self, schedule):
        self.gantt_canvas.delete("all")
        if not schedule:
            return
        all_starts = [v['start'] for v in schedule.values()]
        all_ends = [v['end'] for v in schedule.values()]
        min_time = min(all_starts)
        max_time = max(all_ends)
        total_seconds = (max_time - min_time).total_seconds()
        if total_seconds <= 0:
            return
        width = self.gantt_canvas.winfo_width()
        if width < 100:
            width = 800
        y_step = 40
        resource_ids = [r.id for r in self.resources]
        resource_y = {rid: 20 + i * y_step for i, rid in enumerate(resource_ids)}
        for rid, y in resource_y.items():
            self.gantt_canvas.create_text(10, y, text=self.resource_map[rid].name, anchor="w")
        for op_id, info in schedule.items():
            x1 = (info['start'] - min_time).total_seconds() / total_seconds * (width - 20) + 10
            x2 = (info['end'] - min_time).total_seconds() / total_seconds * (width - 20) + 10
            y = resource_y[info['resource_id']] + 15
            self.gantt_canvas.create_rectangle(x1, y-10, x2, y+10, fill='lightblue')
            self.gantt_canvas.create_text(x1+5, y, anchor='w', text=op_id, font=("Arial", 7))

    # ─── ОБУЧЕНИЕ ──────────────────────────────────────────────
    def show_training(self):
        self.clear_main()
        ctk.CTkLabel(self.main_frame, text="Обучение моделей",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        ctk.CTkButton(self.main_frame, text="Загрузить BC-модель", command=self.load_bc_model).pack(pady=5)
        ctk.CTkButton(self.main_frame, text="Обучить RL (DQN)", command=self.train_dqn_thread).pack(pady=5)
        ctk.CTkButton(self.main_frame, text="Обучить Behaviour Cloning", command=self.train_bc_thread).pack(pady=5)

    def load_bc_model(self):
        try:
            path = self.settings['bc_model_path']
            self.rl_agent = BCAgent(path)
            self.aps.use_rl = True
            self.aps.rl_agent = self.rl_agent
            messagebox.showinfo("BC", f"Модель {path} загружена.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить BC-модель: {e}")

    def train_dqn_thread(self):
        threading.Thread(target=self.train_dqn, daemon=True).start()
    def train_dqn(self):
        try:
            from train_rl_dispatcher import train_agent
            messagebox.showinfo("DQN", "Обучение запущено в консоли.")
            train_agent()
            messagebox.showinfo("DQN", "Обучение завершено.")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def train_bc_thread(self):
        threading.Thread(target=self.train_bc, daemon=True).start()
    def train_bc(self):
        try:
            from train_behavior_cloning import generate_training_data
            from sklearn.neural_network import MLPClassifier
            messagebox.showinfo("BC", "Генерация данных и обучение...")
            X, y = generate_training_data(2000)
            model = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=25, verbose=False)
            model.fit(X, y)
            with open(self.settings['bc_model_path'], "wb") as f:
                pickle.dump(model, f)
            messagebox.showinfo("BC", "Модель BC обучена и сохранена.")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    # ─── НАСТРОЙКИ ─────────────────────────────────────────────
    def show_settings(self):
        self.clear_main()
        ctk.CTkLabel(self.main_frame, text="Настройки приложения",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(frame, text="Обучение RL:").pack(anchor="w")
        row_lr = ctk.CTkFrame(frame)
        row_lr.pack(fill="x", pady=5)
        ctk.CTkLabel(row_lr, text="Скорость обучения DQN:").pack(side="left")
        self.dqn_lr_entry = ctk.CTkEntry(row_lr, width=100)
        self.dqn_lr_entry.insert(0, str(self.settings['dqn_lr']))
        self.dqn_lr_entry.pack(side="left", padx=5)
        row_ep = ctk.CTkFrame(frame)
        row_ep.pack(fill="x", pady=5)
        ctk.CTkLabel(row_ep, text="Количество эпизодов:").pack(side="left")
        self.dqn_ep_entry = ctk.CTkEntry(row_ep, width=100)
        self.dqn_ep_entry.insert(0, str(self.settings['dqn_episodes']))
        self.dqn_ep_entry.pack(side="left", padx=5)
        ctk.CTkLabel(frame, text="Дашборд:").pack(anchor="w", pady=(10,0))
        row_slot = ctk.CTkFrame(frame)
        row_slot.pack(fill="x", pady=5)
        ctk.CTkLabel(row_slot, text="Интервал сетки (минуты):").pack(side="left")
        self.gantt_slot_entry = ctk.CTkEntry(row_slot, width=100)
        self.gantt_slot_entry.insert(0, str(self.settings['gantt_slot_minutes']))
        self.gantt_slot_entry.pack(side="left", padx=5)
        ctk.CTkLabel(frame, text="Интеграция ERP/MES:").pack(anchor="w", pady=(10,0))
        row_erp = ctk.CTkFrame(frame)
        row_erp.pack(fill="x", pady=5)
        ctk.CTkLabel(row_erp, text="Файл заказов ERP:").pack(side="left")
        self.erp_file_entry = ctk.CTkEntry(row_erp, width=200)
        self.erp_file_entry.insert(0, self.settings['erp_file'])
        self.erp_file_entry.pack(side="left", padx=5)
        row_mes = ctk.CTkFrame(frame)
        row_mes.pack(fill="x", pady=5)
        ctk.CTkLabel(row_mes, text="Файл событий MES:").pack(side="left")
        self.mes_file_entry = ctk.CTkEntry(row_mes, width=200)
        self.mes_file_entry.insert(0, self.settings['mes_file'])
        self.mes_file_entry.pack(side="left", padx=5)
        btn_frame = ctk.CTkFrame(self.main_frame)
        btn_frame.pack(pady=20)
        ctk.CTkButton(btn_frame, text="Сохранить настройки", command=self.apply_settings).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Сгенерировать тестовые ERP/MES",
                      command=self.generate_mock_erp_mes).pack(side="left", padx=10)

    def apply_settings(self):
        try:
            self.settings['dqn_lr'] = float(self.dqn_lr_entry.get())
            self.settings['dqn_episodes'] = int(self.dqn_ep_entry.get())
            self.settings['gantt_slot_minutes'] = int(self.gantt_slot_entry.get())
            self.settings['erp_file'] = self.erp_file_entry.get()
            self.settings['mes_file'] = self.mes_file_entry.get()
            self.save_settings()
            messagebox.showinfo("Настройки", "Сохранено.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неверные значения: {e}")

    def generate_mock_erp_mes(self):
        try:
            import erp_mes_mock
            erp_mes_mock.generate(self.settings['erp_file'], self.settings['mes_file'], self.resources)
            messagebox.showinfo("Генерация", f"Файлы созданы.")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    app = APSApp()
    app.mainloop()