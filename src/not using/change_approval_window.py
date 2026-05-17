# change_approval_window.py
import customtkinter as ctk
from tkinter import messagebox
from database import get_proposed_changes, accept_change, apply_accepted_changes, update_schedule

class ChangeApprovalWindow(ctk.CTkToplevel):
    def __init__(self, parent, aps):
        super().__init__(parent)
        self.title("Подтверждение изменений расписания")
        self.geometry("800x500")
        self.aps = aps
        self.parent = parent
        self.changes = get_proposed_changes()
        self.checkboxes = {}
        self.build_ui()

    def build_ui(self):
        if not self.changes:
            ctk.CTkLabel(self, text="Нет предложенных изменений").pack(pady=20)
            return
        ctk.CTkLabel(self, text="Предложенные изменения MILP:").pack(anchor="w", padx=10)
        # Простая таблица с чекбоксами
        for i, ch in enumerate(self.changes):
            frame = ctk.CTkFrame(self)
            frame.pack(fill="x", padx=10, pady=2)
            var = ctk.BooleanVar(value=True)
            self.checkboxes[ch['id']] = var
            ctk.CTkCheckBox(frame, variable=var).pack(side="left", padx=5)
            text = f"{ch['operation_id']} → ресурс {ch['new_resource_id']}, " \
                   f"с {ch['new_start'].strftime('%d.%m.%y %H:%M')} по {ch['new_end'].strftime('%d.%m.%y %H:%M')}"
            ctk.CTkLabel(frame, text=text).pack(side="left", padx=5)
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=10)
        ctk.CTkButton(btn_frame, text="Применить отмеченные", command=self.apply_selected).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Отклонить все", command=self.reject_all).pack(side="left", padx=5)

    def apply_selected(self):
        for cid, var in self.checkboxes.items():
            if var.get():
                accept_change(cid)
        apply_accepted_changes()
        self.aps.current_schedule = self.aps.run_milp_incremental()  # обновить
        self.parent.draw_dashboard()
        self.destroy()

    def reject_all(self):
        # Удалить все предложенные изменения
        for cid in self.checkboxes:
            accept_change(cid)  # пометим accepted, но не применяем (можно и удалить запись)
            # альтернативно: удалить из таблицы proposed_changes
        self.destroy()