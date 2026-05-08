# Интеллектуальная система оптимизации производственной деятельности (IPOS)
Данный репозиторий является проектом ВКРМ и будет дорабатываться.\
Первичная структура проекта:\
APS/IPOS\
├── main_gui_pyside6.py       ← Главный исполняемый файл (интерфейс)\
├── core.py                   ← Ядро бизнес-логики\
├── database.py               ← Вся работа с SQLite\
├── datamodels.py             ← Датаклассы (Order, Operation, Resource)\
├── optimizer_milp.py         ← MILP-оптимизатор (OR-Tools)\
├── training.py               ← Обучение PointerNet + генерация данных\
├── train_ppo_dispatcher.py   ← Обучение PPO\
├── ppo_agent.py              ← Архитектура агента PPO\
├── job_shop_env.py           ← Среда для RL\
├── weights.py                ← Хранилище путей к весовым файлам\
├── predictor.py              ← Нейропредсказатель длительности\
├── benchmark_runner.py       ← Тестирование MILP на бенчмарках FJSP\
├── generate_bc_data.py       ← Генерация экспертных данных для BC\
├── train_ppo_from_milp.py    ← Обучение PPO на MILP-примерах\
├── train_slim.py             ← Self-Labeling обучение (эксперимент)\
├── erp_mes_mock.py           ← Генератор тестовых ERP/MES файлов\
├── logo.png                  ← Иконка приложения\
├── requirements.txt          ← Зависимости Python\
├── milp_dataset.npz          ← Кэшированный датасет для обучения\
├── weights.pkl               ← Хранилище путей к моделям\
└── pointer_net.pth           ← Сохранённые веса PointerNet\

```
Позже отредактирую данный фрагмент и доведу до ума
```
