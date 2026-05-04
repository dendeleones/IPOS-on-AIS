# Интеллектуальная система оптимизации производственной деятельности (IPOS)
Данный репозиторий является проектом ВКРМ и будет дорабатываться.\
Первичная структура проекта:\
RL_APS/\
├── datamodels.py          # Датаклассы Order, Operation, Resource, MESEvent\
├── predictor.py           # Нейросетевой прогноз длительности (scikit-learn)\
├── optimizer_milp.py      # MILP-планировщик (OR-Tools CP-SAT)\
├── job_shop_env.py        # Среда для обучения RL (с прогнозом)\
├── dqn_agent.py           # Агент DQN\
├── train_rl_dispatcher.py # Скрипт обучения агента\
├── aps_core.py            # Ядро системы: HybridAPS\
├── main_gui.py            # GUI на Tkinter (запуск приложения)\
├── requirements.txt       # Зависимости\
└── rl_dispatcher.pth      # Создается после обучения

```
Позже отредактирую данный фрагмент и доведу до ума
```
