# weights.py
import pickle
import os

WEIGHTS_FILE = "weights.pkl"

def load_weights_config():
    """Загружает словарь {название_модели: путь_к_файлу} из pickle-файла."""
    if not os.path.exists(WEIGHTS_FILE):
        return {}
    with open(WEIGHTS_FILE, 'rb') as f:
        return pickle.load(f)

def save_weights_config(config):
    """Сохраняет словарь конфигурации в pickle-файл."""
    with open(WEIGHTS_FILE, 'wb') as f:
        pickle.dump(config, f)

def get_model_path(model_name):
    """Возвращает путь к весам для указанной модели."""
    config = load_weights_config()
    return config.get(model_name)

def set_model_path(model_name, path):
    """Записывает путь к весам для модели и отмечает её как последнюю использованную."""
    config = load_weights_config()
    config[model_name] = path
    config['last_model'] = model_name       # запоминаем, какая модель была загружена последней
    save_weights_config(config)

def get_last_model_name():
    """Возвращает имя последней загруженной модели или None."""
    config = load_weights_config()
    return config.get('last_model')