## Решение задачи классификации обращений граждан чемпионата Курской области - Цифровой прорыв 2022

Для установки необходимых библиотек выполните: `pip3 install -r requirements.txt`.

Ссылка для скачивания моделей: https://drive.google.com/file/d/19ZuVOJLnCZVjk15g23orlKWIzZkvrPS9/view?usp=sharing

Для распаковки моделей `tar -xvf roberta_models.tar.gz`.

В файле `config.ini` находится конфигурация обучения.

В файле `utils.py` находятся вспомогательные функции.

В ноутбуке `train_model.ipynb` находится код обучения моделей.

Для запуска обучения `python3 train.py`.

Для запуска инференса на тестовом датасете `python3 inference.py`.
