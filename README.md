# Перевод шумерских текстов

Пакет `sumbase` обучает и оценивает модели перевода шумерской транслитерации
на английский. Дополнительные задачи: транслитерация клинописных знаков и OCR.

## Setup

Нужны Python 3.10–3.12 и uv 0.10.10 или новее. Все версии зависимостей
зафиксированы в `uv.lock`. Команды выполняются из корня репозитория.

```sh
uv sync --locked --extra cpu --extra train --extra corpus
uv run --no-sync pre-commit install
uv run --no-sync pre-commit run --all-files
uv run --no-sync python -m pytest
```

Для NVIDIA вместо `--extra cpu` используйте `--extra cu128`. Эти варианты
взаимоисключающие. Для генерации DOCX-отчётов добавьте `--extra reports`.
Для минимальной установки инференса: `uv sync --locked --no-dev --extra cpu`.
В сборщике корпуса для некоторых источников также нужен `curl`.

## Train

```sh
uv run --no-sync sumbase prepare --config configs/mt_byt5.yaml
uv run --no-sync sumbase baseline --config configs/mt_byt5.yaml --kind tm
uv run --no-sync sumbase train --config configs/mt_byt5.yaml
uv run --no-sync sumbase evaluate --config configs/mt_byt5.yaml
```

Конфигурация собирается Hydra из групп `data`, `model`, `train`, `generation`.
Переопределения: `-o train.batch_size=4 -o train.epochs=1`.
Для добавления отсутствующего ключа используйте `-o +train.max_steps=2`.
Исходные конфиги других задач также поддерживаются загрузчиком Hydra.

Быстрая проверка обучения на небольшой модели со случайными весами:

```sh
uv run --no-sync sumbase prepare --config configs/smoke/mt.yaml
uv run --no-sync sumbase train --config configs/smoke/mt.yaml
```

Это проверка исполнения, а не оценка качества основной модели.

## Production preparation

После обучения каталог запуска содержит `config.yaml` и `model/`
с весами, конфигурацией Hugging Face и токенизатором. Для предсказания нужны оба.
Экспорт ONNX/TensorRT и серверная поставка пока не реализованы.

## Infer

```sh
uv run --no-sync sumbase predict --run runs/mt_byt5_small --text "1(disz) udu niga"
uv run --no-sync sumbase predict --run runs/mt_byt5_small --input input.txt --output predictions.tsv
```

`input.txt`: UTF-8, один пример на строку. Результат TSV: вход и предсказание.
Для нескольких изображений повторяйте опцию `--image` перед каждым путём.

## Docker

```sh
docker compose build cpu
docker compose run --rm cpu python -m sumbase info
```

Docker также устанавливает зависимости из `uv.lock`. Для CUDA используется
сервис `gpu`. Сборка образа требует сети и отдельной проверки на целевой машине.

# Результаты

| Задача         | Метрика     |  Было | Стало |      Δ |
| -------------- | ----------- | ----: | ----: | -----: |
| Перевод        | BLEU ↑      | 12,12 | 10,12 |  −2,00 |
| Перевод        | chrF ↑      | 48,40 | 24,46 | −23,94 |
| Перевод        | CER, % ↓    | 79,60 | 86,58 |  +6,98 |
| Перевод        | Точные, % ↑ |  1,09 |  0,49 |  −0,60 |
| Транслитерация | BLEU ↑      | 89,87 | 25,78 | −64,09 |
| Транслитерация | chrF ↑      | 91,88 | 59,05 | −32,83 |
| Транслитерация | CER, % ↓    |  9,93 | 30,51 | +20,58 |
| Транслитерация | Точные, % ↑ | 61,95 | 10,27 | −51,68 |

[Таблицы и предсказания](runs/README.md). Без нормализации; Δ - пункты.
