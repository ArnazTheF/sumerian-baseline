# Бейзлайны для шумерского языка

Три задачи на открытых данных, каждая со своим бейзлайном и метриками:

| Задача | Вход и выход | Данные | Модель |
|---|---|---|---|
| ocr | фото таблички → Unicode-клинопись | SumTablets_Photos | Swin-Tiny + трансформерный декодер |
| translit | Unicode-клинопись → транслитерация | SumTablets | ByT5-small |
| mt | транслитерация → английский | CDLI/MTAAC, Ур III | ByT5-small |

Кроме обучаемых моделей есть опорные точки без обучения: translation memory (поиск ближайшей строки в обучающей выборке) и готовые модели с Hugging Face.

## Результаты

Перевод, тест CDLI UrIIILineByLine, 1010 строк:

| Метод | Обучающих пар | BLEU | chrF |
|---|---|---|---|
| translation memory | 5000 | 24.5 | 44.4 |
| translation memory | 7268 (все) | 26.5 | 47.2 |

Простой поиск похожей строки даёт BLEU выше, чем NMT у Punia et al. (2020, около 21), потому что административные тексты Ур III состоят из повторяющихся формул. Напрямую цифры сравнивать нельзя, сплиты другие, но планка для нейросетевой модели получается именно такая.

Обученные модели и готовые модели с Hugging Face ещё не запускались, их результатов пока нет. Ориентиры из литературы: CER 35.4 для OCR по прорисовкам (Img2SumGlyphs), chrF 97.6 для транслитерации (SumTablets).

Смоук-тест (крошечные модели, все три задачи, обучение, оценка, инференс) проходит на CPU примерно за минуту. Он проверяет, что код работает, качество моделей он не измеряет.

## Запуск в Docker

```bash
make build      # образ для GPU, если есть nvidia-smi, иначе для CPU
make smoke      # проверка, что всё работает
make nogpu      # цифры без обучения: translation memory и готовые модели
make lite       # обучение модели перевода, укладывается в 12 ГБ видеопамяти
```

Другие цели: `make lite-translit`, `make lite-ocr`, `make mt`, `make translit`, `make ocr`, `make tensorboard`. Каталоги `data/`, `runs/` и `hf_cache/` монтируются с хоста.

Перед обучением можно проверить, хватит ли видеопамяти: `make memcheck CFG=configs/lite/mt.yaml`. На GPU команда делает два настоящих шага обучения на самом длинном батче, без GPU выдаёт оценку. Оценки для всех конфигов есть в GPU_REQUIREMENTS.md.

На Mac контейнер работает только на CPU: Docker Desktop запускает его в виртуальной машине с Linux, куда GPU Apple не пробрасывается.

## Запуск без Docker

Нужен Python 3.10–3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt   # на Mac, точные версии
# на Linux: pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
#           pip install -r requirements.txt
export PYTHONPATH=src
python -m sumbase info
bash scripts/smoke_test.sh
bash scripts/eval_no_gpu.sh
```

На Mac с Apple Silicon вычисления идут на GPU через Metal (MPS). Если что-то падает, можно переключиться на CPU: `SUMBASE_DEVICE=cpu`. Этот путь пока не проверялся на реальной машине.

## Команды и конфиги

```bash
python -m sumbase prepare  -c configs/mt_byt5.yaml           # скачать и подготовить данные
python -m sumbase train    -c configs/mt_byt5.yaml           # обучить и оценить на тесте
python -m sumbase evaluate -c configs/mt_byt5.yaml
python -m sumbase baseline -c configs/mt_byt5.yaml --kind tm # или --kind zeroshot
python -m sumbase predict  --run runs/mt_byt5_small --text "1(disz) udu niga"
```

Любой параметр конфига меняется флагом `-o`, например `-o train.epochs=5 -o data.max_train=2000`.

- `configs/*.yaml` — основные конфиги для GPU на 16 ГБ и больше.
- `configs/lite/` — те же задачи под 12 ГБ.
- `configs/nogpu/` — оценка без обучения.
- `configs/smoke/` — смоук-тест.

Результаты запуска лежат в `runs/<имя>/`:

- `metrics_test.json` — метрики;
- `predictions_test.tsv` — предсказания рядом с эталоном;
- `train_summary.json` — время обучения и пик видеопамяти;
- `model/` — лучшая модель.

## Данные

- CDLI/MTAAC: параллельные строки «транслитерация — английский» из репозитория cdli-gh/Semi-Supervised-NMT-for-Sumerian-English, лицензия MIT. Транслитерация в ASCII-ATF (`sz` вместо `š`). Английская сторона без пунктуации.
- SumTablets: 91 606 табличек, лицензия CC BY 4.0. Глифы восстановлены из транслитерации по словарю, а не размечены по изображению.
- SumTablets_Photos: фото и прорисовки табличек, около 43 ГБ, лицензия Apache-2.0. Читается потоково, на диск сохраняется только нужное число картинок.

## Ограничения

Готовая модель перевода praeclarum/cuneiform обучалась на CDLI и, вероятно, видела тестовые строки, поэтому её результат завышен. Для готовых моделей SumTablets формат входа восстановлен по их конфигам, а не по документации. Перед тем как использовать эти цифры, их стоит проверить по `predictions_test.tsv`.

Почти все данные — административные тексты Ур III. На литературных текстах качество будет заметно ниже.

## Ссылки

- Punia et al., 2020. Towards the First Machine Translation System for Sumerian Transliterations. COLING.
- Simmons et al., 2024. SumTablets: A Transliteration Dataset of Sumerian Tablets. ML4AL.
- Xue et al., 2022. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.
- Liu et al., 2021. Swin Transformer.