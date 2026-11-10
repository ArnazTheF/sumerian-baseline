#!/usr/bin/env python3
"""Сборка Word-отчёта из сохранённых метрик."""
import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=REPO / "legacy/technical_runs/new_data")
parser.add_argument("--output", type=Path, default=REPO / "runs/new_data/evaluation_report.docx")
args = parser.parse_args()
ROOT = args.root.resolve()
A = json.loads((ROOT / "analysis.json").read_text())
TASKS = A["tasks"]
OUT = args.output.resolve()
TR = TASKS["translit"]["summary"]["cohorts"]["external"]
TC = TASKS["translit"]["summary"]["cohorts"]["legacy_control"]
TD = TASKS["translit"]["diagnostics"]
SAMPLING = TASKS["translit"]["summary"].get("sampling")
POPULATION = SAMPLING["population"] if SAMPLING else 43213
BASELINE = TASKS["translit"]["summary"].get("baseline_subset")
FULL_TRANSLIT = TR["evaluated"] == POPULATION
FULL_TRANSLIT_CONTROL = TC["evaluated"] == 657
REUSED = TD.get("reused_predictions")
TOTAL_EVALUATED = sum(
    t["summary"]["cohorts"][c]["evaluated"]
    for t in TASKS.values()
    for c in ["external", "legacy_control"]
)
TOTAL_PLANNED = sum(
    t["summary"]["cohorts"][c]["planned"]
    for t in TASKS.values()
    for c in ["external", "legacy_control"]
)
ROOT_REL = str(ROOT.relative_to(REPO))
doc = Document()
sec = doc.sections[0]
sec.page_width = Inches(8.5)
sec.page_height = Inches(11)
sec.top_margin = Inches(0.72)
sec.bottom_margin = Inches(0.70)
sec.left_margin = sec.right_margin = Inches(0.8)
sec.header_distance = sec.footer_distance = Inches(0.3)

for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Header", "Footer"]:
    s = doc.styles[name]
    s.font.name = "Arial"
    s.font.color.rgb = RGBColor(0, 0, 0)
    rp = s.element.get_or_add_rPr()
    fonts = rp.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rp.insert(0, fonts)
    for attr in ["ascii", "hAnsi", "cs", "eastAsia"]:
        fonts.set(qn("w:" + attr), "Arial")
    for attr in ["asciiTheme", "hAnsiTheme", "cstheme", "eastAsiaTheme"]:
        fonts.attrib.pop(qn("w:" + attr), None)
    border = s.element.find(".//" + qn("w:pBdr"))
    if border is not None:
        border.getparent().remove(border)

normal = doc.styles["Normal"]
normal.font.size = Pt(11)
normal.paragraph_format.line_spacing = 1.08
normal.paragraph_format.space_after = Pt(7)
doc.styles["Title"].font.size = Pt(23)
doc.styles["Title"].font.bold = True
doc.styles["Title"].paragraph_format.space_after = Pt(10)
doc.styles["Subtitle"].font.size = Pt(10)
doc.styles["Subtitle"].paragraph_format.space_after = Pt(12)
for name, size in [("Heading 1", 15), ("Heading 2", 12)]:
    s = doc.styles[name]
    s.font.size = Pt(size)
    s.font.bold = True
    s.paragraph_format.space_before = Pt(10)
    s.paragraph_format.space_after = Pt(7)
    s.paragraph_format.keep_with_next = True

hp = sec.header.paragraphs[0]
hp.text = "Проект АИРИ    Оценка моделей шумерского языка"
for r in hp.runs:
    r.font.size = Pt(8.5)
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
fp.add_run("Страница ").font.size = Pt(9)
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
fp._p.append(field)
doc.core_properties.title = "Оценка моделей шумерского языка"
doc.core_properties.subject = "Охват тестирования результаты и рекомендации"
doc.core_properties.author = ""


def p(text, bold_start=None, small=False):
    para = doc.add_paragraph()
    if bold_start and text.startswith(bold_start):
        para.add_run(bold_start).bold = True
        para.add_run(text[len(bold_start) :])
    else:
        para.add_run(text)
    if small:
        for r in para.runs:
            r.font.size = Pt(9.5)
    return para


def h(text, new=False, level=1):
    para = doc.add_paragraph(text, style=f"Heading {level}")
    if new:
        para.paragraph_format.page_break_before = True
    return para


def table(headers, rows, widths, font=10, left_columns=()):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for col, w in zip(t.columns, widths):
        col.width = Inches(w)
    pr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        b = OxmlElement("w:" + side)
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "5")
        b.set(qn("w:color"), "D9D9D9")
        borders.append(b)
    pr.append(borders)
    for data in [headers] + rows:
        row = t.rows[0] if data is headers else t.add_row()
        rpr = row._tr.get_or_add_trPr()
        split = OxmlElement("w:cantSplit")
        rpr.append(split)
        if data is headers:
            repeat = OxmlElement("w:tblHeader")
            rpr.append(repeat)
        idx = len(t.rows) - 1 if data is not headers else 0
        for j, (cell, text, w) in enumerate(zip(row.cells, data, widths)):
            cell.width = Inches(w)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for side in ["top", "bottom", "left", "right"]:
                m = OxmlElement("w:" + side)
                m.set(qn("w:w"), "80")
                m.set(qn("w:type"), "dxa")
                margins.append(m)
            cp.append(margins)
            shade = OxmlElement("w:shd")
            shade.set(
                qn("w:fill"), "24445B" if data is headers else ("F1F5F8" if idx % 2 else "FFFFFF")
            )
            cp.append(shade)
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            para.paragraph_format.space_before = Pt(0)
            para.paragraph_format.line_spacing = 1.05
            para.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if j == 0 or (j in left_columns and data is not headers)
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            run = para.add_run(str(text))
            run.font.name = "Arial"
            run.font.size = Pt(font)
            run.font.color.rgb = RGBColor(255, 255, 255) if data is headers else RGBColor(0, 0, 0)
            run.bold = data is headers
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return t


def number(v):
    return f"{v:,.2f}".replace(",", " ").replace(".", ",")


def integer(v):
    return f"{v:,}".replace(",", "\u00a0")


def compact(v):
    return number(v).rstrip("0").rstrip(",")


def metrics(task, cohort, kind="raw"):
    return TASKS[task]["summary"]["cohorts"][cohort][kind]


doc.add_paragraph("Оценка моделей\nшумерского языка", style="Title")
subtitle = (
    "Полная проверка новых строк и контроля транслитерации"
    if FULL_TRANSLIT and FULL_TRANSLIT_CONTROL
    else (
        "Полная проверка новых строк транслитерации"
        if FULL_TRANSLIT
        else (
            f"Расширенная проверка транслитерации на {SAMPLING['requested_fraction']*100:g} процентах пула"
            if SAMPLING
            else "Охват тестирования результаты и рекомендации"
        )
    )
)
doc.add_paragraph(subtitle + "\nДокумент подготовлен 5 ноября 2026 года", style="Subtitle")
if FULL_TRANSLIT:
    p(
        f"Транслитерация проверена на всех {integer(TR['evaluated'])} подготовленных новых строках. Прежний контроль: {integer(TC['evaluated'])} из 657 примеров. Показатели MT и OCR сохранены из предыдущего прогона; их охват указан в таблице. Сквозной перевод изображения в английский не оценивался."
    )
else:
    p(
        "Проверка была частичной относительно всего собранного корпуса. Запланированные прогоны завершены: перевод проверен почти на всём подготовленном новом тесте, OCR — на всём подготовленном Unicode-тесте вырезок, транслитерация — на выборке. Прежние тесты проверены выборочно. Сквозной перевод изображения в английский не оценивался."
    )
additional = (
    TR["evaluated"]
    + TC["evaluated"]
    - REUSED["external"]["evaluated"]
    - REUSED["legacy_control"]["evaluated"]
    if REUSED
    else TR["evaluated"] - SAMPLING["baseline_retained"] if SAMPLING else 0
)
p(
    f"Сводка содержит {integer(TOTAL_EVALUATED)} предсказаний."
    + (
        f" В этом дополнении досчитано {integer(additional)} ответов транслитерации; результаты MT и OCR сохранены из предыдущего прогона."
        if SAMPLING
        else ""
    )
    + " Обучение и обновление весов не выполнялись; новые веса не скачивались. На GitHub ничего не публиковалось."
)
h("Фактический охват")
table(
    ["Проверка", "Доступно\nв этом наборе", "Оценено", "Доля", "Статус"],
    [
        ["Новый тест перевода", "2 471", "2 467", "99,84%", "4 входа пропущены"],
        [
            "Новая транслитерация",
            integer(POPULATION),
            integer(TR["evaluated"]),
            number(100 * TR["evaluated"] / POPULATION) + "%",
            "Полный новый test" if FULL_TRANSLIT else "Выборка по источникам",
        ],
        ["Новый OCR Unicode test", "1 722", "1 722", "100%", "Весь подготовленный test"],
        ["Прежний тест перевода", "1 010", "100", "9,90%", "Контрольная выборка"],
        [
            "Прежняя транслитерация",
            "657",
            integer(TC["evaluated"]),
            number(100 * TC["evaluated"] / 657) + "%",
            "Полный прежний test" if TC["evaluated"] == 657 else "Контрольная выборка",
        ],
        ["Прежний OCR", "200", "100", "50%", "Контрольная выборка"],
        ["Сквозные связки", "341 кандидат", "0", "0%", "Нет проверенного эталона"],
    ],
    [2.05, 1.08, 0.78, 0.78, 2.21],
    9.5,
)
p(
    "«Полный» здесь относится только к конкретному подготовленному тесту. Все редакции корпуса, все изображения, train и valid целиком не прогонялись. OCR-охват 100% не означает проверку всего архива изображений.",
    small=True,
)
p(
    (
        f"Ранее готовые {integer(REUSED['external']['evaluated'])} новых строк и {integer(REUSED['legacy_control']['evaluated'])} контрольных ответов сохранены. Досчитаны все оставшиеся записи без изменения модели."
        if FULL_TRANSLIT and REUSED
        else (
            f"Первоначальные {integer(SAMPLING['baseline_retained'])} строк входят в расширенный тест. Дополнение выбрано пропорционально оставшимся данным по источникам и длине, без изменения модели."
            if SAMPLING
            else "Имя файла summary_full.json означает завершение выбранного плана запуска. Оно не означает, что модель обработала весь доступный корпус."
        )
    ),
    small=True,
)
h("Основной вывод")
p(
    "На новых данных chrF перевода и транслитерации ниже прежнего контроля. OCR-модель целых прорисовок не выдаёт правильный одиночный знак на рельефных вырезках. Дальнейший приоритет — очистка эталонов, исправление входных адаптеров и подготовка сопоставимого теста изображений."
)

h("Результаты моделей", new=True)
p(
    "Перевод: praeclarum/cuneiform. Транслитерация: colesimmons/sumerian-transliteration. OCR: colesimmons/lineart-ocr.",
    small=True,
)
h("Исходные строки", level=2)
rows = []
for task, label in [("mt", "Перевод"), ("translit", "Транслитерация")]:
    for cohort, suffix in [("external", "новый"), ("legacy_control", "контроль")]:
        s = TASKS[task]["summary"]["cohorts"][cohort]
        m = s["raw"]
        rows.append(
            [
                label + " / " + suffix,
                integer(s["evaluated"]),
                number(m["bleu"]),
                number(m["chrf"]),
                number(m["cer"]),
                number(m["exact_match"]),
            ]
        )
table(
    ["Задача и набор", "n", "BLEU ↑", "chrF ↑", "CER ↓", "Точно % ↑"],
    rows,
    [2.15, 0.65, 0.9, 0.9, 0.95, 1.25],
)
p(
    "BLEU и chrF имеют шкалу 0–100 и не означают процент правильно переведённых текстов. CER — число символьных правок относительно длины эталона; при лишней генерации он может превышать 100%.",
    small=True,
)
h("После нормализации записи", level=2)
rows = []
for task, label in [("mt", "Перевод"), ("translit", "Транслитерация")]:
    for cohort, suffix in [("external", "новый"), ("legacy_control", "контроль")]:
        m = metrics(task, cohort, "normalized")
        rows.append(
            [
                label + " / " + suffix,
                number(m["bleu"]),
                number(m["chrf"]),
                number(m["cer"]),
                number(m["exact_match"]),
            ]
        )
table(["Задача и набор", "BLEU ↑", "chrF ↑", "CER ↓", "Точно % ↑"], rows, [2.3, 1.1, 1.1, 1.1, 1.2])
p(
    "Для MT приведён регистр и удалена пунктуация. Для транслитерации приведены регистр и CDLI ASCII, удалены редакторские скобки и маркеры сомнения; пробелы сжаты. Основные исходные ответы сохранены.",
    small=True,
)
h("Распознавание изображений", level=2)
table(
    ["Набор", "n", "CER по\nзнакам ↓", "Точно % ↑", "Лимит\nгенерации %"],
    [
        ["Рельефные вырезки\nMaiCuBeDa", "1 722", "2 370,38", "0,00", "98,95"],
        ["Целые прорисовки\nпрежнего теста", "100", "59,99", "5,00", "1,00"],
    ],
    [2.5, 0.6, 1.2, 1.1, 1.4],
)
p(
    "Эти OCR-строки описывают разные задачи. На вырезках модель генерирует последовательность вместо одного знака; ответов ровно из одного знака — 0%. CER зависит от лимита 32 токена и не является сопоставимой оценкой новых целых табличек.",
    small=True,
)
p(
    "У MT исходный CER контроля завышают повреждённые эталоны и длинная генерация. Без 23 записей с xxx остаются 77 примеров: CER 45,21 и chrF 53,75. Это дополнительная диагностика с другим составом выборки.",
    small=True,
)

h("Причины ошибок и следующие действия", new=True)
h("Что обнаружено", level=2)
findings = [
    (
        "Качество эталонов. ",
        "В новом MT-тесте 12 записей содержат служебное примечание вместо перевода. После их исключения оценено 2 456 сегментов: chrF 24,51 и CER 86,25. Их удаление почти не меняет общий chrF.",
    ),
    (
        "Потери входа. ",
        "Токенизатор транслитерации теряет редкие знаки в 507 из 43 213 строк: 578 из 261 772 знаков. В 109 строках теряются все клинописные знаки. Обозначения x/X встречаются в 5 430 строках и также игнорируются.",
    ),
    (
        "Подмножество без выпавших знаков. ",
        f"В проверенной выборке потеря редких знаков отмечена в {TR['input_signs_dropped_examples']} строках. На {integer(TD['no_dropped_glyphs_or_x_damage']['evaluated'])} строках без потерь и без x/X chrF составляет {number(TD['no_dropped_glyphs_or_x_damage']['raw']['chrf'])}, против {number(metrics('translit', 'legacy_control')['chrf'])} на контроле. Это диагностика с другим составом подмножества.",
    ),
    (
        "Коллизии идентификаторов. ",
        f"В полном пуле транслитерации 236 строк разделяют 22 ID из-за пустого line_id. Сборщик исправлен. Для оценки применены составные ключи; в первоначальном прогоне пересчитаны 17 потерянных ответов. Итоговые файлы содержат {integer(TR['evaluated'])} разных строк нового теста.",
    ),
    (
        "Длинная генерация. ",
        "Лимита 256 токенов достигли 3,36% новых ответов MT. Достижение лимита не всегда означает цикл, однако найдены явные повторы, например I am a satrap. Для OCR предел 32 токена достигнут в 98,95% случаев.",
    ),
]
for lead, body in findings:
    p(lead + body, bold_start=lead)
h("Рекомендации без обучения", level=2)
for text in [
    "1. Переэкспортировать корпус исправленным сборщиком, убрать служебные MT-эталоны и проверить выравнивание нескольких десятков примеров каждого крупного источника. Нынешний срез сохранить для сравнения.",
    "2. Проверять сохранность входа при токенизации. Передавать повреждения поддерживаемым маркером неизвестного знака; неподдерживаемые знаки явно отмечать. Это сохраняет позицию, но не учит модель читать новый знак.",
    "3. Для OCR подготовить вручную проверенные целые таблички или поверхности с соответствующей последовательностью знаков. Одиночные знаки оценивать как отдельную задачу.",
    "4. Проверить часть из 341 сквозной связки по изображению, поверхностям, редакции, строкам и переводу. После этого считать показатель изображения в английский.",
    "5. Параметры остановки и проверки повторов выбирать на valid. Сохранить разбиение по объектам и коллекциям, а 50–100 переводов дополнительно оценить вручную по смыслу, именам и числам.",
]:
    p(text)

h("Методика и воспроизводимость", new=True)
if FULL_TRANSLIT:
    p(
        f"Включены все {integer(POPULATION)} строки нового пула из 20 источников. Прежний контроль включает {integer(TC['evaluated'])} из 657 примеров. Ограничения размера нового теста по коллекциям или диапазонам длины не применялись. Его состав соответствует всему подготовленному пулу."
    )
    if REUSED:
        p(
            f"Ранее полученные {integer(REUSED['external']['evaluated'])} ответов нового теста и {integer(REUSED['legacy_control']['evaluated'])} ответов контроля переиспользованы с проверкой неизменности входов, эталонов и предсказаний. Параметры модели и декодирования не менялись.",
            small=True,
        )
elif SAMPLING:
    p(
        f"Расширенная выборка содержит {integer(TR['evaluated'])} строк из {integer(POPULATION)} и сохраняет все {integer(SAMPLING['baseline_retained'])} первоначальных строк. Дополнительные строки распределены пропорционально остаткам по источникам и третям длины. Целые квоты определены методом наибольших остатков; внутри групп использован стабильный SHA-256 с seed 42."
    )
    p(
        "Исходная выборка давала крупным источникам по 150 строк. В расширении крупнейшие коллекции имеют больший вес. Разницу общих метрик между версиями следует связывать с изменением состава данных; веса модели не менялись.",
        small=True,
    )
else:
    p(
        "Транслитерация выбрана из 20 источников: не более 150 строк на источник. Большие группы разделены на три равные части по длине входа; из каждой взято по 50 строк по стабильному SHA-256 с seed 42. Малые группы взяты целиком. Метрики характеризуют эту выборку, а не частотный состав всех 43 213 строк."
    )
p(
    (
        "Контроль транслитерации пройден целиком; для MT и OCR сохранено по 100 контрольных примеров. "
        if TC["evaluated"] == 657
        else "Для контроля отобрано по 100 примеров стабильным хешированием идентификаторов. "
    )
    + f"В транслитерации медиана числа знаков — {compact(TD['input_lengths']['external']['cuneiform_signs_median'])} в новом тесте и {compact(TD['input_lengths']['legacy_control']['cuneiform_signs_median'])} в контроле. Длина, формат и редакционные соглашения различаются."
)
table(
    ["Задача", "Генерация и вход"],
    [
        [
            "MT",
            "4 луча; до 256 новых токенов; вход до 512 токенов; ASCII-ATF; дополнение справа с attention mask. Четыре длинных входа исключены.",
        ],
        [
            "Транслитерация",
            "Жадное декодирование; до 510 новых токенов; вход до 512 токенов с меткой SURFACE.",
        ],
        [
            "OCR",
            "Жадное декодирование; 32 токена на вырезку и 258 на прорисовку; квадратное дополнение до 384×384; нормализация 0,5/0,5.",
        ],
    ],
    [1.25, 5.65],
    9.5,
    left_columns=(1,),
)
p(
    "Вычисления выполнялись в eval и inference_mode, fp32. Транслитерация и OCR — на MPS; MT — 2 423 оценённых примера на MPS и 144 на CPU после возобновлений. Параметры генерации сохранялись. Скорость по этому запуску сравнивать нельзя.",
    small=True,
)
h("Границы выводов", level=2)
p(
    "У новых кандидатов не обнаружено пересечений с зарегистрированными исходными корпусами по доступным идентификаторам и текстам. Отсутствие этих примеров в обучении готовых моделей не доказано. Прежний контроль также мог входить в обучение."
)
p(
    "Unicode для транслитерации восстановлен из редакторских чтений, а не независимо размечен по фотографиям. Сквозные 341 связка имеют evaluation_ready=false. "
    + (
        "Новый пул транслитерации оценён целиком. Объём контроля указан в таблице охвата. Сопоставимая проверка новых целых табличек и сквозная оценка ещё не выполнены."
        if FULL_TRANSLIT
        else "Полный тест транслитерации из 43 213 строк и полноценная проверка новых целых табличек ещё не выполнены."
    )
)
h("Файлы эксперимента", level=2)
p("Все пути ниже даны относительно каталога проекта sumerian-baseline.", small=True)
for text in [
    ROOT_REL + "/analysis.json — сводка, версии пакетов и диагностика.",
    "Каталоги mt, translit и ocr внутри этого пути содержат predictions.jsonl, predictions_review.tsv, summary_full.json и manifest.json.",
    "manifest.json фиксирует ревизии моделей и SHA-256 тестовых файлов. execution_sessions.jsonl содержит историю возобновлений; mt/skipped_inputs.jsonl — четыре пропущенных входа.",
    (
        "runs/new_data/README.md"
        if FULL_TRANSLIT
        else (
            "legacy/docs/TRANSLITERATION_35PCT_2026-11-05.md"
            if SAMPLING
            else "legacy/docs/COLLECTION_EVALUATION_2026-11-04.md"
        )
    )
    + " — отчёт и ссылки на исходные предсказания.",
]:
    p(text, small=True)
p(
    f"Проверка полноты: {integer(TOTAL_PLANNED)} запланированных записей = {integer(TOTAL_EVALUATED)} предсказаний + {TOTAL_PLANNED-TOTAL_EVALUATED} пропуска. Ключи уникальны; метрики пересчитаны из сохранённых ответов. Регрессионные тесты сборки: 14 успешно.",
    small=True,
)

for task, title in [("mt", "Перевод по коллекциям"), ("translit", "Транслитерация по коллекциям")]:
    h(title, new=True)
    p(
        "Показатели рассчитаны по исходным строкам и дополнительно после нормализации. Малые группы дают нестабильные оценки; различия длины и нотации не позволяют трактовать таблицу как чистое сравнение сложности языка.",
        small=True,
    )
    rows = []
    for source, s in sorted(
        TASKS[task]["summary"]["by_source"].items(), key=lambda item: (-item[1]["planned"], item[0])
    ):
        rows.append(
            [
                source.replace("oracc/", ""),
                str(s["evaluated"]) + "/" + str(s["planned"]),
                number(s["raw"]["chrf"]),
                number(s["normalized"]["chrf"]),
                number(s["raw"]["cer"]),
            ]
        )
    table(
        ["Источник", "Оценено /\nплан", "chrF\nисходный ↑", "chrF\nнорм. ↑", "CER\nисходный ↓"],
        rows,
        [2.3, 1.1, 1.15, 1.15, 1.2],
        10,
    )
    if task == "mt":
        p(
            "Наибольшие группы: EISL — 988 сегментов, OBMC — 632, BLMS — 291. В ETCSL оценено 198 из 201 сегмента; ещё один пропуск относится к CMAWRO2. Пропуски вызваны выбранным лимитом входа 512 токенов."
        )
    else:
        by_source = TASKS["translit"]["summary"]["by_source"]
        p(
            f"Общий chrF {number(TR['raw']['chrf'])} описывает смесь представленных групп и не является средним арифметическим показателей источников. Например, chrF DCCLT/Ebla равен {number(by_source['oracc/dcclt/ebla']['raw']['chrf'])}, OBEL — {number(by_source['oracc/obel']['raw']['chrf'])}."
        )
        if BASELINE:
            p(
                f"На сохранённом первоначальном подмножестве из {integer(BASELINE['evaluated'])} строк chrF остаётся {number(BASELINE['raw']['chrf'])}. Это позволяет отделить изменение состава теста от изменения модели.",
                small=True,
            )

doc.save(OUT)
print(OUT)
