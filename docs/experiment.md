# Протокол обучения и оценки

## Итоговая конфигурация

Датасет FER2013 разделён по официальному полю `Usage`: Training (28 709) для оптимизации весов, PublicTest (3 589) для выбора эпох и коэффициента смешивания, PrivateTest (3 589) для оценки. 
```powershell
python -m emotion_net.experiment --dataset fer --architecture resnet18_lbp --pretrained-weights data/weights/resnet18-f37072fd.pth --class-weights none --selection accuracy --epochs 30 --lr 0.0001 --seed 42 --output artifacts/resnet18_lbp
python -m emotion_net.experiment --dataset fer --architecture resnet18_lbp --pretrained-weights data/weights/resnet18-f37072fd.pth --class-weights none --selection accuracy --strong-augment --label-smoothing 0.1 --epochs 35 --lr 0.00003 --seed 42 --output artifacts/resnet18_lbp_regularized
python -m emotion_net.experiment --dataset fer --architecture resnet18_lbp --pretrained-weights data/weights/resnet18-f37072fd.pth --disable-lbp --class-weights none --selection accuracy --epochs 30 --lr 0.0001 --seed 42 --output artifacts/resnet18_no_lbp
python -m emotion_net.train_lbp --epochs 60 --seed 42 --output artifacts/lbp_expert
```

Первый checkpoint был получен с float16 autocast, последующие прогоны — с bfloat16; сохранённые веса и метрики относятся к фактическим прогонам. Архитектура одинаковая. `models/manifest.json` описывает итоговое смешивание двух CNN+LBP и LBP-классификатора. `models/no_lbp.json` и `models/ablation_fusion.json` задают контрольную пару с одинаковой CNN. Вес LBP 0,2 выбран на PublicTest среди фиксированных значений 0; 0,05; 0,1; 0,15; 0,2; 0,25; 0,3; 0,4. Для итогового ансамбля проверены веса 0; 0,1; 0,2; 0,3, выбран 0,2 по validation accuracy.

```powershell
python -m emotion_net.evaluate_fer --checkpoint models/manifest.json --split val --output artifacts/final
python -m emotion_net.evaluate_fer --checkpoint models/manifest.json --split test --output artifacts/final
python -m emotion_net.evaluate_fer --checkpoint models/no_lbp.json --split test --output artifacts/ablation_no_lbp
python -m emotion_net.evaluate_fer --checkpoint models/ablation_fusion.json --split test --output artifacts/ablation_fusion
python -m emotion_net.evaluate --ck-archive "data/CK+.zip" --checkpoint models/manifest.json --output artifacts/ck_final
```

Несколько моделей в ходе исследования оценивались на PrivateTest. Поэтому результаты этого раздела являются исследовательскими, а не единственной заранее зафиксированной проверкой. Влияние LBP сообщается и для неудачного раннего объединения, и для успешного позднего смешивания. CK+ CSV использован только для внешней оценки; подбор по нему не проводился.

## Исходная компактная модель

1. Зафиксировать seed (по умолчанию 42) и исходное разделение FER2013: Training / PublicTest / PrivateTest. Настройка и выбор лучшей эпохи идут только по PublicTest; PrivateTest читается для финальной оценки.
2. Обучить модель на FER2013 (`--dataset fer`). Используются AdamW, взвешенная cross-entropy с весами обратного квадратного корня частоты класса, cosine learning-rate schedule, умеренные горизонтальное отражение, поворот и изменение яркости. LBP считается **после** аугментации того же изображения, что поступает в CNN.
3. С имеющимся CK+ CSV провести внешнюю оценку лучшего FER2013 checkpoint по всем 902 совместимым строкам без дополнительного обучения и без выбора гиперпараметров на CK+. При получении исходных последовательностей с ID участников можно провести отдельное дообучение (`--dataset ck --init artifacts/fer/best.pt`) с разделением по людям.
4. Сохранить финальные accuracy, macro F1, balanced accuracy, per-class precision/recall/F1 и матрицу ошибок. Графики обучения строятся из реально измеренной истории; метрики публикации не используются как целевые значения.

Примеры команд:

```powershell
python -m emotion_net.experiment --dataset fer --fer-archive "data/archive (2).zip" --epochs 30 --batch-size 128
python -m emotion_net.evaluate --ck-archive "data/CK+.zip" --checkpoint artifacts/fer/best.pt
python -m emotion_net.experiment --dataset ck --init artifacts/fer/best.pt --epochs 12 --batch-size 64
python -m emotion_net.infer path/to/face.png --checkpoint artifacts/ck/best.pt
```

Для контрольного сравнения можно отключить веса классов и выбирать эпоху по accuracy: `--class-weights none --selection accuracy --output artifacts/fer_accuracy`. Этот вариант сохранён отдельно, чтобы не перезаписывать основную модель.

Выходы обучения: `split_summary.json`, `history.json`, `metrics.json`, `best.pt`, `learning_curves.png`, `confusion_matrix.png` в `artifacts/<dataset>/`. 
CK+ отличается от FER2013 по условиям съёмки и размеру, а производные neutral метки CK+ менее надёжны. Метрики двух наборов нельзя напрямую трактовать как одну общую точность системы.
Macro F1 и balanced accuracy усредняются по классам, присутствующим в данном тестовом разделе; `per_class.support` показывает, какие классы там фактически есть.
