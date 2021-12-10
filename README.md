# Распознавание выражений лица: CNN + LBP

Проект распознавания семи классов эмоций. Изображение лица 48×48 поступает в предобученную ResNet18, а пространственная гистограмма LBP обрабатывается отдельной ветвью. Финальный классификатор объединяет две CNN+LBP модели с отражением изображения и независимый LBP-классификатор

## Результаты

| Проверка |   Accuracy |   Macro F1 |
|---|-----------:|-----------:|
| FER2013 PublicTest, 3 589 |     70,91% |     69,90% |
| FER2013 PrivateTest, 3 589 | **71,58%** | **70,57%** |
| CK+ CSV, внешняя проверка, 902 |     78,16% |     50,34% |

## Запуск

Python 3.10+

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python -m emotion_net.ui --checkpoint models/manifest.json
```

Предсказание для одного изображения:

```powershell
python -m emotion_net.infer path/to/face.png --checkpoint models/manifest.json
```

Обученные веса в `models/` позволяют запустить интерфейс без датасетов. Для повторения обучения положите `fer2013.csv` в `data/raw/fer2013.csv`.

```powershell
python -m emotion_net.experiment --dataset fer --architecture resnet18_lbp --pretrained-weights data/weights/resnet18-f37072fd.pth --class-weights none --selection accuracy --output artifacts/resnet18_lbp
python -m emotion_net.train_lbp --output artifacts/lbp_expert
python -m emotion_net.evaluate_fer --checkpoint models/manifest.json --split test --output artifacts/evaluation
```