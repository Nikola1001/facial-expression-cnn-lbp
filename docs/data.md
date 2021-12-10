# Данные

## FER2013

`data/raw/fer2013.csv` с полями `emotion`, `pixels`, `Usage`. Поле `Usage` должно содержать `Training`, `PublicTest`, `PrivateTest`; исходное разделение сохраняется. Коды эмоций: 0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral.

## CK+
 
`data/raw/ckplus` с изображениями последовательностей вида `S005_001_00000011.png` и файлами меток `*_emotion.txt` в каталоге `Emotion`. Допустимы вложенные каталоги исходного архива; загрузчик ищет их рекурсивно.
