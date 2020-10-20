# text_classifier

# как установить

git clone https://github.com/SiarheiKachatkou/text_classifier.git text_classifier\
cd text_classifier\
pip install -r requirements.txt\

# Структура репозитория

configs - файлы конфигураций запуска \
    base - базовые конфиги, из которых составляются более сложные конфигурации \
    eda_config.py- конфигурация для explanatory data analysis \
    train_traditional_config.py - конфигурация для обучения и тестирования традиционного подхода TFID+RandomForest \
    train_modern_config.py - современный подход BERT + Boosting

data - файлы данных\

feature_extractors.py - базовый класс FeatureExtractor и его конкретные реализации TFIDFFeatureExtractor, BERTFeatureExtractor для извлечения признаков текста\

classifier_heads.py базовый класс ClassifierHead и наследники для классификации текста по извлеченным признакам\

explanatory_data_analysis.py - скрипт для первичного анализац данных\

main.py - запус тренировки и кросс-валидации модели, конфигурация которой задачется в выбранном конфиге.

# Параметры конфигурации

is_debug - если True то запускается облегченный режим для дебага: однопоточность, загружаем всего 100 записей данных\
work_dir - папка куда будут складываться обученные модели и записываться метрики в metric.txt
classifier_head_str - строка, при исполнении которой будет вызваться конструктор наследника ClassifierHead с нужными параметрами
feature_extractor_str - строка, при исполнении которой будет вызваться конструктор наследника FeatureExtractor с нужными параметрами

# Что можно улучшить

# Machine Learning:

Визуализировать предложения, на которых классифиатор ошибается. Попытаться понять почему происходит ошибка. Возможно, это подскажет как сделать feature-engineering лучше, или найдем неправильно размеченные предложения.\
Увеличить датасет. Найти размеченные датасеты схожей тематики, или разметить руками.\
Использовать Transfer Leaning для задач sentiment analysis. Скорее всего, есть много хороших предобученных моделей для этой стандратной задачи (классификация твитов например). Нам нужно сделать fine-tuning классификатора на нашем специфичном датасете.\

# Programming:

использовать gpu и батчи для вычисления эмбеддингов от BERT\
делать кроссвалидацию в multiprocessing варианте\
для версионирования натренированных моделей использовать dvc\
сделать логировнние\
 

# Ссылки 
BERT tutorial
https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/




