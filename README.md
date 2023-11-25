# Recsys with @hitrate5 0.581
Целью проекта является создание сервиса рекомендательной системы, в котором буду применяться две модели предсказания для дальнейшего проведения A/B тестов. Качество моделей оценивается по @hitrate5, контрольная модель имеет качество выше 0.53, а тестовая больше 0.57.

## Этапы работы
 - Анализ и обработка данных
 - Обучение моделей (тестовой и контрольной)
 - Создание сервиса для получения рекомендаций

## Резульататы
В результате выполнения проекта, было обучено две модели - контрольная и тестовая. В процессе обучения были рассмотрены 3 основные модели Catboost, XGBoost, Lightgbm лучше всех в выполнении поставленной задачи справился Catboost, показав @hitrate5 0.581, в то время как XGBoost и Lightgbm не смогли превысить результат выше чем @hitrate5 0.54. В результате, в работе сервиса участвуют две модели Catboost. 

## Окружение
Проект собирался на python 3.8

```
numpy == 1.23.5
pandas == 1.3.5
matplotlib == 3.6.2
seaborn == 0.12.1
sklearn == 0.0.post1
catboost == 1.2
category-encoders == 2.6.2
sqlalchemy == 1.4.48
fastapi == 0.95.1
tqdm == 4.64.1
loguru == 0.7.0
```

## Структура проекта

 - Таблицы с информацией о пользователях и постах находятся в папке `data`
 - В папке `preprocessing_data` лежат обученные модельки и ноутбуки с решением
 - Сервис находится в общей папке с названием `app_final`

## Описание таблицы user_data

Cодержит информацию о всех пользователях соц.сети

| Название | Описание |
| ------ | ------ |
| age | Возраст пользователя (в профиле) |
| city | Город пользователя (в профиле) |
| country | Страна пользователя (в профиле) |
| exp_group | Экспериментальная группа: некоторая зашифрованная категория |
| gender | Пол пользователя |
| user_id | Уникальный идентификатор пользователя |
| os | Операционная система устройства, с которого происходит пользование соц.сетью|
|source| Пришел ли пользователь в приложение с органического трафика или с рекламы |


## Описание таблицы post_text_df

Содержит информацию о постах и уникальный ID каждой единицы с соответствующим ей текстом и топиком

| Название | Описание |
| ------ | ------ |
| id | Уникальный идентификатор поста |
| text | Текстовое содержание поста |
| topic | Основная тематика |

## Описание таблицы feed_data

Содержит историю о просмотренных постах для каждого юзера в изучаемый период.

| Название | Описание |
| ------ | ------ |
| timestamp | Время, когда был произведен просмотр |
| user_id | id пользователя, который совершил просмотр |
| post_id | id просмотренного поста |
|action| Тип действия: просмотр или лайк |
|target| 1 у просмотров, если почти сразу после просмотра был совершен лайк, иначе 0. У действий like пропущенное значение.|
