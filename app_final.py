import os
from fastapi import FastAPI
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import warnings
from catboost import CatBoostClassifier
import hashlib
import datetime
import loguru
import tqdm
warnings.warn("ignore")


def get_model_path(path: str) -> str:
    """Получение пути к модельке"""

    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
        return MODEL_PATH
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models(mode: str = None):
    """Загрузка моделей
        mode = `"control"` и `"test"`
    """

    PATH = "C:/Users/Daniil/Desktop/Курсы/courses/final project/22 task Рекомендательные системы/preprocessing_data"

    if mode == "control": # выбор контрольной модели 
        model_path = get_model_path(PATH + str("/catboost_65_control"))
        
        if model_path == PATH + str("/catboost_65_control"):    # Проверка локальный путь или нет
            for _ in tqdm.tqdm(range(1),desc="загрузка модели"):
                model = CatBoostClassifier().load_model(model_path) # загрузка модели
            return model
        
        elif model_path != PATH + str("/catboost_65_control"):
            for _ in tqdm.tqdm(range(1),desc="загрузка модели"):    
                model = CatBoostClassifier().load_model(model_path + str("_control")) # загрузка модели
            return model
    
    elif mode == "test": # выбор тестовой модели
        model_path = get_model_path(PATH + str("/catboost_67"))
        
        if model_path == PATH + str("/catboost_67"):            # Проверка локальный путь или нет
            for _ in tqdm.tqdm(range(1),desc="загрузка модели"):    
                model = CatBoostClassifier().load_model(model_path) # загрузка модели
            return model
        
        elif model_path != PATH + str("/catboost_67"):
            for _ in tqdm.tqdm(range(1),desc="загрузка модели"):    
                model = CatBoostClassifier().load_model(model_path + str("_test")) # загрузка модели
            return model


def batch_load_sql(query: str) -> pd.DataFrame:
    """Загрузка таблицы чанками"""

    CHUNKSIZE = 200000
    engine = create_engine(
        "----"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features(mode:str) -> pd.DataFrame:
    """Загрузка фичей: таблица пользователей, постов или эмбеддинги
    `mode` = `"prep_user"`, `"post"`, `"prep_posts"`
    """
    
    if mode == "prep_user":
        query = """SELECT * FROM daniil_gorenkov_prep_users_lesson_22"""
        df = batch_load_sql(query)
        return df

    elif mode == "post":
        query = """SELECT * FROM public.post_text_df;""" # менять фичи
        df = batch_load_sql(query)
        return df
    
    elif mode == "prep_posts":
        query = """SELECT * FROM daniil_gorenkov_prep_posts_lesson_22""" # запрос на таблицу эмбеддингов
        df = batch_load_sql(query)
        return df


def search_user(id:int,data:pd.DataFrame) -> pd.Series:
    """Поиск пользователя в датафрейме"""
    user_df = data[data["user_id"]==id]
    return user_df


def concat_and_order(user_df:pd.DataFrame,
                     posts_df: pd.DataFrame,
                     time:datetime.datetime,
                     mode:str) -> pd.DataFrame:
    """Объединение таблицы пользователя с таблицей постов для получения общего фрейма
    по принципу растягивания одного пользователя на все посты и организация порядка колонок
    mode = `"test"` и `"control"`
    """
    posts = posts_df.copy()
    user = user_df.copy()  
    
    if mode == "test":
        order = ["user_id",'day_of_week', 'day_hour', 'month', 'gender', 'age', 'country', 'city',
            'exp_group', 'os', 'source', 'emb_0', 'emb_1', 'emb_2', 'emb_3',
            'emb_4', 'emb_5', 'emb_6', 'emb_7', 'emb_8', 'emb_9', 'topic_covid',
            'topic_entertainment', 'topic_movie', 'topic_politics', 'topic_sport',
            'topic_tech']
        
    elif mode == "control":
        order = [
            "user_id",'day_of_week', 'day_hour', 'month', 'gender', 'age', 'country', 'city',
            'exp_group', 'os', 'source', 'topic_covid', 'topic_entertainment',
            'topic_movie', 'topic_politics', 'topic_sport', 'topic_tech'
        ]

    for col in range(len(user.columns)):
        # print(f"col name {col} and type {user[col].dtype}")
        posts.insert(0,user.columns[col],user.values[0][col])
        
    # for i in user.columns:
    #     if i != "age":
    #         posts[i] = posts[i].astype(int)

    posts["day_of_week"] = time.weekday()
    posts["day_hour"] = time.hour
    posts["month"] = time.month
    
    posts = posts[order]

    return posts.drop("user_id", axis=1)


def conv(df:pd.DataFrame, exp_group:str) -> dict:
    """Преобразования предсказанных постов в формат ответа"""
    ls = []

    for i in range(len(df.index)):
        seq = df.iloc[i,:]
        seq.rename({"post_id":"id"},inplace=True)
        ls.append(seq.to_dict())

    return {"exp_group":exp_group,
            "recommendations":ls}


def get_recommended_posts(posts:pd.DataFrame,predictions:np.array,limit:int) -> pd.DataFrame:
    """Склеивание предсказаний по постам и их сортировка"""
    
    posts = posts.copy()
    posts["predictions"] = predictions
    posts_sorted = posts.sort_values("predictions",ascending=False).iloc[:limit,:]
    predicted_posts = posts_sorted[["post_id","text","topic"]]
    
    return predicted_posts


def get_exp_group(user_id: int) -> str:
    """Определение группы пользователя"""
    
    sault = "A_B_test" 
    ciento = 100    

    value = str(user_id) + sault

    hashed_id = int(hashlib.md5(value.encode()).hexdigest(),16)

    percent = hashed_id % ciento

    if percent < 50:
        return "control"
    
    elif percent >= 50:
        return "test"


### ЗАГРУЗКА ВНЕ ЭНДПОИНТА


loguru.logger.info("Загрузка контрольной модели")
model_control = load_models(mode="control")
loguru.logger.info("---------------------------")


loguru.logger.info("Загрузка тестовой модели")
model_test = load_models(mode="test")
loguru.logger.info("З---------------------------")


loguru.logger.info("Загрузка фичей пользователей")
prep_user = load_features("prep_user").drop("index", axis=1)
loguru.logger.info("---------------------------")


loguru.logger.info("Загрузка постов")
posts = load_features("post")
loguru.logger.info("---------------------------")


loguru.logger.info("Загрузка предобработанных постов")
prep_posts = load_features("prep_posts").drop("index", axis=1)
loguru.logger.info("---------------------------")


### КОНЕЦ ПРЕДВАРИТЕЛЬНЫХ ЗАГРУЗОК


app = FastAPI()

@app.get("/post/recommendations/") # эндпоинт
def recommended_posts(id: int, time: datetime.datetime, limit: int = 5):

    exp_group = get_exp_group(id)
    loguru.logger.info(f"Пользователь попал в группу: {exp_group}")

    loguru.logger.info("Поиск среза пользователя в таблице")
    user = search_user(208,prep_user) 
    loguru.logger.info("---------------------------")

    if exp_group == "control":
        loguru.logger.info("Выбрана контрольная модель предсказаний")
        X = concat_and_order(user,prep_posts,time,mode=exp_group)
        preds = model_control.predict_proba(X)[:,1]
        loguru.logger.info("---------------------------")

    elif exp_group == "test":
        loguru.logger.info("Выбрана тестовая модель предсказаний")
        X = concat_and_order(user,prep_posts,time,mode=exp_group)
        preds = model_test.predict_proba(X)[:,1]
        loguru.logger.info("---------------------------")

    content = get_recommended_posts(posts,preds,limit)
    
    return  conv(content, exp_group) # тут 5 предиктов
 