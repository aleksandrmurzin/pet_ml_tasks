"""
Начнём с простейшей модели: будем показывать случайные баннеры на любой клик, который придёт.
Наша модель состоит из двух стадий: отбор кандидатов
(те офферы, на которые пользователь физически может перейти и подписаться, в зависимости от его географии,
устройства, источника и т.д.) и выбор оффера (то, что мы будем реализовывать).

Кандидатная модель уже реализована.
В наш ML-сервис поступает сразу клик (click ID) и набор офферов-кандидатов (offer IDs).
Мы хотим брать случайный из них. Для тестирования интеграции с бекендом этого более чем достаточно.
"""


"""
FastAPI. Нужно знать основы работы с FastAPI, такие как создание маршрутов с помощью декораторов @app.get и @app.put  и обработка параметров запроса.
Обработка событий "startup",  с помощью декоратора @app.on_event("startup") для определения функции, которая будет выполняться при запуске приложения и очищать статистик перед каждым запуском.
Также  может пригодится знание defaultdict и понимание, как взаимодействовать  со словарями. В целом остальное все, что вы уже испольовали в других задачах.
"""

from collections import defaultdict
import numpy as np
import uvicorn
from fastapi import FastAPI

app = FastAPI()

pending_clicks = defaultdict(int) # 1. Рекомендации для объявления (например, pending_clicks)- это словарь, который используется для отслеживания незавершенных кликов.
                    # Ключами словаря являются идентификаторы кликов (click_id), а значениями - идентификаторы предложений (offer_id), связанные с этими кликами.
offer_clicks = defaultdict(int)  # 2. Связь оффера с кликом (например, offer_clicks) – это словарь, который используется для подсчета количества кликов на каждом предложении.
                    # Ключами словаря являются идентификаторы предложений (offer_id), а значениями - количество кликов.
offer_actions = {} # 3. Действие по офферу (например, offer_actions) - это словарь который используется для подсчета количества конверсий (действий) на каждом предложении.
                    # Ключами словаря являются идентификаторы предложений (offer_id), а значениями - количество конверсий.
offer_rewards = {} # 4. Награда по офферу(например, offer_rewards )- это словарь, который используется для хранения суммарных вознаграждений на каждом предложении.
                    # Ключами словаря являются идентификаторы предложений (offer_id), а значениями - суммарное вознаграждение.


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """Greedy sampling"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    if len(pending_clicks) < 100:
        idx = np.random.random_integers(0, len(offers_ids) -1)
        offer_id = offers_ids[idx]


        pending_clicks[click_id] = offer_id
        offer_clicks[offer_id] += 1

        response = {
            "click_id": click_id,
            "offer_id": offer_id,
            "sampler": "random",
        }
        return response


    offer_id = offers_ids[0]
    pending_clicks[click_id] = offer_id
    offer_clicks[offer_id] += 1

    response = {
            "click_id": click_id,
            "offer_id": offer_id,
            "sampler": "greedy",
    }
    return response

    # Sample top offer ID
    # Prepare response



@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:

    response = {
        "click_id": click_id,
        "offer_id": pending_clicks[click_id],
        "is_converstion": reward > 0,
        "reward": int(reward)
    }

    return response

@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    response = {
        "offer_id": offer_id,
        "clicks": 0,
        "conversions": 0,
        "reward": 0,
        "cr": 0,
        "rpc": 0,
    }
    return response


def main():
    uvicorn.run("app:app", host="localhost")

if __name__ == "__main__":
    main()






