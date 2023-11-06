
from collections import defaultdict
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI


pending_clicks = defaultdict(int)
# 1 Словарь, который используется для отслеживания незавершенных кликов.
# Ключами словаря являются идентификаторы кликов (click_id),
# а значениями - идентификаторы предложений (offer_id), связанные с этими кликами.
offer_clicks = defaultdict(int)
# Связь оффера с кликом (например, offer_clicks) – это словарь
# который используется для подсчета количества кликов на каждом предложении.
# Ключами словаря являются идентификаторы предложений (offer_id),
# значениями - количество кликов.
offer_actions = defaultdict(int)
# 3. Действие по офферу (например, offer_actions) - это словарь
# который используется для подсчета количества конверсий (действий) на каждом предложении.
# Ключами словаря являются идентификаторы предложений (offer_id),
# значениями - количество конверсий.
offer_rewards = defaultdict(float)
# 4. Награда по офферу(например, offer_rewards )- это словарь,
# который используется для хранения суммарных вознаграждений на каждом предложении.
# Ключами словаря являются идентификаторы предложений (offer_id),
# значениями - суммарное вознаграждение.


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    pending_clicks.clear()
    offer_clicks.clear()
    offer_actions.clear()
    offer_rewards.clear()


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """_summary_

    Parameters
    ----------
    click_id : int
        _description_
    offer_ids : str
        _description_

    Returns
    -------
    dict
        _description_
    """
    offers_ids = [int(i) for i in offer_ids.split(",")]

    max_prob = 0
    offer_id = offers_ids[np.random.randint(0, max([len(offers_ids) - 1, 1]))]

    for i in offers_ids:
        curr_prop = 0
        a, b = offer_actions[i], offer_clicks[i]

        if a <= 0:
            a = 1
        if b <= 0:
            b = 1
        for _ in range(30):

            curr_prop += np.random.beta(a, b)

        if curr_prop > max_prob:
            offer_id = i

    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        }

    offer_clicks[offer_id] += 1
    pending_clicks[click_id] = offer_id

    return response


@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """_summary_

    Parameters
    ----------
    click_id : int
        _description_
    offer_ids : str
        _description_

    Returns
    -------
    dict
        _description_
    """
    offer_id = pending_clicks[click_id]
    offer_actions[offer_id] += int(reward > 0)
    offer_rewards[offer_id] += reward
    del pending_clicks[click_id]


    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": reward > 0,
        "reward": reward
    }

    return response


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """_summary_

    Parameters
    ----------
    click_id : int
        _description_
    offer_ids : str
        _description_

    Returns
    -------
    dict
        _description_
    """

    response = {
        "offer_id": offer_id,
        "clicks": offer_clicks[offer_id],
        "conversions": offer_actions[offer_id],
        "reward":  offer_rewards[offer_id],
        "cr": offer_actions[offer_id] / max([offer_clicks[offer_id], 1]),
        "rpc": offer_rewards[offer_id] / max([offer_clicks[offer_id], 1]),
    }
    return response


def main():
    uvicorn.run("app:app", host="localhost")


if __name__ == "__main__":
    main()






