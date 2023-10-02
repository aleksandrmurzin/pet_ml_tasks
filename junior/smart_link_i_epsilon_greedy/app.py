import numpy as np
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/sample/")
def sample(offer_ids: str) -> dict:
    offers = list(map(int, offer_ids.split(",")))
    random_id = np.random.random_integers(0, len(offers) -1)
    return {"offer_id": offers[random_id]}

def main():
    uvicorn.run("app:app", host="localhost")

if __name__ == "__main__":
    main()