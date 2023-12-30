from typing import Annotated

import redis
import numpy as np
from fastapi import FastAPI, Path

import consts
from interface import RedisIndex

app = FastAPI()
r = redis.Redis(host=consts.REDIS_HOST, port=consts.REDIS_PORT)
index = RedisIndex(r, prefix=consts.KEY_PREFIX, name=consts.INDEX_NAME, dim=consts.DIM)


@app.get("/search/{item_id}")
async def search(item_id: Annotated[int, Path(ge=0)]):
    target = index.get(item_id)
    r = [neighbour['id'].replace(consts.KEY_PREFIX, "") for neighbour in index.search(target)]
    return {"neighbours": r}