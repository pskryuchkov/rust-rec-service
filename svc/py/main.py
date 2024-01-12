from typing import Annotated
from fastapi import FastAPI, Path

from interface import MilvusIndex

app = FastAPI()
index = MilvusIndex(collection_name="tracks")


@app.get("/similar/{item_id}")
async def similar(item_id: Annotated[int, Path(ge=0)]):
    return {"neighbours": index.similar(id=item_id, limit=10)}
