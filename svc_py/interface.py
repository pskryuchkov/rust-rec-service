import numpy as np

from redis import Redis
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

import consts

class RedisIndex():
    def __init__(self, conn, prefix, name, dim, create=False):
        self.conn = conn
        self.prefix = prefix
        self.name = name
        self.dim = dim
        
        if create:
            self.create()
        
    def create(self):
        schema = (
            VectorField(
                "$.vec",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DISTANCE_METRIC": "COSINE",
                    "DIM": self.dim,
                },
                as_name="vector",
            ),
        )
        definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.JSON)
        res = self.conn.ft(self.name).create_index(
            fields=schema, definition=definition
        )
        
    def info(self):
        return self.conn.ft(self.name).info()

    def get(self, id):
        r = self.conn.json().get(f"{self.prefix}{id}", "$")
        if r:
            return r[0]['vec']

    def search(self, vec, n_top=3):
        query = (Query(f'(*)=>[KNN {n_top} @vector $query_vector AS vector_score]')
                 .sort_by('vector_score')
                 .dialect(2)
                )
        query_data = {'query_vector': np.array(vec, dtype=np.float32).tobytes()}
        return self.conn.ft(self.name).search(query, query_data).docs


if __name__ == "__main__":
    r = Redis(host=consts.REDIS_HOST, port=consts.REDIS_PORT)
    index = RedisIndex(r, prefix=consts.KEY_PREFIX, name=consts.INDEX_NAME, dim=consts.DIM)
    print(index.search(np.random.rand(consts.DIM)))