from pymilvus import connections, Collection


class MilvusIndex:
    def __init__(
        self,
        collection_name,
        id_field="id",
        embeddings_field="embeddings",
        host="localhost",
        port="19530",
        n_probe=10,
        metric_type="L2",
        exclude_first=True,
    ):
        connections.connect("default", host=host, port=port)
        self.collection = Collection(collection_name)
        self.collection.load()
        self.embeddings_field = embeddings_field
        self.id_field = id_field
        self.search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": n_probe},
        }
        self.exclude_first = exclude_first

    def similar(self, id, limit):
        res = self.collection.query(
            expr=f"id in [{id}]",
            output_fields=[self.embeddings_field],
        )
        target_vec = [res[0][self.embeddings_field]]

        resp = self.collection.search(
            target_vec,
            self.embeddings_field,
            self.search_params,
            limit=limit + 1,
            output_fields=[self.id_field],
        )
        ids = [r.entity.get(self.id_field) for r in resp[0]]
        if self.exclude_first:
            ids = ids[1:]
        else:
            ids = ids[:limit]
        return ids
