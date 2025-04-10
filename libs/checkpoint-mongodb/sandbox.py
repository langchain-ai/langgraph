import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection

# from langgraph.store.base import Item, PutOp
# from langgraph.store.mongodb import MongoDBStore
# from langgraph.store.postgres import PostgresStore


MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "long_term_memory"


t0 = (datetime(2025, 4, 7, 17, 29, 10, 0),)
TTL = 1200  # Needs to be a fixed value


def insert_files1(base_path: Path, collection: Collection, ttl=TTL):
    # for file_path in base_path.rglob("*.json"):
    #     try:
    #         with file_path.open() as f:
    #             content = json.load(f)
    for file_path in base_path.rglob("*.*"):
        try:
            with file_path.open() as f:
                content = f.read()

            stat = file_path.stat()
            rel_path = file_path.relative_to(base_path)
            path_parts = list(rel_path.parts[:-1])  # Only folders
            filename = rel_path.name

            collection.find_one_and_update(
                {"namespace": path_parts, "key": filename},
                {
                    "$set": {
                        "value": content,
                        "updated_at": datetime.now(tz=timezone.utc),
                    },
                    "$setOnInsert": {
                        "created_at": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        )
                    },
                },
                upsert=True,
            )
            print(f"Inserted: {rel_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def insert_files2(base_path: Path, collection: Collection, ttl=TTL):
    for file_path in base_path.rglob("*.*"):
        try:
            with file_path.open() as f:
                content = f.read()

            stat = file_path.stat()
            rel_path = file_path.relative_to(base_path)
            path_parts = list(rel_path.parts[:-1])  # Only folders
            filename = rel_path.name

            op = UpdateOne(
                filter={"namespace": list(path_parts), "key": filename},
                update={
                    "$set": {
                        "value": content,
                        "updated_at": datetime.now(tz=timezone.utc),
                    },
                    "$setOnInsert": {
                        "created_at": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ),
                    },
                },
                upsert=True,
            )

            collection.bulk_write([op])

            print(f"Inserted: {rel_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def main():
    client = MongoClient("localhost", 27017)
    collection = client[DB_NAME][COLLECTION_NAME]
    collection.delete_many({})

    # index on namespace array and key strings
    collection.create_index([("namespace", ASCENDING)])
    collection.create_index("key")

    base_path = Path("/Users/casey.clements/src/langgraph/libs/checkpoint")
    t = time.perf_counter()
    insert_files1(base_path, collection, ttl=TTL)
    t1 = time.perf_counter()
    t1 = t1 - t
    print(f"{t1=}")

    print(f"{collection.count_documents({})=}")
    print(f"{collection.find({}, {'namespace':1}).to_list()}=")
    print(f"{collection.find_one({})=}")
    time.sleep(1)
    t = time.perf_counter()
    insert_files1(base_path, collection, ttl=TTL)
    t2 = time.perf_counter()
    t2 = t2 - t
    print(f"{t2=}")
    print(f"{collection.count_documents({})=}")
    print(f"{collection.find({}, {'namespace': 1}).to_list()}=")
    print(f"{collection.find_one({})=}")
    print(f"{t1=}, {t2=}")

    # for i in range(3):
    #     item = Item(key=str(i), namespace=namespaces[i], value=dict(i=i), created_at=t0, updated_at=t0)
    #     operations.append(PutOp(namespace=item.namespace, key=item.key, value=item.value, index=False, ttl=10))


if __name__ == "__main__":
    main()
