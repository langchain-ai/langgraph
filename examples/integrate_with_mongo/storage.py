from pymongo import MongoClient


class MongoDBHandler:
    def __init__(self, uri='mongodb://root:example@localhost:27017/'):
        self.uri = uri

    def _connect(self):
        return MongoClient(self.uri)

    def insert_upsert_dictionary(self, db_name, collection_name, data_dict):
        """
        Inserts or updates a dictionary in MongoDB.

        :param db_name: Name of the database
        :param collection_name: Name of the collection
        :param data_dict: Dictionary to store or update in the collection
        :return: The result of the upsert operation
        """
        client = self._connect()
        db = client[db_name]
        collection = db[collection_name]

        if _id := data_dict.get('_id'):
            filter = {'_id': _id}
            result = collection.update_one(filter, {"$set": data_dict}, upsert=True)
            client.close()
            return _id, result.acknowledged
        else:
            result = collection.insert_one(data_dict)
            client.close()
            return result.inserted_id, result.acknowledged

    def load_dictionary(self, db_name, collection_name, query):
        """
        Loads a dictionary from MongoDB based on a query.
        :param db_name: Name of the database
        :param collection_name: Name of the collection
        :param query: Query to filter documents
        :return: The first dictionary that matches the query
        """
        client = self._connect()
        db = client[db_name]
        collection = db[collection_name]

        document = collection.find_one(query)
        client.close()

        return document

# Example usage:
# db_handler = MongoDBHandler()
# result = db_handler.insert_upsert_dictionary('mydatabase', 'mycollection', {'_id': 1, 'name': 'Alice'})
# document = db_handler.load_dictionary('mydatabase', 'mycollection', {'_id': 1})
# print(result)
# print(document)
