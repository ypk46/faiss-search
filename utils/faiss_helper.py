import faiss
import heapq
import pickle
import logging
import numpy as np


class FaissHelper:
    def __init__(self, dimensions: int, index_type: str):
        self.dim = dimensions
        self.index = faiss.index_factory(dimensions, index_type)
        self.reference = {}

    def save_index(self, path: str):
        faiss.write_index(self.index, path)

    def load_index(self, path: str):
        self.index = faiss.read_index(path)

    def save_reference(self, path: str):
        ref_file = open(path, "wb+")
        try:
            pickle.dump(self.reference, ref_file, True)
        except EnvironmentError as e:
            logging.error("Failed to save index file error:[{}]".format(e))
        except RuntimeError as v:
            logging.error("Failed to save index file error:[{}]".format(v))
        ref_file.close()

    def load_ref(self, path: str):
        ref_file = open(path, "rb")
        self.reference = pickle.load(ref_file)
        ref_file.close()

    def add_vector(self, feature_vector, ref: tuple, _id: int):

        vector_x, vector_y = feature_vector.shape

        # Check matching dimensions
        if vector_y == self.dim and not self.is_duplicate(feature_vector):
            # Generate id matrix
            ids = np.linspace(_id, _id, num=vector_x, dtype="int64")

            # Store the object reference
            self.reference.update({_id: ref})

            # Add vector(s) with id
            self.index.add_with_ids(feature_vector, ids)
        else:
            print("Image is duplicate or have a mismatch feature dimension")

    def is_duplicate(self, feature_vector):
        distances, neighbors = self.index.search(feature_vector, k=1)
        x, y = neighbors.shape

        for i in range(x):
            distances_list = distances[i].tolist()
            if distances_list[0] != 0:
                return False
        return True

    def search(self, query, k: int = 5):
        # Get distance and closest neighbors
        distances, neighbors = self.index.search(query, k=k)
        x, y = neighbors.shape

        if x > 1:
            # Multiple features per image
            result_dict = {}
            results = []

            # Get unique matches
            for i in range(x):
                unique = np.unique(neighbors[i]).tolist()
                for _id in unique:
                    if _id != -1:
                        score = result_dict.get(_id, 0)
                        score += 1
                        result_dict[_id] = score

            # Only take into account 5 or more feature matches
            for key in result_dict:
                matches = result_dict[key]
                if matches >= 5:
                    if len(results) < 10:
                        heapq.heappush(results, (matches, key))
                    else:
                        heapq.heappushpop(results, (matches, key))

            # Get result list
            result_list = heapq.nlargest(10, results, key=lambda x: x[0])

            if len(result_list) == 0:
                print("No result found")
            else:
                for result in result_list:
                    index_id = result[1]
                    matches = result[0]
                    filename = self.reference[index_id][0]
                    print(matches, " features match: ", filename)

        elif x == 1:
            # Single feature per image
            distances_list = distances[0].tolist()
            neighbors_list = neighbors[0].tolist()

            for i in range(k):
                index_id = neighbors_list[i]
                distance = distances_list[i]
                filename = self.reference[index_id][0]
                print(distance, ":", filename)
        else:
            print("No result found")
