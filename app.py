import os
import argparse
from utils.featurizer import Featurizer
from utils.faiss_helper import FaissHelper


def create(path: str):
    # Init Faiss and Featurizer
    _id = 1
    featurizer = Featurizer()
    faiss = FaissHelper(128, "IDMap,Flat")

    # Default path for saving index and ref
    index_path = "assets/index"
    ref_path = "assets/ref"

    # Create directory if not exist
    if not os.path.exists("assets"):
        os.makedirs("assets")

    for filename in os.listdir(path):
        # Create vector from image
        img_path = os.path.join(path, filename)
        vector = featurizer.get_features(img_path)

        # Create reference
        ref = (filename, vector)

        # Add to index
        print("Adding {} to Faiss index...".format(filename))
        faiss.add_vector(vector, ref, _id)
        _id += 1

    faiss.save_index(index_path)
    faiss.save_reference(ref_path)


def search(path: str):
    # Init Faiss and Featurizer
    featurizer = Featurizer()
    faiss = FaissHelper(128, "IDMap,Flat")

    # Default path for loading index and ref
    index_path = "assets/index"
    ref_path = "assets/ref"

    # Load saved index and ref
    faiss.load_index(index_path)
    faiss.load_ref(ref_path)
    print(faiss.index.ntotal)
    # Get query feature vector(s)
    query_features = featurizer.get_features(path)

    # Perform faiss search
    faiss.search(query_features)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument(
        "Path",
        metavar="path",
        type=str,
        help=("Path of the image/directory to be used on index creation or query"),
    )

    parser.add_argument(
        "-c",
        "--create",
        action="store_true",
        help="Create a new index based on files in the specified directory",
    )

    args = parser.parse_args()
    path = args.Path

    # Check if it is a create or search operation
    if args.create:
        create(path)
    else:
        search(path)
