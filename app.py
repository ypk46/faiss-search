import os
import argparse
from utils.featurizer import Featurizer
from utils.faiss_helper import FaissHelper


def add_from_dir(path: str, featurizer, faiss):
    _id = 1
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

    return faiss


def create(path: str, index_path: str, ref_path: str):
    # Init Faiss and Featurizer
    featurizer = Featurizer()
    faiss = FaissHelper(128, "IDMap,Flat")

    # Add vectors from directory
    faiss = add_from_dir(path, featurizer, faiss)

    # Persist index and ref
    faiss.save_index(index_path)
    faiss.save_reference(ref_path)


def update(path: str, index_path: str, ref_path: str, is_dir: bool = False):
    # Init Faiss and Featurizer
    featurizer = Featurizer()
    faiss = FaissHelper(128, "IDMap,Flat")

    # Load saved index and ref
    faiss.load_index(index_path)
    faiss.load_ref(ref_path)

    if is_dir:
        # Add vectors from directory
        faiss = add_from_dir(path, featurizer, faiss)
    else:
        _id = max(faiss.reference.keys()) + 1
        # Get vectors from file
        vector = featurizer.get_features(path)
        ref = (path.split("/")[-1], vector)

        # Add vectors to index
        print("Adding {} to Faiss index...".format(path.split("/")[-1]))
        faiss.add_vector(vector, ref, _id)

    # Persist index and ref
    faiss.save_index(index_path)
    faiss.save_reference(ref_path)


def search(path: str, index_path: str, ref_path: str):
    # Init Faiss and Featurizer
    featurizer = Featurizer()
    faiss = FaissHelper(128, "IDMap,Flat")

    # Load saved index and ref
    faiss.load_index(index_path)
    faiss.load_ref(ref_path)

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

    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Create a new index based on files in the specified directory",
    )

    parser.add_argument(
        "-i", "--index", required=True, help="Path to an existing index",
    )

    parser.add_argument(
        "-r", "--reference", required=True, help="Path to an existing reference",
    )

    args = parser.parse_args()
    path = args.Path
    index_path = args.index
    ref_path = args.reference

    # Check if it is a create or search operation
    if args.create:
        create(path, index_path, ref_path)
    elif args.update:
        update(path, index_path, ref_path, os.path.isdir(path))
    else:
        search(path, index_path, ref_path)
