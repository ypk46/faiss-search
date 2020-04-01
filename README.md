# Faiss Search - Sample Project

This is a sample project to understand and try out Faiss.

## Requirements

- Python 3.6.x
- OpenCV 3.4.2 (Official installation, the **opencv-python** unofficial package does not include SIFT which is used in this project )

## Running the project

### 1. Install dependencies

    pip install -r requirements.txt

### 2. Create a Faiss index and reference dictionary

    python app.py -c -i <path to store index> -r <path to store reference object> <directory with images>

### 3. Search for similar images

    python app.py -i <path to load index> -r <path to load reference object> <image path>

### 4. Update an existing Faiss index

    python app.py -u -i <path to store index> -r <path to store reference object> <directory with images or path to single image>

## Examples

```bash
    # Creates a new index and ref object
    python app.py -c -i assets/index -r assets/ref dataset

    # Updates an existing index and ref object
    python app.py -u -i assets/index -r assets/ref img/0001.jpg

    # Search for similar images
    python app.py -i assets/index -r assets/ref img/0002.jpg
```
