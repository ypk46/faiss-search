# Faiss Search - Sample Project

This is a sample project to understand and try out Faiss.

## Requirements

- Python 3.6.x
- OpenCV 3.4.2 (Official installation, the **opencv-python** unofficial package does not include SIFT which is used in this project )

## Running the project

### 1. Install dependencies

    pip install -r requirements.txt

### 2. Create a Faiss index and reference dictionary

    python app.py -c <directory with images>

### 3. Search for similar images

    python app.py <image path>
