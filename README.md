# SIFT-Based Texture Image Retrieval System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Table of Contents
- [About](#about)
- [Key Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Command Line Interface](#command-line-interface)
- [Dataset](#dataset)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [License](#license)

## 🔍 About <a name="about"></a>

This project implements an advanced image retrieval system for texture images using SIFT (Scale-Invariant Feature Transform) features. The system extracts SIFT descriptors from images, builds a visual vocabulary using K-means clustering, and represents images using either Bag of Words (BoW) or TF-IDF approaches for efficient similarity search.

## ⭐ Key Features <a name="features"></a>

- **SIFT Feature Extraction**: Robust detection of key points and descriptors in texture images
- **Visual Vocabulary Creation**: K-means clustering for creating a visual word dictionary
- **Image Representation**:
  - Bag of Words (BoW) approach for feature quantization
  - TF-IDF weighting for improved retrieval performance
- **Comprehensive Experimentation**:
  - Analysis of training set size impact
  - Evaluation of vocabulary size (number of centroids)
  - Comparison between BoW and TF-IDF representations

## 🚀 Getting Started <a name="getting-started"></a>

### Prerequisites <a name="prerequisites"></a>

- Python 3.8 or higher
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- Jupyter (for notebook execution)

### Installation <a name="installation"></a>

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sift-texture-retrieval.git
   cd sift-texture-retrieval
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage <a name="usage"></a>

### Jupyter Notebook <a name="jupyter-notebook"></a>

The `CV_SIFT_Assignment.ipynb` notebook contains the complete implementation with detailed explanations:

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `CV_SIFT_Assignment.ipynb` and run the cells sequentially.

The notebook workflow:
1. Loads and preprocesses texture images
2. Extracts SIFT features
3. Builds visual vocabularies with different parameters
4. Creates image representations using BoW and TF-IDF
5. Retrieves similar images for given queries
6. Analyzes and visualizes the results

### Command Line Interface <a name="command-line-interface"></a>

For a simpler usage, use the standalone Python script:

```bash
python sift_retrieval.py --images_path ./images --sample_size 200 --training_size 100 --centroids 200 --tfidf --num_queries 3
```

**Arguments:**
- `--images_path`: Path to the directory containing images
- `--sample_size`: Number of images to use (default: 200)
- `--training_size`: Number of images for training (default: 100)
- `--centroids`: Number of visual words/centroids (default: 200)
- `--tfidf`: Use TF-IDF representation instead of BoW (optional flag)
- `--num_queries`: Number of random query images (default: 3)
- `--num_results`: Number of results to show per query (default: 5)

## 📊 Dataset <a name="dataset"></a>

The project uses a comprehensive texture dataset organized in different categories:
- banded
- blotchy
- braided
- bubbly
- bumpy
- chequered
- cobwebbed
- cracked
- and many more texture types

Each category represents a specific texture pattern commonly found in natural and man-made materials.

## 📈 Experimental Results <a name="experimental-results"></a>

Our experiments demonstrate:
- **Optimal Training Size**: More training images generally improve performance up to a certain point
- **Optimal Vocabulary Size**: Larger vocabularies (200-500 centroids) typically provide better discriminative power
- **Representation Comparison**: TF-IDF typically outperforms standard BoW by giving more weight to distinctive visual words

## 📁 Project Structure <a name="project-structure"></a>

```
├── CV_SIFT_Assignment.ipynb     # Main Jupyter notebook with implementation
├── sift_retrieval.py            # Standalone Python script for SIFT retrieval
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
├── README.md                    # Project documentation
├── images/                      # Dataset of texture images
│   ├── banded/
│   ├── blotchy/
│   ├── braided/
│   └── ...                      # Other texture categories
```

## 📜 License <a name="license"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Ahmed Mossad** - [ahmed.abdelfattah.mossad@gmail.com](mailto:ahmed.abdelfattah.mossad@gmail.com)

Project Link: [https://github.com/ahmedm0ssad/SIFT-Texture-Image-Retrieval](https://github.com/ahmedm0ssad/SIFT-Texture-Image-Retrieval)