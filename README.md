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
   git clone https://github.com/ahmedm0ssad/SIFT-Texture-Image-Retrieval.git
   cd SIFT-Texture-Image-Retrieval
   ```

2. **⚠️ IMPORTANT**: Obtain and prepare the texture images dataset:
   - Download the `images.zip` file from the project owner (not included in repo due to size limits)
   - Extract the zip file into the project root directory
   ```bash
   # On Windows
   # Use the built-in File Explorer to extract images.zip
   
   # On macOS/Linux
   unzip images.zip
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage <a name="usage"></a>

> **⚠️ IMPORTANT**: Make sure you have unzipped the `images.zip` file before proceeding. The code expects the texture images to be available in the `images/` directory with category subfolders. And the link drive of the dataset -> https://drive.google.com/file/d/15_4SWSvH94t1uuha_BWWqNa67a0pdoe8/view?usp=sharing    

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

### Obtaining the Dataset

Due to GitHub's file size limitations (100MB max), the full image dataset is not included in this repository. You can obtain the texture images in one of these ways:

1. **Download from project owner**: Contact the project owner to get access to the `images.zip` file (approximately 101MB).

2. **Use your own texture dataset**: You can use any texture dataset with a similar structure, organizing images in category subfolders.

3. **Create sample data**: For testing purposes, you can create a small subset of texture images organized in category folders.

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
