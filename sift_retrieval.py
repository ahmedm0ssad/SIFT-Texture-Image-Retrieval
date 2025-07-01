#!/usr/bin/env python3
"""
SIFT-based Image Retrieval System

This script implements an image retrieval system using SIFT features
and a bag of visual words approach.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
import os
import random
from glob import glob
import argparse

def extract_sift_features(images):
    """Extract SIFT features from a list of images"""
    sift = cv.SIFT_create()
    all_descriptors = []
    all_features = []
    
    print("Extracting SIFT features...")
    for img in tqdm(images):
        if img is None:
            all_features.append(None)
            continue
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is not None and len(keypoints) > 0:
            all_descriptors.append(descriptors)
            all_features.append(descriptors)
        else:
            all_features.append(None)
    
    # Stack all descriptors
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
    else:
        all_descriptors = np.array([])
    
    return all_features, all_descriptors

def build_vocabulary(descriptors, k):
    """Build visual vocabulary by clustering the descriptors"""
    print(f"Building vocabulary with {k} visual words...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(descriptors)
    return kmeans

def create_bow_histograms(features, kmeans):
    """Create Bag of Words histograms for each image"""
    vocab_size = kmeans.cluster_centers_.shape[0]
    histograms = []
    
    print("Creating BoW histograms...")
    for descriptors in tqdm(features):
        hist = np.zeros(vocab_size)
        
        if descriptors is not None:
            # Assign each descriptor to a cluster
            predictions = kmeans.predict(descriptors)
            
            # Create histogram of visual words
            for pred in predictions:
                hist[pred] += 1
                
            # Normalize histogram
            if np.sum(hist) > 0:
                hist = hist / np.sum(hist)
        
        histograms.append(hist)
    
    return np.array(histograms)

def create_tfidf_representation(bow_histograms):
    """Convert BoW histograms to TF-IDF representation"""
    print("Creating TF-IDF representation...")
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(bow_histograms).toarray()
    return tfidf

def retrieve_images(query_idx, feature_matrix, n=5):
    """Retrieve the top N most similar images"""
    query_vector = feature_matrix[query_idx:query_idx+1]
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()
    
    # Exclude the query image itself
    similarities[query_idx] = 0
    
    # Get indices of top N similar images
    top_indices = np.argsort(similarities)[::-1][:n]
    top_similarities = similarities[top_indices]
    
    return top_indices, top_similarities

def display_retrieval_results(query_idx, retrieved_indices, similarities, images, image_paths=None, title="Image Retrieval Results"):
    """Display query image and retrieved images with their similarity scores"""
    n = len(retrieved_indices)
    plt.figure(figsize=(15, 4))
    
    # Display query image
    plt.subplot(1, n+1, 1)
    plt.imshow(cv.cvtColor(images[query_idx], cv.COLOR_BGR2RGB))
    if image_paths:
        plt.title(f"Query: {os.path.basename(image_paths[query_idx])}")
    else:
        plt.title("Query Image")
    plt.axis('off')
    
    # Display retrieved images
    for i, (idx, sim) in enumerate(zip(retrieved_indices, similarities)):
        plt.subplot(1, n+1, i+2)
        plt.imshow(cv.cvtColor(images[idx], cv.COLOR_BGR2RGB))
        if image_paths:
            plt.title(f"{os.path.basename(image_paths[idx])}\nSim: {sim:.4f}")
        else:
            plt.title(f"Sim: {sim:.4f}")
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="SIFT-based Image Retrieval System")
    parser.add_argument("--images_path", type=str, required=True, help="Path to the directory containing images")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of images to use")
    parser.add_argument("--training_size", type=int, default=100, help="Number of images for training")
    parser.add_argument("--centroids", type=int, default=200, help="Number of visual words (centroids)")
    parser.add_argument("--tfidf", action="store_true", help="Use TF-IDF representation")
    parser.add_argument("--num_queries", type=int, default=3, help="Number of random query images")
    parser.add_argument("--num_results", type=int, default=5, help="Number of results to show per query")
    
    args = parser.parse_args()
    
    # Load images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob(os.path.join(args.images_path, '**', ext), recursive=True))
    
    if len(image_paths) == 0:
        print(f"No images found in {args.images_path}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Use a subset of images for efficiency
    if args.sample_size < len(image_paths):
        image_paths = image_paths[:args.sample_size]
    
    # Load images
    images = []
    for path in tqdm(image_paths, desc="Loading images"):
        img = cv.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not read image {path}")
    
    # Select random query indices
    query_indices = random.sample(range(len(images)), args.num_queries)
    
    # Select random subset for training
    training_indices = np.random.choice(len(images), args.training_size, replace=False)
    training_images = [images[i] for i in training_indices]
    
    # Extract features from training images
    _, all_descriptors = extract_sift_features(training_images)
    
    # Build vocabulary from training features
    kmeans = build_vocabulary(all_descriptors, args.centroids)
    
    # Extract features from all images
    all_features, _ = extract_sift_features(images)
    
    # Create BoW histograms
    bow_histograms = create_bow_histograms(all_features, kmeans)
    
    # Create TF-IDF representation if requested
    if args.tfidf:
        representation = create_tfidf_representation(bow_histograms)
        method_name = "TF-IDF"
    else:
        representation = bow_histograms
        method_name = "Bag of Words"
    
    # For each query image
    for query_idx in query_indices:
        # Retrieve similar images
        retrieved_indices, similarities = retrieve_images(query_idx, representation, n=args.num_results)
        
        # Display results
        display_retrieval_results(
            query_idx, retrieved_indices, similarities, 
            images, image_paths, f"Query Results - {method_name}"
        )

if __name__ == "__main__":
    main()
