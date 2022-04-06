# CUDA-based Bag-of-words Scene Recognition

Team members: Hanyu Chen (hanyuche)

## Summary

The project is based on 16-385 _Computer Vision_ Assignment 5 ([link](http://16385.courses.cs.cmu.edu/spring2021/assignments)). I would like to parallelize different parts of the bag-of-words scene recognition algorithm (originally written in Python) primarily using CUDA on my RTX 2070 Super graphics card.

## Background

The current algorithm can be described in four parts:

- For a set of N images (belonging to C classes), a filter bank with K filters is sequentially applied to each image of shape H x W to produce a response map of shape H x W x K (assuming grayscale).
- M points are chosen at random from each image, and the corresponding M vectors of length K are taken from the response map as visual words. In total we end up with NM visual words with length K. Then, k-means is used to cluster the K-dimensional visual words into k clusters.
- Each pixel in each image is assigned to the nearest cluster based on the distance between its corresponding visual word and the cluster center. For each image, a histogram is created based on the number of pixels that is assigned to each cluster, and the histogram is normalized to produce a discrete probability distribution, which is used as the feature of the image.
- A common classification algorithm is trained on the image features (i.e. the probability distributions) along with the C class labels to produce a classifier capable of recognizing different scenes.


## The Challenge

In the algorithms, different parts can benefit from parallelization differently, and there are often multiple axes of parallelization that are possible. For example, in convolving multiple images with multiple filters, one might choose to parallelize between image pixels, filter pixels, across images, or across filters. Testing is needed to determine which one works the best. Another example would be during k-means, each pixel needs to compute its distance to each cluster center and find the one with the minimum distance. One might choose to parallelize between pixels, cluster centers, or along the K-dimensional vectors. Synchronization and atomic operations might also be useful when accumulating sums and determining the minimum values.


## Resources

I will mainly be structuring my code based on the current Python code I have from 16-385 _Computer Vision_ Assignment 5 ([link](http://16385.courses.cs.cmu.edu/spring2021/assignments)) and first rewriting it in C++ to obtain a baseline sequential algorithm. I will then try to parallelize different parts of the algorithm from there.


## Goals & Deliverables

Currently the algorithm is written fully in Python, and although the algorithm utilizes libraries like NumPy and SciPy that has some parallelization capabilities, a major portion of the pipeline is still sequential. 

I aim to parallelize different parts of the algorithm: 
- convolving images with filters in parallel
- clustering visual words using k-means in parallel
- assigning cluster centers for each pixel and creating histogram features in parallel
- implementing parallelized kNN classifier and/or logistic regression with parallelized matrix multiplication (if time permits)

I would consider completing up to the kNN classifier what I plan to achieve and consider implementing other classification algorithms like logistic regression and support vector machines to be extra goals that I hope to achieve.
