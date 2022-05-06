# CUDA-based Bag-of-words Scene Recognition

Team members: Hanyu Chen (hanyuche)

## Final Report

[Report](https://github.com/leniumC/BowSceneRecognition/blob/master/15_418_report.pdf) & [presentation video](https://drive.google.com/file/d/1wrNvVvT71HLRdieH3rnzjiK50D9u7F25/view?usp=sharing)


## Proposal

### Summary

The project is based on 16-385 _Computer Vision_ Assignment 5 ([link](http://16385.courses.cs.cmu.edu/spring2021/assignments)). I would like to parallelize different parts of the bag-of-words scene recognition algorithm (originally written in Python) primarily using CUDA on my RTX 2070 Super graphics card.

### Background

The current algorithm can be described in four parts:

- For a set of N images (belonging to C classes), a filter bank with K filters is sequentially applied to each image of shape H x W to produce a response map of shape H x W x K (assuming grayscale).
- M points are chosen at random from each image, and the corresponding M vectors of length K are taken from the response map as visual words. In total we end up with NM visual words with length K. Then, k-means is used to cluster the K-dimensional visual words into k clusters.
- Each pixel in each image is assigned to the nearest cluster based on the distance between its corresponding visual word and the cluster center. For each image, a histogram is created based on the number of pixels that is assigned to each cluster, and the histogram is normalized to produce a discrete probability distribution, which is used as the feature of the image.
- A common classification algorithm is trained on the image features (i.e. the probability distributions) along with the C class labels to produce a classifier capable of recognizing different scenes.


### The Challenge

In the algorithms, different parts can benefit from parallelization differently, and there are often multiple axes of parallelization that are possible. For example, in convolving multiple images with multiple filters, one might choose to parallelize between image pixels, filter pixels, across images, or across filters. Testing is needed to determine which one works the best. Another example would be during k-means, each pixel needs to compute its distance to each cluster center and find the one with the minimum distance. One might choose to parallelize between pixels, cluster centers, or along the K-dimensional vectors. Synchronization and atomic operations might also be useful when accumulating sums and determining the minimum values.


### Resources

I will mainly be structuring my code based on the current Python code I have from 16-385 _Computer Vision_ Assignment 5 ([link](http://16385.courses.cs.cmu.edu/spring2021/assignments)) and first rewriting it in C++ to obtain a baseline sequential algorithm. I will then try to parallelize different parts of the algorithm from there.


### Goals & Deliverables

Currently the algorithm is written fully in Python, and although the algorithm utilizes libraries like NumPy and SciPy that has some parallelization capabilities, a major portion of the pipeline is still sequential. 

I aim to parallelize different parts of the algorithm: 
- convolving images with filters in parallel
- clustering visual words using k-means in parallel
- assigning cluster centers for each pixel and creating histogram features in parallel
- implementing parallelized kNN classifier and/or logistic regression with parallelized matrix multiplication (if time permits)

I would consider completing up to the kNN classifier what I plan to achieve and consider implementing other classification algorithms like logistic regression and support vector machines to be extra goals that I hope to achieve.

### Platform Choice

I have chosen to implement this algorithm using CUDA because there is little communication between threads in this algorithm, and the problem can be divided into many smaller subproblems than can mostly be solved independently. Therefore, being able to parallelize the algorithm on a large number of CUDA cores would lead to a significant speedup over the sequential version.


## Milestone

April 1: Completed basic utility functions such as loading and storing images. Implemented sequential baseline algorithm for applying filters to images.

April 4: Implemented three parallel versions of image filtering. The first algorithm parallelizes over both image pixels (each block responsible for an image pixel) and filter pixels (each thread responsible for a filter pixel). The performance is not great since multiple threads are simultaneously writing to global variables, requiring atomic operations and synchronization. The second algorithm parallelizes over only image pixels (each thread responsible for an image pixel) and convolves an image block with a filter sequentially. This can be thought as the most naive version of parallel image filtering, achieving a speedup of 24x at best. The third algorithm parallelizes over both image pixels and filters (each thread responsible for an image pixel and a filter). It also combines all images and all filters into single 1D arrays that are copied to the GPU memory only once at the start of the algorithm. This final version achieves a 56x speedup at best.

April 7: Implemented sequential baseline algorithm for k-means, including utility function for loading and storing filter responses and selecting random points.

April 10: Implemented parallel versions of k-means. A naive implementation would be to put everything inside of a single kernel. However, this is not desirable since this would require many synchronizations between threads and even across the entire grid in each iteration of the algorithm, which impacts performance. Instead, the algorithms parallelizes over data points for both steps (assigning clusters and computing cluster centers) with two separate kernel calls. In each iteration, the kernels are called sequentially, allowing it to alternate between two steps until convergence (i.e. cluster assignments do not change). This version achieves 40x speedup at best.

For the final poster session, I am thinking about showing speedup graphs for different stages of the algorithm and compare the performance when parallelizing over different axes. I will draw graphs to visually demonstrate how the parallel algorithms are implemented. To help better understand the task, I will also show intermediate results of the algorithm, such as filter responses, cluster centers, and the clusters that each pixel belongs to for several images. Although parallelization should not affect the final classification results, I will briefly mention the classification accuracies and potential algorithms that can be implemented (and parallelized) to improve the performance.
