# DL-Foundation-from-Scratch
Stanford Lecture - CS231n theory implementation using NumPy for deep learning fundamentals

## 1. k-Nearest Neighbor (k-NN) Implementation
Implemented a complete k-NN classifier from scratch using NumPy.

* **Vectorization:** optimized performance by replacing nested loops with matrix operations (Broadcasting), achieving significant speedup.
* **Hyperparameter Tuning:** Conducted K-fold cross-validation to find the optimal $k$.
* **Modules:**
    * `k_nn_utils.py`: Core logic for distance calculation (L2) and prediction.
    * `knn_cifar10.py`: Script for training, testing, and visualization.

---

## 🔬 Experiment & Analysis: Limitations of k-NN
### Hypothesis & Experiment
Investigated the relationship between dataset size and model performance to verify the effectiveness of Pixel-wise L2 distance in high-dimensional space (CIFAR-10).
* **Setup:** Compared accuracy between small dataset ($N=5,000$) and full dataset ($N=50,000$).
* **Result:** Accuracy saturated around **33%** despite a 10x increase in data.

### Key Insights
**1. The Curse of Dimensionality**
Even with 50,000 samples, the data remains sparse in the 3,072-dimensional space. The distance to the nearest neighbor does not decrease significantly, leading to diminishing returns in performance (Logarithmic growth).

**2. Semantic Gap in L2 Distance**
* **Observation:** The model often misclassifies images based on dominant background colors rather than object shapes.
* **Analysis:** L2 distance calculates the sum of independent pixel differences. It is sensitive to **global color distributions** (e.g., green background) but fails to capture **local semantic features** (e.g., edges, shapes).
* **Conclusion:** Pure data scaling cannot overcome the structural limitations of pixel-based distance metrics. This necessitates the use of feature-extraction-based models like Linear Classifiers or CNNs.
![Graph](images/k-NN/knn_sample_is_5000.png)
![Graph](images/k-NN/knn_sample_is_50000.png)
![Graph](images/k-NN/ex_log.png)

## 2. Linear Classifier (SVM) Implementation
Implemented a Multiclass SVM (Hinge Loss) classifier to overcome the memory and prediction speed limitations of k-NN.

* **Parametric Approach:** Transitioned from memory-based (k-NN) to model-based learning ($f(x, W) = Wx + b$), compressing the knowledge of the entire dataset into a weight matrix $W$.
* **Fully Vectorized Loss:** Implemented the SVM loss function without explicit loops, utilizing NumPy broadcasting and advanced indexing for massive performance gains.
* **Modules:**
    * `linear_classifier.py`: Implements forward pass (score calculation) and vectorized loss computation.

---

## 🔬 Experiment & Analysis: Initial Loss Verification
### Context & Observation
Verified the correctness of the vectorized implementation by analyzing the initial loss value with unoptimized random weights.
* **Setup:** Initialized $W$ using standard normal distribution (`np.random.randn`) scaled by $0.01$. Input images were scaled to $[0, 1]$.
* **Result:** Calculated Initial Loss $\approx 338.4$ (on $N=50,000$).

### Key Insights
**1. Validation of Vectorization**
The calculation for 50,000 images was completed almost instantly. The resulting loss value (~338) aligns with the expected theoretical range for unnormalized random weights, confirming that the broadcasting and masking logic works correctly across the entire batch.

**2. The Need for Optimization**
Unlike k-NN, where performance is fixed by the dataset, this high loss value serves as the baseline for learning. The quantitative loss metric proves that the current random model is failing to classify correctly, setting the stage for implementing **Gradient Descent** to minimize this loss.