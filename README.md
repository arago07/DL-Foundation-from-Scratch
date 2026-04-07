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

## 3. Optimization & Training
Implemented the core training logic to minimize the SVM Loss using Gradient Descent.

* **Analytic Gradient:** Derived and implemented the gradient of the SVM loss function ($\nabla_W L$) using fully vectorized NumPy operations, avoiding inefficient numerical differentiation.
* **Stochastic Gradient Descent (SGD):** Transformed the training loop from Batch Gradient Descent (using all 50k images) to SGD (Mini-batch size: 200), achieving massive speed improvements.
* **Hyperparameter Tuning:** Experimented with Learning Rate and Batch Size to stabilize training.

---

## 🔬 Experiment Log: Overcoming Challenges
### 1. The "Exploding Gradient" Incident
* **Observation:** During the first training attempt, the Loss skyrocketed from **321** to **71,047** within 10 iterations (Divergence).
* **Root Cause Analysis:**
    * The input data was unscaled ($0 \sim 255$), resulting in large score values.
    * The large scores caused massive gradients, and combined with the learning rate, the weights updated too aggressively ("overshooting" the minima).
* **Solution:**
    * **Data Preprocessing:** Applied Normalization (`X_train /= 255.0`) to scale pixel values between $[0, 1]$.
    * **Type Casting:** Converted data to `float32` before division to prevent type mismatch errors.

### 2. Successful Convergence & SGD Analysis
* **Result after Fix:**
    * **Initial Loss:** Dropped to **10.6** (Close to the theoretical expected loss for random weights: $\approx 9.0$).
    * **Training Dynamics:** Loss decreased steadily without divergence.
* **SGD Efficiency:**
    * Switching to SGD (Batch size 200) accelerated the training loop by approx. **250x** compared to Batch GD.
    * **Final Loss:** Reached **~7.9** after 1,500 iterations.
    * **Fluctuation:** Observed the characteristic "noisy" descent of SGD (e.g., Loss jumping 8.7 $\to$ 9.3 $\to$ 7.9), confirming the stochastic nature of sampling.

## 4. Final Evaluation & Visualization
Successfully evaluated the best SVM model on the test set and visualized the learned templates.

* **Hyperparameter Tuning:** Searched through multiple `learning_rates` and `reg_strengths`.
    * **Best Combination:** `lr: 0.001`, `reg: 0.25`
    * **Best Validation Accuracy:** 36.20%
* **Final Test Performance:** Achieved **33.66%** Accuracy on the CIFAR-10 test set.
* **Weight Visualization:** Observed that the model learns "spatial templates" for each class (e.g., green blobs for frogs, blue backgrounds for ships).
* The following images represent the learned weights (templates) for each class:

![SVM Weights](images/linear_classifier/svm_weights.png)

## 5. Neural Network Visualization
Unlike the linear classifier, the two-layer neural network learns distributed representations. The following image shows the weights ($W_1$) learned by the 50 hidden neurons:

![Hidden Layer Weights](images/two_layer_net/w1_visualization.png)

* **Observation:** The neurons act as various filters for edges, colors, and blobs, which are then combined in the second layer to classify the image.

## 6. Convolutional Neural Network (CNN) Implementation
Implemented a modular Convolutional Neural Network (CNN) from scratch to capture spatial hierarchies in image data, moving beyond the limitations of flat vector inputs used in Linear Classifiers and MLPs.

* **Full Modular Architecture:** Implemented `Conv - ReLU - Pool - Affine - ReLU - Affine - Softmax` architecture.
* **Manual Backpropagation:** Derived and implemented the analytic gradients for Convolution and Max Pooling layers using the Chain Rule, handling 4D tensors ($N, C, H, W$) without automatic differentiation.
* **Modules:**
    * `layers.py`: Contains `forward` and `backward` methods for `Conv_naive`, `MaxPool_naive`, etc.
    * `cnn.py`: Assembles the layers into a `ThreeLayerConvNet` class.
    * `train_overfit.py`: Script for verifying implementation integrity.

---

## 🔬 Experiment & Analysis: Implementation Verification
### Context & Observation
Before training on the full dataset, it is crucial to verify the correctness of the complex backpropagation logic (specifically dimensions and gradient flow in 4D tensors).
* **Setup:** Trained the model on a tiny dataset ($N=5$ images) with a high learning rate ($0.1$) for 20 epochs.
* **Hypothesis:** If the forward and backward passes are implemented correctly, the model should have enough capacity to perfectly memorize (overfit) the small dataset, driving the loss to near zero.
* **Result:**
    * **Initial Loss:** $\approx 2.3$ (Random guessing).
    * **Final Loss:** $\approx 0.02$ (Perfect memorization).

### Key Insights
**1. Validation of Gradient Flow (Chain Rule)**
The convergence to near-zero loss confirms that the gradient of the loss function is correctly flowing back through the Max Pooling (routing gradients to max indices) and Convolution layers (cross-correlating gradients with filters). If there were any dimension mismatch or mathematical error in `conv_backward`, the loss would have stagnated or exploded.

**2. SGD Dynamics: The "Overshooting" Phenomenon**
* **Observation:** During the training (Epoch 10-12), the loss temporarily spiked ($0.86 \to 1.93$) before settling down.
* **Analysis:** This illustrates the behavior of Stochastic Gradient Descent with a high learning rate. The optimizer "overshot" the local minimum due to the large step size but successfully corrected its trajectory. This confirms that the update logic ($W \leftarrow W - \eta \cdot \nabla W$) is working robustly even under aggressive hyperparameter settings.

## 🔬 Experiment & Analysis: Overfitting Test (Sanity Check)

To verify the correctness of the implementation (especially Backpropagation), I conducted a "Sanity Check" by overfitting a small dataset ($N=5$).

### 📊 Experiment 1: Conservative Learning Rate
* **Setup:** `learning_rate = 0.01`, `epochs = 20`
* **Observation:** The loss decreased very slowly ($2.29 \to 2.03$).
* **Analysis:** The gradient updates were too small to converge within 20 epochs. This indicated the need for a more aggressive learning rate for this tiny dataset.

### 📊 Experiment 2: Aggressive Learning Rate (Success)
* **Setup:** `learning_rate = 0.1`, `epochs = 20`
* **Observation:**
    * **Overshooting:** A spike in loss occurred at Epoch 11 ($1.16$) and 12 ($2.00$), indicating the step size was large enough to jump over the local minima temporarily.
    * **Convergence:** The optimizer successfully corrected the trajectory, driving the final loss to **0.0225**.
* **Conclusion:** The model has sufficient capacity to memorize the dataset, confirming that the `forward` and `backward` passes are mathematically correct.

![Sanity Check Graph](images/cnn/overfit_result.png)

## 7. CNN Training & Feature Visualization
Scaled up the experiment from the sanity check to learning actual visual features from the CIFAR-10 dataset using the implemented CNN architecture.

* **Data Pipeline:** Implemented efficient data loading for a subset ($N=5,000$) to optimize for CPU-based training.
    * **Preprocessing:** Applied Mean Subtraction and dimension transposition (`HWC` $\to$ `CHW`) to align with the custom `im2col` implementation.
* **Training Loop:** Implemented the full SGD loop (`cnn_cifar10.py`) with real-time loss logging and visualization logic.

---

## 🔬 Experiment & Analysis: Visualizing Learned Filters
### Context & Setup
The goal was to verify if the implementation could learn meaningful visual representations (spatially organized features) rather than just memorizing pixel values.
* **Setup:** Trained on 5,000 CIFAR-10 images using `batch_size=50`.
* **Tuning:** Initially, with `lr=0.001`, the loss stagnated at **2.302** (Random Guessing). Increasing the learning rate to `0.01` triggered immediate convergence.

### Key Insights
**1. Emergence of Visual Patterns (Gabor Filters)**
Upon visualization, the first-layer weights ($W_1$) transformed from random noise into distinct, smooth patterns.
* **Color Blobs:** Specific filters (e.g., #17 Green, #16 Pink) learned to activate on dominant background colors.
* **Edge Detectors:** Other filters (e.g., #5, #14) evolved into "Edge Detectors" capable of recognizing horizontal or diagonal lines.
* **Conclusion:** This visually proves that the **Convolution operation** and **Backpropagation** are correctly extracting low-level features (edges, colors) from raw pixels, which is the fundamental basis of Deep Learning vision models.

**2. Loss Dynamics & SGD Fluctuation**
* **Observation:** The loss dropped significantly to **~1.57** but showed high fluctuation ($1.6 \leftrightarrow 2.0$) in later iterations.
* **Analysis:** This behavior is characteristic of Stochastic Gradient Descent with a small batch size ($50$) and an aggressive learning rate. The model successfully escaped the initial plateau and converged to a meaningful state, proving the robustness of the update rule even with limited data.

![CNN Filters](images/cnn/filter_visualization.png)

---

# Part 2: PyTorch Practice (Deep Learning with Frameworks)
This section documents the transition from manual NumPy implementations to efficient deep learning workflows using **PyTorch**, focusing on hardware acceleration and automatic differentiation for Vision AI research.

## 8. Tensor Fundamentals & Hardware Acceleration (Day 1)
Explored the core data structure of PyTorch and configured the environment for high-performance computing on Apple Silicon.

* **Tensor Manipulation:** Mastered tensor creation, indexing, and broadcasting, which are the building blocks for handling high-dimensional image data.
* **Device Management:** Implemented logic to detect and utilize **MPS (Metal Performance Shaders)**, enabling GPU acceleration on Mac devices.
* **Interoperability:** Leveraged the bridge between NumPy and PyTorch to maintain flexibility in data preprocessing.

---

## 9. Autograd: The Engine of Deep Learning (Day 2)
Deep-dived into the mechanics of **Automatic Differentiation**, the core technology that replaces manual gradient derivations in Stanford CS231n theory.

* **Computational Graphs:** Understood how PyTorch dynamically builds graphs to track operations on tensors with `requires_grad=True`.
* **The Backward Mechanism:** Practiced triggering the chain rule via `.backward()` to automatically compute gradients for complex functions.
* **Gradient Accumulation:** Identified that PyTorch accumulates gradients by default, necessitating the use of `optimizer.zero_grad()` to prevent interference between training iterations.

---

## 10. Linear Regression & Standard Training Loop (Day 3)
Integrated the fundamentals into a complete end-to-end training pipeline for a simple regression task ($y = 2x + 1$).

* **Standardized Workflow:** Established the 5-step routine: `Forward -> Loss -> Zero_grad -> Backward -> Step`.
* **Directory:** `pytorch_practice/day3_linear_regression.py`

---

## 🔬 Experiment & Analysis: Learning Rate Sensitivity
### Context & Setup
Investigated the impact of the **Learning Rate ($\eta$)** on the convergence of a linear model with Gaussian noise. This experiment highlights the critical role of optimizer hyperparameters in model stability.
* **Epochs:** 200
* **Optimizer:** SGD
* **Loss Function:** MSELoss

### Key Insights

**1. Gradient Explosion & Divergence (The "NaN" Problem)**
* **Setup:** `lr = 0.8`
* **Observation:** Loss skyrocketed to $10^{33}$ within 10 epochs and eventually hit `inf` and `nan`.
* **Analysis:** The step size was too aggressive, causing the optimizer to overshoot the global minimum and diverge. This proves that without proper LR scaling, even a convex problem can fail to initialize.

**2. Slow Convergence (The "Turtle" Problem)**
* **Setup:** `lr = 0.0001`
* **Observation:** The loss decreased at an imperceptible rate. After 200 epochs, the weight reached only ~1.4 (Target: 2.0).
* **Analysis:** While stable, the updates were too small to reach the optimal solution.

**3. Optimal Convergence**
* **Setup:** `lr = 0.01`
* **Observation:** The model successfully converged to $w \approx 1.96, b \approx 1.03$, effectively filtering out the synthetic noise to find the underlying trend.

### 📊 Comparative Visualization
| **Explosion (lr=0.8)** | **Slow (lr=0.0001)** | **Optimal (lr=0.01)** |
| :---: | :---: | :---: |
| ![High LR](images/pytorch_practice/linear_regression_result_high.png) | ![Low LR](images/pytorch_practice/linear_regression_result_low.png) | ![Optimal LR](images/pytorch_practice/linear_regression_result_moderate.png) |
*(Note: High LR results in no fitted line due to NaN coordinates in the weight/bias tensors.)*

---

## 11. Multi-variable Linear Regression (Day 4)
Expanded the linear model to handle multiple input features, transitioning from scalar-based logic to **Vectorized Matrix Operations** using PyTorch.

* **High-Dimensional Mapping:** Implemented a model to predict a single target value from three independent features ($$x_1, x_2, x_3$$), following the hypothesis: $$y = w_1x_1 + w_2x_2 + w_3x_3 + b$$.
* **Vectorization with `matmul`:** Replaced manual summation with `torch.matmul(X, W)` to process 100 samples simultaneously, ensuring high computational efficiency on Apple Silicon (MPS).
* **Internal Weight Logic:** Analyzed `nn.Linear`'s storage format ($$out \times in$$) and verified that PyTorch internally transposes weights ($$W^T$$) to execute $$y = XW^T + b$$.
* **Modules:**
    * `pytorch_practice/day4_multivariable.py`: Full implementation of multi-variable training loop and weight analysis.

---

## 🔬 Experiment & Analysis: High-Dimensional Parameter Recovery
### Context & Setup
The objective was to verify if the optimizer could accurately recover multiple hidden weights ($$W = [2, 3, 4], b = 5$$) from a dataset contaminated with Gaussian noise ($$std=0.5$$).
* **Hyperparameters:** `epochs = 2000`, `learning_rate = 0.01`, `optimizer = SGD`.
* **Convergence:** Successfully reached a stable state around **Epoch 400**.

### Key Insights

**1. The "Noise Floor" Phenomenon**
* **Observation:** Despite 2,000 epochs of training, the loss reached a minimum plateau of **~0.2300** and never hit zero.
* **Analysis:** This represents the **theoretical limit of error** (Noise Floor). Since we added noise with $$\sigma = 0.5$$, the minimum achievable MSE is approximately $$\sigma^2 = 0.25$$. The model effectively learned 100% of the underlying signal, with the remaining loss being irreducible noise.

**2. Statistical Accuracy in Weights**
* **Result:** Final learned weights were $$W \approx [1.997, 3.103, 4.007], b \approx 5.092$$.
* **Analysis:** The slight deviation from the exact integers ($$2, 3, 4$$) is expected due to the stochastic nature of the noise and limited sample size ($$N=100$$). This proves that the model found the **statistically optimal solution** rather than simply memorizing the noisy data points.

**3. Vectorized Data Extraction**
* **Technique:** Refined the logic for extracting trained parameters based on their dimensionality.
    * **Scalar Bias ($$b$$):** Used `.item()` for direct conversion to float.
    * **Weight Vector ($$W$$):** Used `.detach().tolist()` to safely extract the multi-dimensional array while disconnecting it from the Autograd graph to prevent memory overhead.

---

## 12. Dataset & DataLoader: Scalable Data Pipelines (Day 5)
Transitioned from processing entire datasets in memory to a scalable data pipeline using PyTorch's standardized components.

* **Modular Design:** Subclassed `torch.utils.data.Dataset` to encapsulate data storage (`__init__`), size reporting (`__len__`), and index-based retrieval (`__getitem__`).
* **Mini-batch Management:** Integrated `DataLoader` to partition 100 samples into batches of size 8, implementing `shuffle=True` to prevent the model from memorizing data order.
* **Nested Training Loop:** Designed a two-tier loop structure (Epoch and Batch) to enable memory-efficient parameter updates.
* **Modules:**
    * `pytorch_practice/day5_dataloader.py`: Implementation of the custom Dataset class and mini-batch training loop.

---

## 🔬 Experiment & Analysis: Mini-batch SGD Convergence
### Context & Observation
Analyzed the convergence behavior of Mini-batch Stochastic Gradient Descent (SGD) on a noisy multi-variable dataset.
* **Setup:** `batch_size = 8`, `learning_rate = 0.01`, `epochs = 100`.
* **Result:** Successfully reduced the Average Loss from **~55.0** to a stable **~0.0102**.

### Key Insights

**1. The "Invisible Gradient" Incident (Parentheses Syntax)**
* **Observation:** The loss initially stagnated at ~55 despite the training loop running.
* **Root Cause Analysis:** `loss.backward` was called without parentheses `()`, meaning the function was referenced but not executed. This prevented gradient computation, leaving weights ($W, b$) static despite calling `optimizer.step()`.
* **Solution:** Explicitly invoked the function using `loss.backward()` to trigger the autograd engine.

**2. Theoretical Loss Floor Verification**
* **Analysis:** Since the synthetic data included Gaussian noise with $\sigma = 0.1$, the theoretical minimum MSE is $\sigma^2 = 0.01$.
* **Conclusion:** The final loss of **0.0102** indicates that the model has perfectly recovered the underlying signal ($W=[2, 3, 4], b=5$), with only the irreducible noise remaining.

**3. Format Specifiers for Log Readability**
* **Implementation:** Utilized the `:3d` format specifier in f-strings to ensure consistent vertical alignment of epoch logs regardless of the number of digits.

---

## 13. Non-linearity & Multi-Layer Perceptron (Day 6)
Overcame the mathematical limitations of linear models by implementing a Multi-Layer Perceptron (MLP) and introducing non-linear activation functions to solve complex decision boundaries.

* **nn.Module Subclassing:** Adopted the standard PyTorch design pattern by subclassing `nn.Module`, ensuring proper parameter registration via `super().__init__()`.
* **Breaking Linearity:** Introduced `nn.ReLU` and `nn.Sigmoid` between linear layers to prevent "Linear Stacking," where multiple linear layers mathematically collapse into a single transformation.
* **Refactoring with `nn.Sequential`:** Compared manual layer linking in `forward()` with the more concise `nn.Sequential` container, optimizing code readability for feed-forward architectures.
* **Modules:**
    * `pytorch_practice/day6_mlp_xor.py`: Implementation of MLP architectures (V1: Manual, V2: Sequential) for the XOR problem.

---

## 🔬 Experiment & Analysis: Solving the XOR Problem
### Context & Observation
The XOR problem is the classic benchmark for non-linearity, as its classes cannot be separated by a single linear hyperplane. The goal was to verify if a 2-layer MLP could distort the input space to achieve perfect classification.
* **Setup:** `input: 2`, `hidden: 10`, `output: 1`. Used `BCELoss` for binary classification and `lr: 1.0` to handle the small dataset.
* **Result:** Successfully reached **100.0% Accuracy**. Average Loss dropped from **0.6924** (random guessing) to **0.000038** after 10,000 steps.

### Key Insights

**1. The Chain of Linearity & Activation Functions**
* **Theory:** Without non-linear activations, $y = W_2(W_1x + b_1) + b_2$ simplifies to $y = W_{new}x + b_{new}$.
* **Experiment:** Verified that removing `nn.ReLU` causes the accuracy to plateau at **50%**, proving that "Depth" without "Non-linearity" is mathematically equivalent to a single-layer linear model.

**2. Optimization Dynamics: High Learning Rate ($lr=1.0$)**
* **Analysis:** For extremely small datasets like XOR ($N=4$), the gradient updates are infrequent (once per epoch in full batch).
* **Strategy:** A high learning rate was essential to escape the flat plateaus (saddle points) of the Sigmoid-based loss landscape. Unlike large-scale datasets where $lr=1.0$ might cause divergence, here it facilitated rapid convergence toward the global minimum.

**3. Binary Cross Entropy (BCE) vs. MSE**
* **Observation:** Transitioned from `MSELoss` to `BCELoss` for the classification task.
* **Mechanism:** Combined with a `Sigmoid` output, `BCELoss` penalizes wrong predictions exponentially as they approach the opposite class, providing a much stronger gradient signal for binary outcomes than squared error.

**4. Advanced Tensor Post-processing**
* **Technique:** Implemented the `.detach().numpy()` chain for result visualization.
* **Insight:** Mastered the requirement of disconnecting tensors from the autograd graph (`.detach()`) before converting them to NumPy for interoperability with standard Python data tools.

---

## 14. Softmax Classification & MNIST (Day 7)
Expanded beyond binary classification to implement a Multi-class Classification model for the MNIST dataset (0-9 digits).

* **Softmax & Cross Entropy:** Utilized the Softmax function to ensure the sum of output probabilities equals 1, and applied `nn.CrossEntropyLoss` to maximize the predicted probability of the correct class.
* **Efficient Pipeline:** Implemented `view(-1, 784)` to transform $28 \times 28$ images into 784-dimensional vectors and used the `drop_last=True` option in the DataLoader to ensure batch consistency.
* **Advanced Optimization:** Transitioned from basic SGD to the **Adam** optimizer, which combines Momentum and RMSProp (Adaptive Learning Rate) for superior performance.

---

## 🔬 Experiment & Analysis: Optimizer Benchmarking (SGD vs. Adam)

### 📊 Comparative Visualization
![MNIST Loss Comparison](images/pytorch_practice/mnist_loss_comparison.png)

### 📈 Quantitative Result
| Optimizer | Final Cost | Test Accuracy | Note |
| :--- | :---: | :---: | :--- |
| **SGD (lr=0.1)** | 0.0300 | 97.37% | Stable, but convergence is relatively slow. |
| **Adam (lr=0.001)** | **0.0117** | **97.92%** | **2.5x Cost reduction, 0.55%p Accuracy increase.** |

### Key Insights

**1. Numerical Stability of `nn.CrossEntropyLoss`**
* PyTorch's `nn.CrossEntropyLoss` internally combines `LogSoftmax` and `NLLLoss`.
* By passing "Raw Logits" to the loss function instead of manually applying Softmax at the final layer, the model avoids potential **Overflow** during exponential calculations, ensuring numerical stability.

**2. Superior Convergence of Adam**
* **Observation:** As shown in the graph above, Adam (blue) reached a lower loss plateau much faster than SGD (gray).
* **Analysis:** The **Adaptive Learning Rate** mechanism, which adjusts the step size for each parameter individually, allowed the model to react more sensitively to critical features in the 784-dimensional MNIST input, effectively navigating the complex loss landscape.

**3. Inference Mode & Memory Management**
* **Technique:** Utilized `with torch.no_grad():` during the testing phase to disable the generation of the computational graph.
* **Insight:** This minimized memory overhead and accelerated inference. The predicted labels were derived using `.argmax()`, which identifies the index with the highest probability among the 10 classes.

---

## 15. Convolutional Neural Networks (CNN) Deep Dive (Day 8)
Advanced from simple MLPs to **Convolutional Neural Networks (CNNs)** to effectively capture spatial hierarchies in image data. Conducted a rigorous benchmark study to optimize model architecture using modern deep learning techniques.

* **Architecture Evolution:**
    * **Basic CNN:** A standard 2-layer structure (`Conv - ReLU - MaxPool` $\times$ 2) to establish a baseline.
    * **Deep CNN:** A robust 3-layer architecture integrated with **Batch Normalization** and **Dropout** to improve stability and generalization.
* **Initialization Strategy:** Transitioned from Xavier Initialization (for Sigmoid/Tanh) to **He (Kaiming) Initialization**, which is mathematically optimal for ReLU activation functions.
* **Modules:**
    * `pytorch_practice/day8_cnn_comparison.py`: Full implementation of the comparative experiment, including visualization logic.

---

## 🔬 Experiment & Analysis: Architecture & Data Scaling
### Context & Setup
The objective was to break the 99% accuracy barrier on the MNIST dataset by overcoming the limitations of shallow networks.
* **Dataset:** MNIST (Normalized to $[0, 1]$)
* **Hyperparameters:** `epochs = 15`, `batch_size = 100`, `lr = 0.001` (Adam).
* **Comparison:**
    * **Model A (Basic):** Simple Feature Extractor + Classifier.
    * **Model B (Deep):** Added Depth (3rd Layer), **Batch Normalization** (to fix Internal Covariate Shift), and **Dropout** (p=0.5, for Regularization).

### 📊 Comparative Visualization
![CNN Comparison](images/pytorch_practice/cnn_comparison_result.png)

### 📈 Quantitative Result
| Model | Final Cost | Test Accuracy | Error Rate |
| :--- | :---: | :---: | :---: |
| **Basic CNN** | 0.0069 | 98.99% | 1.01% |
| **Deep CNN** | **0.0059** | **99.29%** | **0.71%** |

### Key Insights

**1. The "Silent Killer": Data Scaling Mismatch**
* **Incident:** Initially, the Deep CNN yielded a disastrous **64% accuracy** despite low training loss.
* **Root Cause Analysis:** The training data was normalized via `transforms.ToTensor()` ($0 \sim 1$), but the test data was raw pixel values ($0 \sim 255$).
* **Critical Lesson:** **Batch Normalization** layers are extremely sensitive to input statistics ($\mu, \sigma$). Feeding unscaled data ($255\times$ larger magnitude) during inference completely shattered the learned statistics. Applying `/ 255.0` to the test set immediately restored accuracy to **99.29%**.

**2. The Power of Batch Normalization**
* **Observation:** As seen in the *Training Cost* graph (Left), the Deep CNN (Orange) starts at a significantly lower cost and converges faster than the Basic CNN (Blue).
* **Analysis:** BN standardizes the inputs to each layer ($\mu=0, \sigma=1$), preventing gradients from vanishing or exploding. This allowed the model to focus on learning complex features from the very first epoch without wasting time adapting to shifting distributions.

**3. The Weight of 0.3% (Error Rate Reduction)**
* **Analysis:** While the accuracy difference ($98.99\% \to 99.29\%$) seems small, it represents a **~30% reduction in the Error Rate** ($1.01\% \to 0.71\%$).
* **Conclusion:** In the high-performance regime (above 98%), marginally increasing accuracy requires exponentially better feature extraction. The combination of **Deeper Layers** (semantic complexity) and **Dropout** (ensemble effect) was necessary to correctly classify the most ambiguous edge cases in MNIST.

---

## 16. Transfer Learning with Pre-trained Models (Day 9)
Transitioned from training networks from scratch to leveraging massive pre-trained architectures (ResNet-18) via Transfer Learning, solving the critical issue of data scarcity in computer vision tasks.

* **Feature Extractor Freezing:** Frozen the pre-trained weights of the ResNet backbone (`requires_grad = False`) to retain the rich hierarchical features learned from ImageNet, preventing catastrophic forgetting during early training.
* **Classifier Replacement:** Dynamically extracted the number of input features (`num_ftrs`) and replaced the final Fully Connected (FC) layer to output 2 classes (Ants vs. Bees) instead of the original 1000.
* **Advanced Data Augmentation:** Implemented a robust `transforms` pipeline including `RandomResizedCrop` and `RandomHorizontalFlip` to artificially expand the highly limited training dataset.
* **Modules:**
  * `day9_transfer_learning.py`: Full implementation of the fine-tuning pipeline, custom training loop with model checkpointing, and visualization logic.

---

## 🔬 Experiment & Analysis: Fine-Tuning ResNet-18 on a Micro Dataset

### Context & Setup
The objective was to evaluate the extreme efficiency of Transfer Learning by fine-tuning a deep CNN on a "micro" dataset that would typically lead to severe overfitting if trained from scratch.
* **Dataset:** Hymenoptera (Ants vs. Bees)
  * **Train:** 244 images
  * **Validation:** 153 images
* **Hyperparameters:** `epochs = 5`, `batch_size = 4`, `lr = 0.001` (SGD with Momentum 0.9).
* **Hardware:** Apple Silicon (MPS).

### 📊 Visualizing the Results

**1. Learning Curve & Convergence**
![Transfer Learning Curve](images/pytorch_practice/day9_learning_curve.png)

**2. Model Prediction Check**
![Model Predictions](images/pytorch_practice/day9_prediction.png)

### 📈 Quantitative Result

| Model | Pre-trained | Train Size | Training Time | Best Val Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet-18 (Fine-tuned)** | Yes (ImageNet) | 244 | **~15 secs** | **94.77%** |

### Key Insights

**1. The "Reversed" Learning Curve Phenomenon**
* **Observation:** As seen in the Learning Curve graph, the Validation Accuracy (Orange) starts significantly higher than the Training Accuracy (Purple).
* **Analysis:** This counter-intuitive result is caused by **Data Augmentation** and **Pre-trained Knowledge**. The training set is heavily distorted (`RandomResizedCrop`, flips) making classification artificially difficult. Conversely, the validation set is cleanly center-cropped, allowing the already-smart ResNet backbone to easily classify the inputs from Epoch 1.

**2. Apple Silicon (MPS) Architecture Constraints**
* **Incident:** Encountered a `TypeError` during accuracy calculation: *"Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64."*
* **Solution:** Identified that Apple's Metal Performance Shaders (MPS) currently lack native support for 64-bit precision (`.double()`). Explicitly downcasted the correct predictions tensor to 32-bit precision using `.float()` before division, successfully enabling GPU acceleration on the Mac environment.

**3. State Dictionary & Memory Isolation (`copy.deepcopy`)**
* **Observation:** In Stochastic Gradient Descent with small datasets, validation loss often fluctuates, meaning the final epoch's model is rarely the optimal one.
* **Implementation:** Designed a checkpointing logic inside the training loop to capture the weights (`model.state_dict()`) whenever a new highest validation accuracy is achieved.
* **Insight:** Crucially utilized `copy.deepcopy()` to physically isolate the saved weights in memory. Using a shallow copy (assignment) would result in the "best" weights being continuously overwritten by subsequent suboptimal updates due to PyTorch's reference-based memory management.

**4. Inverse Normalization for Visualization**
* **Technique:** To visualize the tensor predictions back as raw images, implemented an `Un-normalize` pipeline.
* **Mechanism:** Re-applied the ImageNet statistics by multiplying the standard deviation (`std * img`) and adding the mean (`+ mean`), followed by `np.clip(img, 0, 1)` to handle overflow. This restored the distorted tensors into mathematically correct RGB color spaces for human-readable output.

---

## 17. 3D Vision: Monocular Depth Estimation (Day 11)
Transitioned from 2D image processing to **3D Scene Understanding** by implementing Monocular Depth Estimation, which extracts spatial depth information from a single RGB image.

* **Pre-trained Model:** Leveraged Intel Labs' **MiDaS (v2.1 Small)** to achieve near real-time inference on consumer-grade hardware.
* **Image Pipeline:** Optimized the input flow using OpenCV for BGR-to-RGB correction and the `small_transform` pipeline to ensure tensor compatibility ($256 \times 256$) for the model's mathematical constraints.
* **Tensor Manipulation:** Utilized `unsqueeze(1)` and `interpolate` operations to restore the low-resolution depth output back to the original image dimensions via sophisticated upsampling techniques.

---

## 🔬 Experiment & Analysis: AI Monocular Depth Perception
### Context & Observation
To analyze how the AI perceives spatial distance, a close-up still-life photo (Sushi) was used as the primary input.
* **Setup:** Loaded the `MiDaS_small` model and applied the specific `transforms` required for monocular depth inference.
* **Result:** Generated a high-fidelity **Depth Map** where proximal objects (sushi) are represented with high intensity (white) and distal backgrounds are represented with low intensity (black).

### Key Insights
**1. Semantic Spatial Linearity**
The AI does not rely solely on pixel brightness; it understands **semantic structures**. The smooth gradient transition from the foreground sushi to the background water glass demonstrates the model's ability to maintain spatial continuity across the scene.

**2. Impact of Bicubic Interpolation**
When upsampling the $256 \times 256$ output to match the original resolution, **Bicubic mode** was utilized. Unlike `Nearest` neighbor upsampling, this method calculates neighboring 픽셀을 mathematically to suppress aliasing, ensuring a natural transition at the depth boundaries.

![Depth Result](/images/3d_vision/result_depth.png)

---

## 18. Digital Portrait Mode Implementation
Implemented a digital **Portrait Mode** (Bokeh effect) by using the extracted depth map as a mathematical **Alpha Channel** to blend sharp and blurred image layers.

* **Gaussian Blur:** Applied a spatial smoothing operation using a Gaussian distribution with a $(71 \times 71)$ kernel.
* **Mathematical Blending:** Performed a pixel-wise **Alpha Blending** (Hadamard Product) between the original image ($I_{orig}$) and the blurred image ($I_{blur}$) based on the depth mask ($M$).

---

## 🔬 Experiment & Analysis: Alpha Blending & Kernel Dynamics
### Context & Setup
Conducted hyperparameter tuning to minimize visual artifacts and simulate the optical bokeh characteristic of high-end DSLR lenses.
* **Algorithm:** $$Output = (M \odot I_{orig}) + ((1 - M) \odot I_{blur})$$
* **Hyperparameters:** `ksize = (71, 71)`, `sigma = 0`.

### Key Insights
**1. Min-Max Mask Normalization**
As the model output consists of arbitrary depth values, applying **Min-Max Scaling** to the range $[0.0, 1.0]$ was critical. This transformed the raw depth matrix into a probabilistic weight map, determining exactly what percentage of the original pixel should be preserved vs. blurred.

**2. Kernel Size and Visual Thresholds**
Experiments showed that a kernel size of `71` provided a significantly more pronounced "Bokeh" effect compared to `51`, effectively isolating the subject from the background. The use of an odd-sized kernel ensures a symmetrical central pixel, which is essential for accurate spatial scattering simulation.

**3. Anisotropic Dimension Expansion**
Used `np.expand_dims` and `np.repeat` to align the 2D depth mask with the 3D RGB image. This serves as a practical application of **NumPy Broadcasting** principles to optimize image synthesis pipelines.

![Portrait Result](/images/3d_vision/result_portrait.png)

---

## 19. Object Detection with Faster R-CNN (Day 12)
Transitioned from image classification and monocular depth estimation to **Object Detection**, a comprehensive vision task that simultaneously predicts object locations (Localization via Bounding Boxes) and their categories (Classification).

* **Pre-trained Architecture:** Utilized `fasterrcnn_resnet50_fpn` from `torchvision`, leveraging weights pre-trained on the COCO dataset (capable of detecting 91 object categories).
* **Format Interoperability:** Bridged the gap between OpenCV (`NumPy ndarray`, BGR) and PyTorch's native preprocessing pipeline by converting frames to `PIL Image` format, ensuring the model's specific normalization and scaling transforms were correctly applied to high-resolution inputs ($4000 \times 3000$).
* **Post-processing & Visualization:** Implemented confidence thresholding to filter raw tensor outputs and utilized OpenCV (`cv2.rectangle`, `cv2.putText`) to mathematically map bounding box coordinates back onto the original image.

---

## 🔬 Experiment & Analysis: High-Resolution Object Detection
### Context & Setup
Evaluated the Faster R-CNN model on a custom, high-resolution ($4000 \times 3000$) real-world image containing overlapping objects (laptop, mouse, tumbler, background chairs).
* **Algorithm Pipeline:** OpenCV (Read) $\to$ PIL (Transform) $\to$ PyTorch (Inference) $\to$ Thresholding ($> 0.8$) $\to$ OpenCV (Render).
* **Result:** Filtered 52 raw region proposals down to 8 highly accurate bounding boxes.

### Key Insights
**1. Domain Limitation and Semantic Matching (The "Sushi" Anomaly)**
* **Observation:** When initially tested on a photo of sushi, the model confidently classified the objects as `bowl` or `cup`.
* **Analysis:** The COCO dataset does not contain a 'sushi' class. Rather than failing, the model mapped the visual features to the closest morphological equivalents in its 91-word vocabulary. This highlights that an AI's operational boundary is strictly confined by its training domain data, necessitating Fine-tuning for specialized tasks.

**2. Boolean Indexing for False Positive Suppression**
* **Observation:** The raw inference yielded 52 bounding boxes, many of which were low-confidence noise or duplicates.
* **Implementation:** Applied a Boolean mask (`keep_indices = scores > 0.8`) to slice the `boxes`, `labels`, and `scores` tensors simultaneously. This vectorized thresholding effectively eliminated False Positives without the need for slow, iterative loops. (Note: Internal overlaps were already resolved by the model's native Non-Maximum Suppression (NMS) layer).

**3. Feature Robustness on Out-of-focus Objects**
* **Result:** The model successfully detected and classified a `laptop` (99.93%) and `mouse` (99.83%) in the foreground, while also detecting `chairs` (~98.9%) in the background.
* **Analysis:** Despite the background chairs being heavily blurred (optical bokeh), the ResNet50 backbone coupled with the Feature Pyramid Network (FPN) demonstrated exceptional spatial feature extraction capabilities, proving that Deep CNNs can recognize structural semantics even when high-frequency details (sharp edges) are lost.

**4. Coordinate Systems and BGR Rendering Matrix**
* **Technique:** Rendered bounding boxes using `(0, 255, 0)` and adjusted text placement to `(x1, y1 - 10)`.
* **Analysis:** Unlike standard RGB, OpenCV operates in a **BGR** color space, making `(0, 255, 0)` pure green. Furthermore, because computer vision coordinate systems place the origin $(0,0)$ at the **Top-Left**, subtracting from the $y$-coordinate moves the text *upwards*, preventing the label from obscuring the bounding box stroke.

![Object Detection Result](images/object_detection/result_macbook.jpg)

---

## 20. Semantic Segmentation (Day 13)
### Improved Version: Ablation Study on Model & Domain Shift

To overcome the limitations found in the baseline (incorrect labels and orientation issues), a three-step ablation study was conducted to find the optimal configuration for an XR-based environment.

### Experiment 1: Orientation Correction (DeepLabV3 - PASCAL VOC)
* **Conditions:** Fixed image orientation (Portrait) + Baseline DeepLabV3 (21 classes).
* **Original Baseline (Wrong Orientation)**
<img src="./images/semantic_segmentation/result_deeplabv3.png" width="500px">

* **Experiment 1 Result (Fixed Orientation)**
<img src="./images/semantic_segmentation/result_deeplabv3_rotated.png" width="500px">

* **Result:** Significantly improved segmentation of background objects (Chairs, Person, Table). The model showed **exceptional edge precision**, tracing the curves of the furniture and legs accurately.
* **Limitation:** Due to the limited 21 classes of PASCAL VOC, modern office objects like 'Laptop' and 'Mouse' were still misclassified as 'Dining Table'.

### Experiment 2: Model & Domain Shift (Mask R-CNN - COCO, Threshold 0.7)
* **Conditions:** Switched to Mask R-CNN with COCO weights (80 classes) to detect office-specific objects.
<img src="./images/semantic_segmentation/result_maskrcnn_coco_rotated.png" width="500px">

* **Result:** Successfully identified the 'Tumbler' (Cup) and separated the user's legs as distinct instances.
* **Challenge (The Egocentric Trap):** Discovered the **"Slime Effect"** where masks bled into the background. Since the model was trained on 3rd-person views (Exocentric), the distorted 1st-person (Egocentric) view caused Bounding Box failures, leading to blurred and imprecise edges compared to DeepLabV3.

### Experiment 3: Threshold Tuning (Confidence 0.7 → 0.5)
* **Conditions:** Lowered the confidence threshold to 0.5 to retrieve the missing 'Laptop'.
<img src="./images/semantic_segmentation/result_maskrcnn_coco_rotated_lower_threshold.png" width="500px">

* **Result:** Failed to detect the laptop. Instead, it introduced **Rendering Glitches**.
* **Insight:** Lowering the threshold allowed multiple low-confidence, overlapping masks to survive. This caused a "XOR-like" rendering error where mask interiors appeared hollow or fragmented, proving that threshold adjustment is not a substitute for domain-specific fine-tuning.

---

### 💡 Key Insights & Future Work
1.  **Semantic vs. Instance for XR:** For "Occlusion" effects in XR, **DeepLabV3 (Semantic)**'s pixel-level edge precision is more desirable than Mask R-CNN's box-dependent masks, even if it lacks instance separation.
2.  **Egocentric Vision Gap:** Standard COCO-trained models struggle with the unique geometry of 1st-person views (truncated limbs, skewed table angles). 
3.  **Next Step:** Future experiments should focus on **Fine-tuning** a semantic segmentation model with an egocentric dataset rather than simply switching to a larger exocentric model.
