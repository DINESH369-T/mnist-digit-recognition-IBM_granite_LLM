This repository contains experimental Jupyter Notebooks exploring foundational concepts in both Computer Vision (CV) and Natural Language Processing (NLP).


## 📁 Projects Included in the Notebook

### 1. Handwritten Digit Recognition (MNIST)
A feedforward neural network built with TensorFlow and Keras to classify grayscale images of handwritten digits (0 through 9).

#### 🔍 Step-by-Step Breakdown & Concepts Used:
* **Data Loading (`mnist.load_data()`):** Imports the classic MNIST dataset consisting of 28x28 pixel grayscale images of handwritten digits along with their respective labels.
* **Normalization (`x_train / 255.0`):** Scales pixel values from a range of `[0, 255]` down to `[0.0, 1.0]`. This helps stabilize gradient descent and speeds up neural network convergence.
* **One-Hot Encoding (`to_categorical`):** Converts integer class labels (e.g., `5`) into a categorical vector (e.g., `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`) to match the multi-class probability outputs of the network.
* **Flatten Layer (`Flatten`):** Reshapes the 2D image matrix ($28 \times 28$ pixels) into a 1D flat array ($784$ input features) required for standard dense layers.
* **Dense Layers & ReLU Activation (`Dense(..., activation='relu')`):** Fully connected layers where every neuron connects to all inputs from the previous layer. The **ReLU** (Rectified Linear Unit) activation function introduces non-linearity, allowing the model to learn complex patterns.
* **Softmax Output Layer (`Dense(10, activation='softmax')`):** The final output layer containing 10 neurons (one for each digit). Softmax converts raw model scores into normalized probabilities that sum up to 1.
* **Optimization & Loss (`Adam` & `Categorical Crossentropy`):** 
  * **Adam Optimizer:** An adaptive learning rate optimization algorithm used to efficiently update network weights.
  * **Categorical Crossentropy:** The loss function used to measure the error between predicted probability distributions and the true one-hot encoded labels.
* **Evaluation & Prediction (`model.evaluate` & `np.argmax`):** Measures final test accuracy on unseen data and translates output probabilities back into readable digit predictions.

1. Download the `.ipynb` notebook file from this repository.
2. Open it in [Google Colab](https://colab.research.google.com/).
3. Run the cells sequentially from top to bottom.

### 2. IBM Granite LLM Tokenization & Inference
