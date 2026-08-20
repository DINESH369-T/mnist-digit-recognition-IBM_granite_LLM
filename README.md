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

---
### 2. IBM Granite LLM Tokenization & Inference
An implementation using the Hugging Face `transformers` library to load, tokenize, and prompt an instruction-tuned Large Language Model (**IBM Granite 3.1 2B Instruct**).

#### 🔍 Step-by-Step Breakdown & Concepts Used:
* **Library Imports (`transformers` & `torch`):** 
  * Imports `AutoModelForCausalLM` and `AutoTokenizer` from Hugging Face, which provide standardized classes to easily download and interface with pre-built transformer models.
  * Imports `PyTorch` (`torch`) as the underlying tensor and deep learning framework.
* **Model Identification (`model_id`):** 
  * Sets the model identifier string (`'ibm-granite/granite-3.1-2b-instruct'`), pointing directly to IBM's open-weights model hosted on Hugging Face. The `2b` denotes a lightweight 2-billion parameter model optimized for efficient edge and server deployment.
* **Tokenizer Initialization (`AutoTokenizer.from_pretrained`):** 
  * Loads the specific text tokenizer tied to the model. A tokenizer's job is to split raw human text into sub-words or tokens and convert them into numerical IDs that the neural network can process.
* **Model Loading & Optimization (`AutoModelForCausalLM.from_pretrained`):** 
  * Downloads and initializes the multi-layer transformer network. 
  * **`torch_dtype=torch.float16` (FP16 Precision):** Loads model weights using 16-bit half-precision instead of standard 32-bit floating-point, reducing the memory (VRAM) footprint by half and accelerating computation.
  * **`device_map='auto'`:** Automatically inspects available hardware (CPU vs. GPU) and maps the model layers to the optimal hardware device.
* **Chat Structuring (`messages`):** 
  * Organizes the input prompt using a standardized dictionary format (`'role': 'user'`, `'content': '...'`). This chat template structure is crucial for instruction-tuned models to accurately differentiate between system instructions, user questions, and model responses.
---
1. Download the `.ipynb` notebook file from this repository.
2. Open it in [Google Colab](https://colab.research.google.com/).
3. Run the cells sequentially from top to bottom.
