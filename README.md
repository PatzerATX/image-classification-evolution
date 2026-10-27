# The Evolution of Image Classification

This project compares how image representation and model architecture affect classification performance on **MNIST** and **CIFAR-10**. It begins with raw pixels and linear classifiers, introduces handcrafted Histogram of Oriented Gradients (HOG) features, and then progresses to fully connected and convolutional neural networks.

The central result is that **representation mattered more than classifier choice**. Replacing raw pixels with HOG produced a large improvement, while changing from logistic regression to a linear SVM with the same HOG features changed accuracy very little. Convolutional neural networks achieved the highest accuracy because they learned spatial features directly from the images.

## Classification Pipelines

The project evaluates five pipelines on each dataset:

1. Raw pixels with multiclass logistic regression
2. HOG features with multiclass logistic regression
3. Selected HOG features with a linear SVM
4. Flattened pixels with a fully connected neural network
5. Spatial image tensors with a convolutional neural network

All experiments use fixed, stratified 80/20 training-validation splits. Hyperparameters are selected using validation results, and the official test sets are reserved for final evaluation.

## Final Results

| Dataset | Representation | Classifier | Validation accuracy | Test accuracy |
|---|---|---|---:|---:|
| MNIST | Raw pixels | Logistic regression | 91.36% | 92.30% |
| MNIST | HOG | Logistic regression | 97.71% | 97.86% |
| MNIST | HOG | Linear SVM | 97.64% | 97.74% |
| MNIST | Flattened pixels | Fully connected network | 97.09% | 97.56% |
| MNIST | Spatial tensor | CNN | **98.78%** | **99.02%** |
| CIFAR-10 | Raw pixels | Logistic regression | 36.24% | 36.55% |
| CIFAR-10 | HOG grayscale | Logistic regression | 51.57% | 50.84% |
| CIFAR-10 | HOG grayscale | Linear SVM | 51.75% | 50.86% |
| CIFAR-10 | Flattened pixels | Fully connected network | 53.64% | 53.96% |
| CIFAR-10 | Spatial RGB tensor | CNN | **67.71%** | **67.28%** |

## Key Findings

- HOG improved test accuracy over raw pixels by **5.56 percentage points on MNIST** and **14.29 points on CIFAR-10**.
- Switching from HOG logistic regression to HOG linear SVM changed test accuracy by only **-0.12 points on MNIST** and **+0.02 points on CIFAR-10**.
- The CNN produced the best result on both datasets: **99.02% on MNIST** and **67.28% on CIFAR-10**.
- MNIST was easier for every pipeline because its centered grayscale digits have consistent structure and backgrounds.
- CIFAR-10 benefited more from learned spatial features because its objects vary in position, viewpoint, color, background, and appearance.
- The CIFAR-10 neural-network experiments showed overfitting as training performance continued to improve after validation performance had leveled off.

## Repository Contents

- [`project1_image_classification.ipynb`](./project1_image_classification.ipynb): complete Colab workflow and experimental outputs
- [`Evolution of Image Classification.pdf`](./Evolution%20of%20Image%20Classification.pdf): final project report
- [`prepare_data.py`](./prepare_data.py): optional local dataset preparation and inspection script
- [`requirements.txt`](./requirements.txt): Python dependencies for local execution

MNIST and CIFAR-10 are downloaded automatically through `torchvision`. Dataset files and generated outputs are excluded from version control.

## Run in Google Colab

1. Open `project1_image_classification.ipynb` in Google Colab.
2. Select a GPU runtime for the neural-network sections: **Runtime > Change runtime type > T4 GPU**.
3. Run the notebook from top to bottom.

The classical models can run on a CPU, but HOG extraction and model training may take several minutes. The neural-network and CNN experiments run substantially faster with a GPU.

## Run Locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py
```

The notebook can also be opened locally with Jupyter after installing the dependencies.

## Tools and Libraries

- NumPy and pandas for data preparation and experiment summaries
- Matplotlib and seaborn for visualizations
- scikit-image for HOG feature extraction
- scikit-learn for logistic regression, feature standardization, and linear SVM
- PyTorch and Torchvision for datasets, fully connected networks, and CNNs

## Reproducibility

- Fixed random seed: `1234`
- Fixed stratified training-validation splits
- Standardization fitted only on training features
- Validation data used for model selection
- Test data used only after selecting the final configuration

## Course

Completed for **CS 4337: Computer Vision** at Texas State University, Fall 2026.
