# Food Image Classification Project

A comprehensive machine learning project for multiclass food image classification using the food-101 dataset. This project compares multiple approaches including Convolutional Neural Networks (CNNs), traditional Neural Networks, and Support Vector Machines (SVMs).

## Authors
- Michael Wang
- Jay Zhu

## Research Paper
**Paper Title:** Multi-Class Food Image Classification Across 101 Categories  
**Link:** [Google Docs](https://docs.google.com/document/d/1vjy_1jv1RdCnkwuTe5NayBLfGsdrQFyLz-9ItQAVns8/edit?usp=sharing)

## Project Overview

This project implements and compares various machine learning approaches for classifying food images across 101 different food categories. The implementation includes comprehensive experimentation with different model architectures, hyperparameter tuning, and evaluation metrics.

## Dataset

- **Dataset:** food-101
- **Categories:** 101 different food classes
- **Task:** Multiclass image classification
- **Data Augmentation:** RandomFlip, RandomRotation, RandomZoom, and other techniques

## Model Architectures

### 1. Convolutional Neural Networks (CNNs)
- **3-layer CNN:** Basic convolutional architecture
- **5-layer CNN:** Deeper convolutional network with enhanced feature extraction
- Multiple kernel sizes and pooling strategies
- Batch normalization and dropout for regularization

### 2. Neural Networks
- **Flatten-based architectures:** Traditional feedforward networks
- Various hidden layer configurations
- Hyperparameter optimization for layer sizes and activation functions

### 3. Support Vector Machines (SVMs)
- **Linear SVM:** Basic linear classification
- **RBF Kernel SVM:** Non-linear classification with radial basis function
- **Polynomial Kernel SVM:** Non-linear classification with polynomial kernels
- Grid search for optimal hyperparameters

### 4. Dimensionality Reduction
- **Principal Component Analysis (PCA):** Unsupervised analysis for feature reduction
- Variance analysis and component visualization

## Technical Implementation

### Framework & Libraries
- **TensorFlow/Keras:** Deep learning framework
- **scikit-learn:** Traditional machine learning algorithms
- **NumPy/Pandas:** Data manipulation and analysis
- **Matplotlib/Seaborn:** Visualization and plotting

### Data Processing
- Image preprocessing and normalization
- Data augmentation for improved generalization
- Train/validation/test split management

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Model comparison analysis

## Getting Started

### Prerequisites
```bash
pip install tensorflow
pip install scikit-learn
pip install numpy pandas matplotlib seaborn
```

### Running the Project
1. Clone this repository
2. Open `Machine_Learning_Final.ipynb` in Jupyter Notebook
3. Execute cells sequentially to reproduce experiments
4. Test images are available in the `test/` directory

## Project Structure
```
food-image-predictor/
├── Machine_Learning_Final.ipynb    # Main implementation notebook
├── test/                          # Test images for various food categories
├── README.md                      # Project documentation
└── LICENSE                        # Project license
```

## Results

In this project, we explored food image classification using the **Food-101** dataset, which contains 101,000 images across 101 food categories. We implemented and compared three machine learning models:

- **Support Vector Machines (SVMs):**  
  SVMs struggled with the dataset's complexity, achieving low accuracy (~5.6%). While RBF kernels and regularization helped slightly, SVMs are not ideal for high-dimensional, variable image data.

- **Traditional Neural Networks:**  
  Flattened image inputs (~250,000 features) led to poor performance (~6% accuracy). The model underfit the data and lacked the complexity to learn spatial patterns and textures.

- **Convolutional Neural Networks (CNNs):**  
  CNNs achieved the best results, with validation accuracy around **26%**. Their ability to capture shapes and patterns makes them well-suited for this task. With deeper architectures and longer training, CNNs have strong potential for further improvement.

CNNs offer a promising foundation for food image classification. With further training and expansion to more categories, this approach can support applications in:

- 🍽️ Dietary tracking  
- 🤖 AI cooking assistants  
- 📍 Restaurant recommendation systems


## Test Images

The project includes test images for various food categories including:
- Burgers, Cheesecake, Chicken Wings
- Crème Brûlée, Dumplings, and more
- Representative samples from the food-101 dataset

## Future Work

- Implement transfer learning with pre-trained models
- Explore additional data augmentation techniques
- Experiment with ensemble methods
- Deploy model for real-time food classification

## License

This project is licensed under the terms specified in the LICENSE file.

