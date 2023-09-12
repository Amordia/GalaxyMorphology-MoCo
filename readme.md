# Galaxy Morphology Classification with MoCo

This repository contains the code and documentation for a semi-supervised machine learning algorithm based on MoCo (Momentum Contrast) aimed at galaxy morphology classification.

## Background

Galaxy morphology research is an integral part of astronomical studies. It aids in our understanding of the history and evolution of the universe. The traditional classification of galaxies is a complex and time-consuming task. Therefore, there's a growing interest in developing automated methods for large-scale galaxy morphology data. This project leverages the power of deep learning and MoCo for the task.

## Project Structure

- `plot.py`: Script for generating result visualizations.
- `resnet18_MoCo_test.py`: Testing script for the model based on ResNet-18 and MoCo.
- `MoCo_DNN.py`: Main implementation of the MoCo-based semi-supervised algorithm.
- `resnet18_MoCo.py`: Training script for the model based on ResNet-18 and MoCo.

## Results

While the model showed promise, the current accuracy stands at around 60% for the four-category classification task. The performance metrics, such as per-class accuracy and ROC-AUC, vary across categories, indicating potential areas for improvement.

## Dependencies

- **Python Libraries**:
    - numpy
    - pandas
    - matplotlib
    - pylab
    - astropy (specifically `Table`)
    - PIL (specifically `ImageFilter`)
- **PyTorch and related libraries**:
    - torch (and its submodules like `autograd`, `nn`, and `utils.data`)
    - torchvision (and its models like `resnet18`)
    - torch.nn.functional

## How to Run

1. Ensure you have all the dependencies installed:
    ```bash
    pip install numpy pandas matplotlib pylab astropy torchvision torch
    ```

2. Clone the repository: 
    ```bash
    git clone [URL of this repository]
    ```

3. Navigate to the project directory:
    ```bash
    cd [Project Directory]
    ```

4. Start by training the MoCo model with ResNet-18 architecture:
    ```bash
    python resnet18_MoCo.py
    ```

5. Once the training is complete, test the model:
    ```bash
    python resnet18_MoCo_test.py
    ```

6. Finally, run the classification script:
    ```bash
    python MoCo_DNN.py
    ```

7. For result visualizations, you can use:
    ```bash
    python plot.py
    ```