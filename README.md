# Galaxy Morphology Classification using Momentum Contrast (MoCo)

This repository presents the semi-supervised approach for galaxy morphology classification using the Momentum Contrast (MoCo) algorithm. MoCo, a contemporary contrastive learning method, learns robust features without requiring extensive labeled data, making it ideal for tasks like galaxy morphology classification where labeled samples can be scarce.

## Background and Significance

### Project Background

Astronomical studies, particularly those concerning the universe's history and evolution, heavily rely on galaxy morphology research. The task of galaxy classification, while invaluable, poses challenges given its complexity and the extensive human expertise it demands. The quest for a proficient automated technique, capable of handling large-scale galaxy morphology data, is thus paramount.

While deep and machine learning has made strides in image classification, opening up prospects for automated galaxy morphology classification, traditional supervised techniques often falter due to their dependency on vast labeled datasets. The unique characteristics of astronomical data, including its inherent noise and imbalance, further complicate the direct application of conventional deep learning models.

### Why MoCo for Galaxy Morphology?

The Momentum Contrast (MoCo) technique, a novel form of contrastive learning, brings forward a solution by contrasting different augmented versions of a sample, referred to as the anchor and the positive, against other samples' negative instances. This method efficiently learns feature representations without the need for extensive labeled data, offering a workaround to data imbalance and noise issues.

In the context of galaxy morphology classification, the MoCo approach, leveraging ResNet-18 as its backbone, offers advantages such as computational efficiency given ResNet-18's lightweight architecture, proven performance across various image classification tasks, and resilience against gradient vanishing issues due to its residual structure.

By employing MoCo, this project aspires to enhance the accuracy and efficiency of galaxy morphology classification, driving forward astronomical research and promoting the application of semi-supervised and contrastive learning techniques in astronomical data processing.

## Results and Evaluations

MoCo demonstrates a significant potential in the domain of galaxy morphology classification. Preliminary experiments achieved an accuracy rate of around 75% for this task. While this may not surpass some of the existing techniques, the results remain promising when one considers the semi-supervised nature of the approach. Detailed performance metrics and visualizations can be found in the `results/` directory.

For a holistic understanding and comparative insight, consider exploring the related repositories:
- [Supervised Galaxy Morphology Classification using ResNet50](https://github.com/Amordia/GalaxyMorphology-ResNet50.git)
- [Semi-Supervised Galaxy Morphology Classification using Convolutional AutoEncoders (CAE)](https://github.com/Amordia/GalaxyMorphology-CAE.git)

## Project Structure
- `plot.py`: Script for generating result visualizations.
- `resnet18_MoCo_test.py`: Testing script for the model based on ResNet-18 and MoCo.
- `MoCo_DNN.py`: Main implementation of the MoCo-based semi-supervised algorithm.
- `resnet18_MoCo.py`: Training script for the model based on ResNet-18 and MoCo.

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

## License

This project falls under the MIT License.


