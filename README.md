# Facial-Attribute-Recognition

[PDF](paper.pdf)

## Project Overview

This project aims to recognize various facial attributes in images. This is achieved by using the Transformer architecture, a model architecture that uses self-attention mechanisms and has been successful in various tasks in the field of natural language processing.

My model is built upon the EfficientFormerV2 transformer backbone, a creation of Snap Inc., Northeastern University, and UC Berkeley. More details can be found [here](https://github.com/snap-research/EfficientFormer).

## Results

The current iteration of the model achieves 90.9% accuracy on the CelebA dataset.
It uses 15.6M parameters and with an inference latency of 1.18ms on an iPhone 14 Pro Max it is well suited of real-time facial attribute classification.

Below are comparisons of our model with other state-of-the-art models. The baseline is established by predicting attributes on the test dataset based on the attribute distribution in the training dataset. If an attribute probability is below 0.5, it is subtracted from 1.
![Comparison of results with other models in a Radar Plot](images/radarplot.png)
![Comparison of results with other models in a Bar Chart](images/barchart.png)

## Installation and Setup

1. Clone the repository: `git clone https://github.com/Cyanosite/Intel-Image-Classification.git`
2. Navigate into the project directory: `cd Facial-Attribute-Recognition`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Execute the `solution.ipynb` notebook to train the model, evaluate its performance, and reproduce the results.

## Contact

Zsombor Szenyán - zsomborszenyan@edu.bme.hu
