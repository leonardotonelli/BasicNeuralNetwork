# Neural Network from Scratch

## Overview

This project implements a simple feedforward neural network from scratch using NumPy. The neural network is designed for multi-class classification tasks and employs backpropagation for training. The example provided demonstrates how to train the network on a synthetic dataset generated using the `make_blobs` function from the `sklearn` library.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Neural Network Structure](#neural-network-structure)
- [Training Process](#training-process)
- [Results](#results)
- [License](#license)

## Features

- **Feedforward Neural Network**: Implements a multi-layer neural network with customizable architecture.
- **Backpropagation**: Trains the network using backpropagation and gradient descent.
- **Activation Functions**: Supports sigmoid activation for hidden layers and softmax for the output layer.
- **One-Hot Encoding**: Handles multi-class labels using one-hot encoding.
- **Loss Function**: Implements cross-entropy loss for training evaluation.

## Installation

To run this project, you need to have Python 3.x installed along with the required libraries. You can install the necessary libraries using pip:

```bash
pip install numpy scikit-learn
