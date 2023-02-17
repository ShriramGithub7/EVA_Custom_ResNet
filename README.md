{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNc7A9KaGWWrURRczBUTk/F"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Problem Statement:"
      ],
      "metadata": {
        "id": "HUZqTmXwmpfj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "Write a custom ResNet architecture for CIFAR10 that has the following architecture:\n",
        "  PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]\n",
        "  Layer1 -\n",
        "      X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]\n",
        "      R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] \n",
        "      Add(X, R1)\n",
        "  Layer 2 -\n",
        "      Conv 3x3 [256k]\n",
        "      MaxPooling2D\n",
        "      BN\n",
        "      ReLU\n",
        "Layer 3 -\n",
        "      X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]\n",
        "      R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]\n",
        "      Add(X, R2)\n",
        "MaxPooling with Kernel Size 4\n",
        "FC Layer \n",
        "SoftMax\n",
        "Uses One Cycle Policy such that:\n",
        "    Total Epochs = 24\n",
        "    Max at Epoch = 5\n",
        "    LRMIN = FIND\n",
        "    LRMAX = FIND\n",
        "    NO Annihilation\n",
        "Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)\n",
        "Batch size = 512"
      ],
      "metadata": {
        "id": "ZOuq0Fd5n7K3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Solution:\n"
      ],
      "metadata": {
        "id": "hF4zIoSVm2kI"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "Link to custom_ResNet: https://github.com/ShriramGithub7/CNN-Master/blob/main/model/custom_resnet.py\n"
      ],
      "metadata": {
        "id": "aYHu2TwVm50P"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Data Transformation is applied as per the given requirement. Below is git link of master report where this code is present\n",
        "https://github.com/ShriramGithub7/CNN-Master/blob/main/utils/dataTransform.py\n"
      ],
      "metadata": {
        "id": "CSn8cRx-nLs9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Training / Testing code is present at below location:\n",
        "https://github.com/ShriramGithub7/CNN-Master/blob/main/main.py"
      ],
      "metadata": {
        "id": "hRzJl29knhCA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "CR64u4jgnfjJ"
      }
    }
  ]
}
