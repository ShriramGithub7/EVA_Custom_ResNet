# Probelm Statement


Write a custom ResNet architecture for CIFAR10 that has the following architecture:
  * PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  * Layer1 -
      * X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
      * R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
      * Add(X, R1)
  * Layer 2 -
      * Conv 3x3 [256k]
      * MaxPooling2D
      * BN
      * ReLU
* Layer 3 -
      * X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
      * R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
      * Add(X, R2)
* MaxPooling with Kernel Size 4
* FC Layer 
* SoftMax
* Uses One Cycle Policy such that:
    * Total Epochs = 24
    * Max at Epoch = 5
    * LRMIN = FIND
    * LRMAX = FIND
    * NO Annihilation
* Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
* Batch size = 512

# Solution

* Custom resnet model is written in master repo as per above criteria
  *  custom_resnet.py link - https://github.com/ShriramGithub7/CNN-Master/blob/main/model/custom_resnet.py
  <br><br>
* Data Transformation is applied as per the given requirement
  * utils folder has this dataTransform.py file. Link: https://github.com/ShriramGithub7/CNN-Master/blob/main/utils/dataTransform.py
 <br><br>
* Training / Testing code is written in master repo
  * Link for training and testing code: https://github.com/ShriramGithub7/CNN-Master/blob/main/main.py
  <br>
* Below are logs after training model for 24 epochs
<br>
  
  
    EPOCH: 1
    Loss=-0.42 Batch_id=781 Accuracy=26.15: 100%|██████████| 782/782 [00:35<00:00, 22.22it/s]_

    Test set: Average loss: -0.3751, Accuracy: 3772/10000 (37.72%)

    EPOCH: 2
    Loss=-0.55 Batch_id=781 Accuracy=40.59: 100%|██████████| 782/782 [00:34<00:00, 22.63it/s]

    Test set: Average loss: -0.4351, Accuracy: 4373/10000 (43.73%)

    EPOCH: 3
    Loss=-0.54 Batch_id=781 Accuracy=47.62: 100%|██████████| 782/782 [00:35<00:00, 22.18it/s]

    Test set: Average loss: -0.5398, Accuracy: 5417/10000 (54.17%)

    EPOCH: 4
    Loss=-0.56 Batch_id=781 Accuracy=50.75: 100%|██████████| 782/782 [00:35<00:00, 21.81it/s]

    Test set: Average loss: -0.5366, Accuracy: 5370/10000 (53.70%)

    EPOCH: 5
    Loss=-0.50 Batch_id=781 Accuracy=55.68: 100%|██████████| 782/782 [00:36<00:00, 21.61it/s]

    Test set: Average loss: -0.5866, Accuracy: 5893/10000 (58.93%)

    EPOCH: 6
    Loss=-0.47 Batch_id=781 Accuracy=56.67: 100%|██████████| 782/782 [00:34<00:00, 22.87it/s]

    Test set: Average loss: -0.5922, Accuracy: 5935/10000 (59.35%)

    EPOCH: 7
    Loss=-0.44 Batch_id=781 Accuracy=56.66: 100%|██████████| 782/782 [00:34<00:00, 22.37it/s]

    Test set: Average loss: -0.5984, Accuracy: 5992/10000 (59.92%)

    EPOCH: 8
    Loss=-0.39 Batch_id=781 Accuracy=59.78: 100%|██████████| 782/782 [00:36<00:00, 21.31it/s]

    Test set: Average loss: -0.6055, Accuracy: 6083/10000 (60.83%)

    EPOCH: 9
    Loss=-0.50 Batch_id=781 Accuracy=61.01: 100%|██████████| 782/782 [00:34<00:00, 22.76it/s]

    Test set: Average loss: -0.6326, Accuracy: 6339/10000 (63.39%)

    EPOCH: 10
    Loss=-0.53 Batch_id=781 Accuracy=62.31: 100%|██████████| 782/782 [00:34<00:00, 22.36it/s]

    Test set: Average loss: -0.6450, Accuracy: 6459/10000 (64.59%)

    EPOCH: 11
    Loss=-0.52 Batch_id=781 Accuracy=63.20: 100%|██████████| 782/782 [00:34<00:00, 22.86it/s]

    Test set: Average loss: -0.6467, Accuracy: 6467/10000 (64.67%)

    EPOCH: 12
    Loss=-0.62 Batch_id=781 Accuracy=64.26: 100%|██████████| 782/782 [00:34<00:00, 22.62it/s]

    Test set: Average loss: -0.6585, Accuracy: 6600/10000 (66.00%)

    EPOCH: 13
    Loss=-0.56 Batch_id=781 Accuracy=65.02: 100%|██████████| 782/782 [00:34<00:00, 22.57it/s]

    Test set: Average loss: -0.6686, Accuracy: 6699/10000 (66.99%)

    EPOCH: 14
    Loss=-0.72 Batch_id=781 Accuracy=65.68: 100%|██████████| 782/782 [00:35<00:00, 22.03it/s]

    Test set: Average loss: -0.6601, Accuracy: 6611/10000 (66.11%)

    EPOCH: 15
    Loss=-0.53 Batch_id=781 Accuracy=66.24: 100%|██████████| 782/782 [00:34<00:00, 22.64it/s]

    Test set: Average loss: -0.6762, Accuracy: 6761/10000 (67.61%)

    EPOCH: 16
    Loss=-0.69 Batch_id=781 Accuracy=66.82: 100%|██████████| 782/782 [00:34<00:00, 22.70it/s]

    Test set: Average loss: -0.6794, Accuracy: 6799/10000 (67.99%)

    EPOCH: 17
    Loss=-0.57 Batch_id=781 Accuracy=67.48: 100%|██████████| 782/782 [00:34<00:00, 22.75it/s]

    Test set: Average loss: -0.6847, Accuracy: 6848/10000 (68.48%)

    EPOCH: 18
    Loss=-0.56 Batch_id=781 Accuracy=68.05: 100%|██████████| 782/782 [00:34<00:00, 22.54it/s]

    Test set: Average loss: -0.6880, Accuracy: 6895/10000 (68.95%)

    EPOCH: 19
    Loss=-0.62 Batch_id=781 Accuracy=68.69: 100%|██████████| 782/782 [00:34<00:00, 22.69it/s]

    Test set: Average loss: -0.7015, Accuracy: 7028/10000 (70.28%)

    EPOCH: 20
    Loss=-0.56 Batch_id=781 Accuracy=69.20: 100%|██████████| 782/782 [00:35<00:00, 22.18it/s]

    Test set: Average loss: -0.7012, Accuracy: 7029/10000 (70.29%)

    EPOCH: 21
    Loss=-0.64 Batch_id=781 Accuracy=69.84: 100%|██████████| 782/782 [00:34<00:00, 22.71it/s]

    Test set: Average loss: -0.7084, Accuracy: 7090/10000 (70.90%)

    EPOCH: 22
    Loss=-0.75 Batch_id=781 Accuracy=70.38: 100%|██████████| 782/782 [00:34<00:00, 22.66it/s]

    Test set: Average loss: -0.7067, Accuracy: 7070/10000 (70.70%)

    EPOCH: 23
    Loss=-0.63 Batch_id=781 Accuracy=70.99: 100%|██████████| 782/782 [00:34<00:00, 22.75it/s]

    Test set: Average loss: -0.7124, Accuracy: 7134/10000 (71.34%)

    EPOCH: 24
    Loss=-0.69 Batch_id=781 Accuracy=71.69: 100%|██████████| 782/782 [00:34<00:00, 22.75it/s]

    Test set: Average loss: -0.7123, Accuracy: 7127/10000 (71.27%)_


  
