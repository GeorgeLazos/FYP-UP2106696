=================================== INFO ============================================
- Mini Batch Gradient Descent(MBGD) CNN Implementation for Pneumonia Detection In Xray Images

       - For Stochastic Gradient Descent(SGD) use Batch size 1

       - For Batch Gradient Descent(BGD) use Batch size of total number of samples (5216 for current TRAIN dataset)

- Uses He's Initialization for weights

- Uses Relu activation in conv layers and Leaky Relu in dense layers except for last layer which uses Sigmoid activation

- Uses Binary Cross Entropy Loss Function with weighted loss for imbalanced dataset

- Applies Batch Normalization after each conv and dense layer(except last dense)

- Uses Max Pooling after each conv layer

- Uses L2 regularization in dense layers

- Uses LR decay

- Uses Dropout

- Uses image augmentation to expand dataset artificially

- Applies early stopping
