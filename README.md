Image Classification with using CNN on CIFAR10 Dataset

While still traditional neural networks have achieved commendable image classification performance, they have been defined by feature engineering, 
a time-consuming process that results in poor generalization to test data. I offer a convolutional neural network (CNN) strategy for classifying CIFAR-10 datasets 
in this report. This strategy has previously been shown to increase performance without the use of feature engineering. To extract underlying image information, 
learnable filters and pooling layers were used. Dropout, regularization, and variation in convolution techniques were used to reduce overfitting while maintaining 
high validation and testing accuracy. A deeper network improved test accuracy while reducing overfitting.

This repository based on article which is about classification on CIFAR10 Dataset. Article is used tensorflow but ı used pytorch to set convolutional neural networks. 
After using dropout and L2 regularization in convolution techniques, I got 0.6960 in test data. Besides, I loaded results which show about training and validation 
accuracy and loss.

