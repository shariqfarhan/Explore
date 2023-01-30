# Assignment Submission for Session 5 - Batch Normalization & Regularization

In this assignment we aim to learn about different types of normalizations. The objective of this assignment is to implement a Neural network which would take in the type of normalization as an input and then create the neural network.

We compare the performance of 3 types of Normalizations

1.   Network with Group Normalization
2.   Network with Layer Normalization
3.   Network with L1 Regularization + Batch Normalization

## Summary

**Target**


1.   Define neural network in such a way that we can do any normalization in the same network
2.   Identify sample mis-classified images for each normalization technique


**Result**

*   We were able to successfully create a neural network with all 3 different types of Normalizations
*   We also implemented L1 Regularization for the model with Batch Normalization


**Analysis**

1.   For MNIST dataset, L1 + Batch Normalization & Group Normalization have a positive effect on model performance i.e. both of these improve model performance
1.   But Layer Normalization drastically reduces the model performance, such that the accuracy hovers < 10%
2.   As the number of epochs increase, L1 Regularization + Batch Normalization (BN) performs better than Group Normalization (GN) . We see a clear superiority in test accuracy in BN.
3.   The Test losses for GN are higher than L1 + BN and the test accuracy is lower 


## Comparison between Normalizations

**Performance**

One of the key objectives of this assignment was to determine the impact of Normalization on model performance. The below 2 graphs show a comparison between Normalizations on 2 key metrics - Test Loss & Test Accuracy.

As we can see, adding Layer Normalization to the network drastically reduces performance leading to high losses and much lower accuracy. Comparison of Group Normalization to the combination of L1 Regularization & Batch Normalization outperform Layer Normalization on MNIST dataset and they seem to go toe-to-toe in terms of performance.

<img width="466" alt="image" src="https://user-images.githubusercontent.com/57046534/215472072-db7ee27f-13ae-4cf3-afb9-0cf94e6fd706.png"> <img width="470" alt="image" src="https://user-images.githubusercontent.com/57046534/215472126-f5a36403-b93d-42b1-ad1a-9e2badfc278d.png">

In order to determine the winner between these 2 i.e. L1 Regularization(L1) & Batch Normalization (BN) and Group Normalization (GN), a closer look is needed. When we do so, we see that - L1 + BN performs much better than GN

<img width="410" alt="image" src="https://user-images.githubusercontent.com/57046534/215493068-854262da-1c96-4ef1-9698-3f70d3cf3b3b.png"> <img width="425" alt="image" src="https://user-images.githubusercontent.com/57046534/215493007-3c20b061-bf82-44c0-a401-694ccf6152c6.png">

While the L1 + BN is a smoother learning curve, we see a few highs and lows in the GN graph. Batch Normalization seems to have a better performance than Group Normalization. We need to check if this is the case even without L1 Regularization.


## Mis-Classified Images

Below we see the output of misclassified images for the 3 types of normalizations. For Layer Normalization, the images are relatively easier to classify which the network misses. 

In the case of L1 + BN and GN, some of these are really difficult to predict, even for a human.

<img width="1134" alt="image" src="https://user-images.githubusercontent.com/57046534/215488339-04789734-8d2e-404c-a5cd-966ff20464ee.png">

<img width="1134" alt="image" src="https://user-images.githubusercontent.com/57046534/215490495-9ad36c89-a9f2-471a-b8f9-0a8146fffe2f.png">

<img width="1134" alt="image" src="https://user-images.githubusercontent.com/57046534/215492650-a9453d6d-2fd2-411a-95f4-37ca906714d3.png">


