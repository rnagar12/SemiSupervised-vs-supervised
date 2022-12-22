## In the real world labeled data is scarce, or consumes a large amount of time and money to acquire. Semi-supervised learning is designed to yield similar results to fully supervised learning but with a fraction of labeled data. Using the STL-10 dataset and the new BYOL self-supervised training method, a semi-supervised model was trained and evaluated.

### What is Bootstrap Your Own Latent (BYOL)?

Please see the original Paper: https://arxiv.org/abs/2006.07733

<img width="1317" alt="Screen Shot 2022-12-21 at 3 45 19 pm" src="https://user-images.githubusercontent.com/91589070/208823519-57e64d3b-180d-4049-8fe7-d06eadfe2598.png">

A single image is augmented differently and fed into two networks; target and online. The networks use the output from each other to predict the original image correctly. The online network learns from the target at a fast rate, whereas the target network learns from online much slower. This is to ensure that there still remains a connection to the original image and its correct classification. 

The target network in this case acts as a 'ground truth' for the online network to predict. The target network also has a smaller architecture whereas the online network is deeper.

This process is carried out using entirely unlabelled data. 

The loss function of this arrangement is interesting. It assumes the online prediction step to be predicting the output of the target network. This is not symmetrised to avoid a situation where the networks minimise loss by converging to identical image vectors, thus resulting in an incorrect loss of 0.

```math
L _{\theta, \xi } \overset{\underset{\mathrm{\triangle }}{}}{=} \left\|\bar{q_{\theta }}\left ( z_{\theta } \right ) - \bar{z'_{\xi }} \right\|_{2}^{2}
```
### How was it all done?

1. Train a supervised ResNet model as a baseline
2. Train a BYOL unsupervised model with unlabelled data
3. Train a model using labelled data by taking the encoder 'f' from the online network of the BYOL model and placing it into a new ResNet model

Training was carried out using Google Colab and a custom VM created on Google Cloud. Unsupervised training (step 2) was maintained at 5 epochs with the final step 3 being carried out for 50 epochs.

#### Dataset

To simulate real world conditions STL-10 was used by allocating 10% of the total training set to be used for labelled training. This corresponded to 500 images. 90% of the remaining set was used for labelled validation corresponding to 4500 images.

The entire set of 100,000 unlabelled images was used for unlabelled training. 

The total 8000 images were used for testing.

#### Image augmentation
<img width="467" alt="Screen Shot 2022-12-21 at 4 49 16 pm" src="https://user-images.githubusercontent.com/91589070/208831185-1169551c-9b3d-4ca6-8855-eff267cfb353.png">
Above is the original image augmentation setup used in the paper linked above. The numbers in the columns refer to the probability of the augmentation being applied.

The authors of the paper found that the primary contributors to increased accuracy of the network was through the colour jittering and cropping. To test this almost all transformations were removed from both T and T’ except for a crop and flip to still maintain some difference between the two networks’ input image. The result was a testing accuracy of 37.68% and a training time of 41 minutes on a ResNet18 model. This finding was surprising, as the drop in accuracy was very small even though almost all augmentations were removed. The highest performance was found to be when T’ was augmented with a 1.0 probability on all but the flip transformations, whereas the T augmentation set was kept as per baseline. This resulted in a testing accuracy of 39.60% with a training time of 39 minutes. This augmentation set was maintained as the optimum and the files are named as such.

### How to run?

1. Upload the Colab files to your drive, open in Colab
2. Edit the directory paths in block 3 to point correctly to your dataset location
3. Run the blocks sequentially
