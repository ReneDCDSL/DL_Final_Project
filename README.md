# Project for DLC 2021

## Structure

## Project 1 - Classification, weight sharing, auxiliary losses
The objective of this project is to test different architectures to compare two digits visible in a
two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an
auxiliary loss to help the training of the main objective.

It should be implemented with PyTorch only code, in particular without using other external libraries
such as scikit-learn or numpy.

### 1.1 Data
The goal of this project is to implement a deep network such that, given as input a series of 2 × 14 × 14
tensor, corresponding to pairs of 14 × 14 grayscale images, it predicts for each pair if the first digit is
lesser or equal to the second.

The training and test set should be 1, 000 pairs each, and the size of the images allows to run
experiments rapidly, even in the VM with a single core and no GPU.
You can generate the data sets to use with the function `generate_pair_sets(N)` defined in the file
`dlc_practical_prologue.py`. This function returns six tensors:
![tensors table](/figures/tensors-table.png)

### 1.2 Objective
The goal of the project is to compare different architectures, and assess the performance improvement
that can be achieved through weight sharing, or using auxiliary losses. For the latter, the training can
in particular take advantage of the availability of the classes of the two digits in each pair, beside the
Boolean value truly of interest.

All the experiments should be done with 1, 000 pairs for training and test. A convnet with ∼ 70, 000
parameters can be trained with 25 epochs in the VM in less than 2s and should achieve ∼ 15% error rate.

Performance estimates provided in your report should be estimated through 10+ rounds for each
architecture, where both data and weight initialization are randomized, and you should provide estimates
of standard deviations.

## Project 2 - Mini deep-learning framework
The objective of this project is to design a mini “deep learning framework” using only pytorch’s
tensor operations and the standard math library, hence in particular *without using autograd or the
neural-network modules*.

### 2.1 Objective 


