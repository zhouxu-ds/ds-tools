# Machine Learning Fundamentals

Here are some notes that I jot down from [Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do). Please contact me if there is any copyright issue.

## Table of Content

- [What is machine learning?](#what)

- [Why use machine learning?](#why)

- [Types of machine learning systems](#types)

- [Example 1-1 using Linear Regression](#example_1)

- [Main challenges in machine learning](#challenges)

<a name="what"></a>

## What is machine learning? 

- Machine Learning is the science (and art) of programming computers so they can learn from data . 
- A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. Tom Mitchell, 1997

 <a name="why"></a>

## Why use machine learning? 

1. It can be easier to update, maintain and most likely more accurate compared to traditional approaches.
2. Some of the problems may be too complex for traditional approaches, or have no algorithms.
3. Can dig into data and discover patterns that are not immediately apparent.

<a name="types"></a>

## Types of machine learning systems 

- **Supvervised vs Unsupervised**
  - **Supervised**: with labels
    - k-nearest neighbors
    - linear regression
    - logistic regression
    - support vector machines (SVMs)
    - decision  trees and random forests
  - **Unsupervised**: without labels
    - clustering
    - visualization and dimensionality reduction
    - association rule learning
  - **Semisupervised learning**: usually a lot of unlabeled data with a little bit of labeled data, eg. Google photos
  - **Reinforcement learning**: Uses reward and penalties to guide
- **Batch vs online learning**
  - **Batch**: In batch learning , the system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called offline learning . 
  - **Online**: In online learning , you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini-batches . Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.
- **Instance-based vs model-based learning**
  - **Instance-based**: The system learns the example by heart and then generalizes to new cases using a similarity measure.
  - **Model-based**: Build models and then make predictions.

<a name="example_1"></a>

## Example 1-1 

See [html](http://htmlpreview.github.io/?https://github.com/xuzhou338/DS_tools/blob/master/machine_learning/ML_fundamentals/linear_regression_example.html) or [jupyter notebook](linear_regression_example.ipynb) 

Modified from the book.

<a name="challenges"></a>

## Main challenges in machine learning

- **Insufficient quantity of training data**: more data usually lead to better performance.
- **Nonrepresentative training data**: watch out the sampling bias.
- **Poor-quality data**: errors, missing data, outliers and noise.
- **Irrelevant features**: need to consider feature selection and feature extractions.
- **Overfitting the training data**: Happens when the model is too complex relative to the amount and noisiness of the training data. Can be constrained through <u>regularization</u> and control by <u>hyperparameters</u>.
- **Under fitting the training data**: Opposit to overfitting. Need to use more complex models with more parameters and loose constraints.





