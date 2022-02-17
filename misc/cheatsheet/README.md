# Cheat Sheet

[toc]

## Stats

### Central Limit Theorem

https://en.wikipedia.org/wiki/Central_limit_theorem#Classical_CLT

In some situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution even if the original variables themselves are not normally distributed.

The sum of the samples will be in normal distribution with mean $\mu'$ and standard deviation $\sigma'$:
$$
\mu'=n\times mu \\
\sigma'= \sqrt{n}\times \sigma
$$



### Types of Correlations

Reference: https://datascience.stackexchange.com/a/64261

There are three types of common correlations:

- Pearson's Correlation Coefficient
- Kendall's Tau Coefficient
- Spearman's Rank Correlation Coefficient

**Pearson correlation vs Other two**:

- Pearson is parametric and the other two are non-parametric
- Non-parametric correlations are less powerful because they use less information
- Non-parametric correlations can be used for not only for continuous data, but also for ordinal data. Also, the normal distribution approximation assumption is not required.

**Kendall correlation vs Spearman correlation**:

- Kendall is more robust and efficient than Spearman when there are smaller samples or some outliers.
- Spearman has lower time complexity ($O(n\cdot log(n)$ vs $O(n)$), so will be better for larger sample size.

**Conclusion**:

- Use **Pearson** when data are continuous and approximately normal
- Use **Kendall** when there are ordinal data or normal distribution does not apply

## Machine Learning

### Machine Learning Pipelines

1. Understand business problems
2. Data collection
3. Exploratory data analysis
4. Data cleaning
5. Data transformation
6. Data segregation
7. Modeling 
8. Model evaluation

### Difference between bagging, boosting and stacking

https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205

**Bagging (Bootstrap Aggregation)**: Several instance of the same base model are trained in parallel on different bootstrap samples and then aggregated by voting (averaging). Aims for lowering variance, so suitable for low bias high variance base models.

**Boosting**: Several instance of the same base model are trained in sequence such that in each iteration, the way to train the current weak learner depends on the previous weak learner, especially on how they are performing on the data. Aims for lowering bias, so suitable for high bias, low variance base models.

**Stacking**: Different weak learners are fitted independantly from each other and a meta-model is trained on top to predict outputs from outputs returned by the base models.

### Why is XGBoost Good

https://www.kaggle.com/general/196541

XGBoost is good becuase of several reasons:

- hardware optimization
- parallelized tree building
- efficient handling of missing data
- tree pruning using ‘depth-first’ approach
- built-in cross-validation capability (at each iteration)
- regularization through both LASSO (L1) and Ridge (L2) for avoiding overfitting

### XGBoost vs LightGBM

The main difference between these frameworks is the way they are growing. XGBoost applies **level-wise** tree growth where LightGBM applies **leaf-wise** tree growth. Level-wise approach grows horizontal whereas leaf-wise grows vertical.

### What is Feature Store

https://www.phdata.io/blog/what-is-a-feature-store/

A feature store is a tool for storing commonly used features. When data scientists develop features for a [machine learning](https://www.phdata.io/blog/the-ultimate-guide-to-building-a-machine-learning-solution/) model, those features can be added to the feature store. This makes those features available for reuse. 

When new examples (e.g. users of an application, customers of a business, or items in a product catalog) are added, the previously developed features will be pre-computed so that the features are available for inference.

A full-fledged feature store:

- Transforms raw [data into feature values by executing data pipelines](https://www.phdata.io/blog/building-data-pipelines-with-aws-cloudformation/).
- Stores and manages feature values.
- Retrieves data for training or inference.

![feature_store](feature_store.png)



## Programming

### Git Undo Changes

Reference: https://stackoverflow.com/a/14075772/12985675

Here are several ways to rollback to a certain state:

`git reset` This will unstage all files that are staged with `git add`

`git reset --hard` This will revert all uncommitted changes (can work from any subdirectory)

`git checkout .` This will revert all uncommited changes (Only at repo root, or replace `.` with particular files or directories)
