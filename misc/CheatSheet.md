# Cheat Sheet

## Stats

### Central Limit Theorem

https://en.wikipedia.org/wiki/Central_limit_theorem#Classical_CLT

In some situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution even if the original variables themselves are not normally distributed.

The sum of the samples will be in normal distribution with mean $\mu'$ and standard deviation $\sigma'$:
$$
\mu'=n\times mu \\
\sigma'= \sqrt{n}\times \sigma
$$

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

