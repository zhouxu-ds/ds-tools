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

## Programming

### Git Undo Changes

Reference: https://stackoverflow.com/a/14075772/12985675

Here are several ways to rollback to a certain state:

`git reset` This will unstage all files that are staged with `git add`

`git reset --hard` This will revert all uncommitted changes (can work from any subdirectory)

`git checkout .` This will revert all uncommited changes (Only at repo root, or replace `.` with particular files or directories)
