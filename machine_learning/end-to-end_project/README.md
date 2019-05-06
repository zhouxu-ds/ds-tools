# End-to-End Project

Here are some notes that I jot down from [Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do). 

[Useful end-to-end project checklist](project_checklist.md)

## Table of Content

## Find Data

### Popular open data repositories

[UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php)

[Kaggle Datasets](https://www.kaggle.com/datasets)

[Amazon's AWS Dataset](https://registry.opendata.aws)

### Meta portals (list of data repositories)

http://dataportals.org

http://opendatamonitor.eu/ 

http://quandl.com/ 

## Look at big picture

- See the objective
- How benefit from it
- Current solution
- Think about models
- Choose performance measure
  - Ways to measure the distance - $l$ norms
  - $l_k$ norm: $||v||_k=(|v_0|^k+|v_1|^k+\dots+|v_n|^k)^{\frac{1}{k}}$
  - eg, $l_0$ norm (just give the number of non-zero elements in the vector), $l_1$ norm (mean absolute error(MAE or Manhattan norm) ), $l_2$ norm (root of mean sum square(RMSE), $l_\infty$ norm (gives the max absolute value))
  - **The higher the norm index, the more it focuses on large values and neglects small ones**. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.

- Check assumptions

## Get the data

Can right similar things to fetch the data:

```python
# Make the function to fetech the data automatically
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```



