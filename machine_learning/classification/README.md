# Classification

Here are some notes that I jot down from [Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do). 



|           |          | Prediction | Prediction |
| :-------: | :------: | :--------: | :--------: |
|           |          |  Positive  |  Negative  |
| **Truth** | Positive |     TP     |     FN     |
| **Truth** | Negative |     FP     |     TN     |

**Accuracy**: $Accuracy=\cfrac{TP+TN}{TP+FN+FP+TN}$

**Precision**: $Precision=\cfrac{TP}{TP+FP}$

We look at the <u>first  column</u> of the confusion matrix for precision.

**Recall** (True positive rate): $Recall=\cfrac{TP}{TP+FN}$

We look at the <u>first row</u> of the confusion matrix for recall.

**Specificity** (True negative rate): $Specificity=\cfrac{TN}{FP+TN}$

We look at the second row of the confusion matrix for specificity.

**FPR**: $FPR=1-Specificity=\cfrac{FP}{FP+TN}$

