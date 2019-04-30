
# Loading the data



```python
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors, svm
import warnings
warnings.filterwarnings('ignore')
```

### 7. Attribute Information: (class attribute has been moved to last column)

   ###  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)

### 8. Missing attribute values: 16

   There are 16 instances in Groups 1 to 6 that contain a single missing 
   (i.e., unavailable) attribute value, now denoted by "?".  

### 9. Class distribution:
 
   Benign: 458 (65.5%)
   Malignant: 241 (34.5%)


```python
df = pd.read_csv('data/breast-cancer-wisconsin.data')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>clump_thickness</th>
      <th>unif_cell_size</th>
      <th>unif_cell_shape</th>
      <th>marg_adhesion</th>
      <th>single_ephith_cell_size</th>
      <th>bare_nuclei</th>
      <th>bland_chrom</th>
      <th>norm_nucleoli</th>
      <th>mitoses</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1016277</td>
      <td>6</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1017023</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# Cleaning the data


```python
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
```


```python
x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
```

# Spliting the Train data and Test data


```python
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
```


```python
x_train
```




    array([[1, 1, 1, ..., 2, 1, 1],
           [8, 10, 10, ..., 7, 8, 1],
           [5, 1, 2, ..., 3, 1, 1],
           ...,
           [4, 1, 1, ..., 2, 1, 1],
           [10, 10, 10, ..., 7, 10, 1],
           [4, 10, 4, ..., 9, 10, 1]], dtype=object)




```python
y_train
```




    array([2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 4, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2,
           4, 2, 2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2, 4, 4, 2, 4, 4, 2, 2, 2, 2,
           2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2,
           4, 2, 2, 2, 2, 4, 2, 4, 4, 4, 4, 4, 4, 4, 2, 2, 4, 2, 2, 2, 2, 2,
           4, 2, 2, 2, 4, 2, 4, 2, 4, 4, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4,
           2, 2, 2, 2, 2, 2, 4, 2, 4, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2,
           2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 4, 4, 2, 2,
           4, 4, 4, 2, 2, 2, 2, 4, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 4, 4, 2, 2,
           2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 4, 4, 4, 4, 2, 2, 4,
           2, 4, 2, 4, 2, 4, 2, 2, 4, 2, 4, 2, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2,
           4, 4, 4, 4, 4, 2, 2, 4, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2,
           4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2,
           2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 2, 4, 4, 4, 2, 4, 2,
           2, 2, 4, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 4, 4, 4,
           2, 4, 2, 4, 2, 2, 4, 2, 2, 2, 2, 4, 2, 4, 4, 2, 2, 4, 4, 2, 4, 2,
           2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 4, 2, 2, 2, 4, 4, 2, 4, 4, 2, 2, 4, 4, 2, 2, 4, 2, 4, 2, 2,
           2, 4, 4, 4, 4, 4, 2, 2, 4, 2, 2, 4, 4, 4, 2, 2, 2, 4, 4, 2, 2, 2,
           4, 4, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 4, 4, 2, 2, 4,
           2, 4, 2, 2, 2, 4, 4, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 4, 4,
           2, 4, 2, 2, 4, 2, 4, 2, 2, 4, 4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2,
           2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 4, 2, 2, 2,
           2, 4, 2, 4, 2, 4, 2, 4, 4])



# Creating and training the classifier


```python
clf = svm.SVC()
clf.fit(x_train, y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)



# Checking model accuracy


```python
accuracy = clf.score(x_test, y_test)
accuracy
```




    0.9714285714285714



# Tests


```python
# Generating test data
```


```python
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)
```


```python
tipos_de_cancer = {4 : "maligo", 2 : "benigno"}
```


```python
prediction = clf.predict(example_measures)
print('Tipo de cancer é: {}'.format(tipos_de_cancer[prediction.item(0)]))
```

    Tipo de cancer é: benigno

