# Imbalance Metrics
This work presents a study for a set of performance evaluation metrics, in which the bias that occurs in multiclass metrics and the information obtained through the binary confusion matrix for an imbalanced data set will be analyzed. For the following study, the Matthews correlation coefficient (MCC) is considered a robust metric, as MCC exhibits the lowest bias in cases of data imbalance (Luque, 2019).

[User's guide](https://github.com/andyeli/Metrics_functions/blob/main/User_guide.md)

>**imbalance_metrics.report(_y_true, y_pred_)**

Este es un repositorio que ofrece un conjunto de métricas de evaluación de rendimiento para clasificación.

|I/O             | Description                                         |
|----------------| ----------------------------------------------------| 
|**Parameters**: |   **y_true**: **_1d array-like_**<br>Ground truth(correct) target value.<br>**y_pred**: **_1d array-like_**<br>Estimated targets as returned |
|**Returns**:        | x                                               |
| | |


**Examples**
```python
import imbalance_metrics

y_true = [0,1,0,0,1,1,1]
y_pred = [0,1,1,0,1,1,1]

results = im.report(y_true,y_pred)
```
```
----------------------------------- Score -----------------------------------
Class        PRC          NPV          ACC          F1           MCCn         MKn         
-----------------------------------------------------------------------------
1            1.0000       0.8000       0.8571       0.8000       0.8651       0.9000      
2            0.8000       1.0000       0.8571       0.8889       0.8651       0.9000      
Average      0.9000       0.9000       0.8571       0.8444       0.8651       0.9000      
---------------------- Bias ----------------------------------
Class        PRC          NPV          ACC          F1           MCCn         MKn         
--------------------------------------------------------------
1            0.0000       0.0500       0.0238       0.0000       0.0116       0.0250      
2            0.0500       0.0000       0.0238       0.0317       0.0116       0.0250      
Average      0.0250       0.0250       0.0238       0.0159       0.0116       0.0250      
------------------------- Coefficients -------------------------------
Class        LambdaPP     LambdaNN     Delta       
----------------------------------------------------------------------
1            0.6667       1.0000       -0.1429      
2            1.0000       0.6667       0.1429       
----- Multiclass metrics -----
OA       0.8571  
BA       0.8333  
BAW      0.8095  
AA       0.8571  
K        0.6957  
MCC      0.7303  
```