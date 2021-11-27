# Class imbalance in classification performance metrics

The [imbalance_metrics](https://github.com/andyeli/Metrics_functions/blob/main/imbalance_metrics.py) is a function with contain of set of performance evaluation metrics, as can be seen in the next table.

|Function            | Description | Returns |
|----------------| ---------|-----|
|**confusion_matrix_to_vectors**(_Confusion_matrix_) |   Convert matrix confusion to vectors|y_true,y_pred |
|**sns**(_Confusion_matrix_) | Compute the sensitivity metric| {'Balanced': , 'Imbalanced': ,'Bias': }|
|**spc**(_Confusion_matrix_) | Computed specificity metric| {'Balanced': , 'Imbalanced': ,'Bias': } |
|**prc**(_Confusion_matrix_) | Compute the precision|{'Balanced': , 'Imbalanced': ,'Bias': } |
|**npv**(_Confusion_matrix_) |   Compute the Negative Predictive Value |{'Balanced': , 'Imbalanced': ,'Bias': } |
|**acc**(_Confusion_matrix_) | Compute Accuracy| {'Classic':<br>'Balanced':<br>'Imbalanced': <br>'Bias': <br>'Average_Acc': <br>'Average_Acc_Balanced': <br>'Overall_Acc': <br>'Balanced_Acc': <br>'Balanced_Acc_weighted': |
|**f1**(_Confusion_matrix_) |   Compute the F1 score |{'Balanced': , 'Imbalanced': ,'Bias': } |
|**gm**(_Confusion_matrix_) |   Compute the Geometric Mean |{'Balanced': , 'Imbalanced': ,'Bias': } |
|**mccn**(_Confusion_matrix_) |   Compute the Normalized versions of Matthews Correlation Coefficient  |{'Balanced': , 'Imbalanced': ,'Bias': } |
|**bmn**(_Confusion_matrix_) |   Compute the Bookmaker Informedness |{'Balanced': , 'Imbalanced': ,'Bias': } |
|**mkn**(_Confusion_matrix_) | Compute the Markedness|{'Balanced': , 'Imbalanced': ,'Bias': } |
| | | |

>## Mathematical definition
Given a multi-class confusion matrix ![](https://latex.codecogs.com/svg.latex?\mathbf{M}\in\mathbb{R}^{(C&space;\times&space;C)}),  obtained from the true labels ![](https://latex.codecogs.com/svg.latex?\mathbf{Y})  and the prediction  ![](https://latex.codecogs.com/svg.latex?\mathbf{\hat{Y}}), where ![](https://latex.codecogs.com/svg.latex?C) denotes the number of classes of interest, and the element ![](https://latex.codecogs.com/svg.latex?\mathbf{m}_{ij}\in\mathbf{M}) is the number of samples that belong to class i-th, but that are classified as members of class j-th, i.e., ![](https://latex.codecogs.com/svg.latex?y_{n}=i) and ![](https://latex.codecogs.com/svg.latex?\hat{y}_{n}=j), for ![](https://latex.codecogs.com/svg.latex?n=1,\dots,N). 

In this work, we use the mathematical definition of imbalance given by (Luque, 2019). The imbalance coefficient ![](https://latex.codecogs.com/svg.latex?\delta_{c}) , for a given class ![](https://latex.codecogs.com/svg.latex?c) , is featured with a value in the ![](https://latex.codecogs.com/svg.latex?[-1,1]) range, where ![](https://latex.codecogs.com/svg.latex?0) means that classes are perfectly balanced. The ![](https://latex.codecogs.com/svg.latex?c)-class imbalance coefficient can be computed by

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\delta_{c}=2&space;\frac{\sum_{j=1}^{C}&space;m_{c&space;j}}{\sum_{i=1}^{C}&space;\sum_{j=1}^{C}&space;m_{i&space;j}}-1" />
</p>

A performance evaluation metric ![](https://latex.codecogs.com/svg.latex?\mu) is a function that assigns to each confusion matrix, a real value on the set ![](https://latex.codecogs.com/svg.latex?\mathbb&space;{R}) i.e.,

<p align="center">
<img src=https://latex.codecogs.com/svg.latex?\mu:&space;Y&space;\times&space;\hat{Y}&space;\rightarrow&space;\mathbb{R} />
</p>

In general, the result of this function is in the range ![](https://latex.codecogs.com/svg.latex?[0,1]), where ![](https://latex.codecogs.com/svg.latex?0) means the perfect misclassification, and ![](https://latex.codecogs.com/svg.latex?1) the perfect classification.

>## Metrics for multi-class classification
### **Overall Accuracy**
The Overall Accuracy ![](https://latex.codecogs.com/svg.latex?\mathrm{OA}) is the ratio between the number of correctly classified elements and the overall number of samples, and it is computed from multi-class confusion matrix by 

<p align="center">
<img src= https://latex.codecogs.com/svg.latex?\mathrm{OA}=\frac{\sum_{i=1}^{C}&space;m_{i&space;i}}{\sum_{i=1}^{C}&space;\sum_{j=1}^{C}&space;m_{i&space;j}}&space;. />
</p>

This metric is one of the most used for classification performance evaluation. Nevertheless, if the dataset is imbalanced, the ![](https://latex.codecogs.com/svg.latex?\mathrm{OA}) is not a reliable measure, as it produces optimistic results (Chicco, 2020).

### **Balanced Accuracy**
An alternative metric to reduce the impact of imbalanced classes in performance evaluation is Balanced Accuracy ![](https://latex.codecogs.com/svg.latex?\mathrm{BA}), computed by the average of another well-known metric, the Sensitivity (SNS) per class, i.e.,

<p align="center">
<img src=https://latex.codecogs.com/svg.latex?\mathrm{BA}=\frac{\sum_{c=1}^{C}&space;\mathrm{SNS}_{c}}{c}, />
</p>

where ![](https://latex.codecogs.com/svg.latex?\mathrm{SNS}_{C}=\frac{m_{c&space;c}}{\sum_{j=1}^{C}&space;m_{c&space;j}}) . ![](https://latex.codecogs.com/svg.latex?\mathrm{SNS}) measures the proportion of the number of elements correctly classified from an individual class.


### **Balanced Accuracy Weighted**
The Balanced Accuracy Weighted ![](https://latex.codecogs.com/svg.latex?\mathrm{(BAW)}) is a metric used for imbalanced classes, in which the ![](https://latex.codecogs.com/svg.latex?\mathrm{SNS}_{C}) of each class is weighted by its relative frequency ![](https://latex.codecogs.com/svg.latex?w_{c}). The formula for ![](https://latex.codecogs.com/svg.latex?\mathrm{BAW}) is

<p align="center">
<img src=https://latex.codecogs.com/svg.latex?\mathrm{BAW}=\frac{\sum_{i=1}^{C}&space;\mathrm{SNSw}_{C}}{\sum_{c=1}^{C}&space;w_{c}}&space;, />
</p>


where ![](https://latex.codecogs.com/svg.latex?\mathrm{SNSw}_{C}=\frac{m_{cc}}{\sum_{j=1}^{C}&space;m_{c&space;j}}&space;w_{c}) and ![](https://latex.codecogs.com/svg.latex?w_{c}=\frac{\sum_{i=1}^{C}&space;\sum_{j=1}^{C}&space;m_{i&space;j}}{C&space;\sum_{j=1}^{C}&space;m_{c&space;j}}) (P, 2021). This metric is an efficient performance indicator, since the recalls are weighted by a relative frequency to the classes size (P, 2021, Grandini, 2020).

### **Cohen’s Kappa Coefficient**
The Kappa Coefficient ![](https://latex.codecogs.com/svg.latex?\mathrm{(K)}) is currently one of the most popular metrics in machine learning for classification performance evaluation (Chicco, 2020). This metric measures the inter-rater concordance, as the degree of agreement among raters. It is computed by

<p align="center">
<img src=https://latex.codecogs.com/svg.latex?\mathrm{K}=\frac{q&space;s-\sum_{c=1}^{C}&space;p_{c}&space;t_{c}}{s^{2}-\sum_{c=1}^{C}&space;p_{c}&space;t_{c}}, />
</p>

where ![](https://latex.codecogs.com/svg.latex?q=\sum_{c=1}^{C}&space;m_{c&space;c}) denotes the overall number of elements correctly predicted, ![](https://latex.codecogs.com/svg.latex?s=\sum_{i=1}^{C}&space;\sum_{j=1}^{C}&space;m_{i&space;j}) is the total number of samples, ![](https://latex.codecogs.com/svg.latex?p_{c}=\sum_{j=1}^{C}&space;m_{c&space;j}) the number of times class ![](https://latex.codecogs.com/svg.latex?c) was predicted, and ![](https://latex.codecogs.com/svg.latex?t_{c}=\sum_{i=1}^{C}&space;m_{i&space;c}) the number of times class ![](https://latex.codecogs.com/svg.latex?c) truly occurs (Grandini, 2020). Kappa coefficient is high sensitive to the marginal totals (Chicco, 2020).

### **Matthews Correlation Coefficient**
The Matthews Correlation Coefficient (![](https://latex.codecogs.com/svg.latex?\mathrm{MCC})) is generally considered a balanced performance evaluation metric. All the elements of the confusion matrix are included in the numerator and denominator of its formula, so this metric is less biased by imbalanced datasets than other metrics (Chicco, 2020). 

<p align="center">
<img src=https://latex.codecogs.com/svg.latex?\mathrm{MCC}=\frac{q&space;s-\sum_{c=1}^{C}&space;p_{c}&space;t_{c}}{\sqrt{\left(s^{2}-\sum_{c=1}^{C}&space;p_{c}^{2}\right)\left(s^{2}-\sum_{c=1}^{C}&space;t_{c}^{2}\right)}}  />
</p>

The main disadvantage is that ![](https://latex.codecogs.com/svg.latex?\mathrm{MCC}) is undefined for extreme cases, for instance, when a whole row or column of the confusion matrix is zero. Generally MCC and Kappa are used in their normalized version to be in the range ![](https://latex.codecogs.com/svg.latex?[0,1]), computed by ![](https://latex.codecogs.com/svg.latex?\mathrm{MCCn}=\frac{\mathrm{MCC}&plus;1}{2}) and ![](https://latex.codecogs.com/svg.latex?\mathrm{Kn}=\frac{\mathrm{K}&plus;1}{2}) respectively.

>## Metrics based on the binary confusion matrix
For the particular case of binary classification, the elements of the confusion matrix are defined as the True Positives ![](https://latex.codecogs.com/svg.latex?\mathrm{(TP)}), is the case where the prediction is true and the actual output is also true, ![](https://latex.codecogs.com/svg.latex?\mathrm{TP}=m_{11}), the False Negatives ![](https://latex.codecogs.com/svg.latex?\mathrm{(FN)}), is the case where the prediction is false and the actual output is true, ![](https://latex.codecogs.com/svg.latex?\mathrm{FN}=m_{12}), the False Positives ![](https://latex.codecogs.com/svg.latex?\mathrm{(FP)}), is when the prediction is true and the actual output is false, ![](https://latex.codecogs.com/svg.latex?\mathrm{FP}=m_{21}), and the True Negatives ![](https://latex.codecogs.com/svg.latex?\mathrm{(TN)}), is the cases where the prediction is false and the actual output is false., ![](https://latex.codecogs.com/svg.latex?\mathrm{TN}=m_{22}).  


In the work of Luque (Luque, 2019), a classical metric based on the binary confusion matrix ![](https://latex.codecogs.com/svg.latex?\mu\mathrm{(TP},&space;\mathrm{TN},&space;\mathrm{FP},&space;\mathrm{FN)}) is expressed as a function ![](https://latex.codecogs.com/svg.latex?\mu\left(\lambda_{P&space;P},&space;\lambda_{N&space;N},&space;\delta\right)), where ![](https://latex.codecogs.com/svg.latex?\lambda_{P&space;P}) represents the ratio of the correctly classified positive elements ![](https://latex.codecogs.com/svg.latex?m_{P&space;P}=$&space;$\mathrm{TP}) and the total of truly positive elements ![](https://latex.codecogs.com/svg.latex?m_{P}=$&space;$\mathrm{TP}&plus;\mathrm{FN}), i.e., ![](https://latex.codecogs.com/svg.latex?\lambda_{P&space;P}=\frac{m_{P&space;P}}{m_{P}}), and ![](https://latex.codecogs.com/svg.latex?\lambda_{N&space;N}) denotes the ratio of the correctly classified negative ![](https://latex.codecogs.com/svg.latex?m_{N&space;N}=&space;\mathrm{FP}) and the total of truly negative elements ![](https://latex.codecogs.com/svg.latex?m_{N}=\mathrm{FP}&plus;\mathrm{TN}), i.e., ![](https://latex.codecogs.com/svg.latex?\lambda_{N&space;N}=\frac{m_{N&space;N}}{m_{N}}). Considering the balanced case, ![](https://latex.codecogs.com/svg.latex?\mu_{b}=\mu(\lambda_{P&space;P},&space;\lambda_{N&space;N},&space;0) ) it is possible to define the impact of imbalance by the bias of a metric ![](https://latex.codecogs.com/svg.latex?B_{\mu}) as  

<p align="center">
<img src=https://latex.codecogs.com/svg.latex?B_{\mu}=\mu-\mu_{b} />
</p>


**Table 1** Classification performance metrics as classical metric. (Luque, 2019)
|Metrics             | ![](https://latex.codecogs.com/svg.latex?\mu&space;\mathrm{(TP},&space;\mathrm{TN},&space;\mathrm{FP},&space;\mathrm{FN)})  |
|----------------| ----------------------------------------------------| 
|**PRC(PPV)**: |   ![](https://latex.codecogs.com/svg.latex?\frac{\mathrm{TP}}{\mathrm{TP}&plus;\mathrm{FP}})|
|**NPV**: |  ![](https://latex.codecogs.com/svg.latex?\frac{\mathrm{TN}}{\mathrm{TN}&plus;\mathrm{FN}})|
|**ACC**: | ![](https://latex.codecogs.com/svg.latex?\frac{\mathrm{TP}&plus;\mathrm{TN}}{\mathrm{TP}&plus;\mathrm{FN}&plus;\mathrm{TN}&plus;\mathrm{FP}})|
|**F1**: |  ![](https://latex.codecogs.com/svg.latex?2&space;\frac{\mathrm{PRC}&space;\cdot&space;\frac{\mathrm{TP}}{\mathrm{TP}&plus;\mathrm{FN}}}{\mathrm{PRC}&plus;\frac{\mathrm{TP}}{\mathrm{TP}&plus;\mathrm{FN}}})|
|**GM**: |   ![](https://latex.codecogs.com/svg.latex?\sqrt{\frac{\mathrm{TP}}{\mathrm{TP}&plus;\mathrm{FN}}&space;\cdot&space;\frac{\mathrm{TN}}{\mathrm{TN}&plus;\mathrm{FP}}})|
|**MCCn**: |  ![](https://latex.codecogs.com/svg.latex?\frac{\mathrm{TP}&space;\cdot&space;\mathrm{TN}-\mathrm{FP}&space;\cdot&space;\mathrm{FN}}{\sqrt{(\mathrm{TP}&plus;\mathrm{FP})(\mathrm{TP}&plus;\mathrm{FN})(\mathrm{TN}&plus;\mathrm{FP})(\mathrm{TN}&plus;\mathrm{FN})}})|
|**MKn**: | ![](https://latex.codecogs.com/svg.latex?\mathrm{PPV}&plus;\mathrm{NPV}-1)|
| | |

**Table 2** Classification performance metrics as a function of imbalance. (Luque, 2019)
|Metrics             | ![](https://latex.codecogs.com/svg.latex?\boldsymbol{\mu}(\lambda_{P&space;P},&space;\lambda_{N&space;N},&space;\boldsymbol{\delta)})|
|----------------| ----------------------------------------------------| 
|**PRC(PPV)**: |   ![](https://latex.codecogs.com/svg.latex?\frac{\lambda_{P&space;P}(1&plus;\delta)}{\lambda_{P&space;P}(1&plus;\delta)&plus;\left(1-\lambda_{N&space;N}\right)(1-\delta)})|
|**NPV**: |   ![](https://latex.codecogs.com/svg.latex?\frac{\lambda_{N&space;N}(1-\delta)}{\lambda_{N&space;N}(1-\delta)&plus;\left(1-\lambda_{P&space;P}\right)(1&plus;\delta)})|
|**ACC**: | ![](https://latex.codecogs.com/svg.latex?\lambda_{P&space;P}&space;\frac{1&plus;\delta}{2}&plus;\lambda_{N&space;N}&space;\frac{1-\delta}{2})|
|**F1**: | ![](https://latex.codecogs.com/svg.latex?\frac{2&space;\lambda_{P&space;P}(1&plus;\delta)}{\left(1&plus;\lambda_{P&space;P}\right)(1&plus;\delta)&plus;\left(1-\lambda_{N&space;N}\right)(1-\delta)})|
|**GM**: | ![](https://latex.codecogs.com/svg.latex?\sqrt{\lambda_{P&space;P}&space;\cdot&space;\lambda_{N&space;N}})|
|**MCCn**: |  ![](https://latex.codecogs.com/svg.latex?\frac{1}{2}\left(\frac{\lambda_{p&space;p}&plus;\lambda_{N&space;N}-1}{\sqrt{\left[\lambda_{p&space;p}&plus;\left(1-\lambda_{N&space;N}\right)&space;\frac{1-\delta}{1&plus;\delta}\right]\left[\lambda_{N&space;N}&plus;\left(1-\lambda_{p&space;P}\right)&space;\frac{1&plus;\delta}{1-\delta}\right]}}&plus;1\right))|
|**MKn**: | ![](https://latex.codecogs.com/svg.latex?\frac{1}{2}\left(\frac{1&plus;\delta}{(1&plus;\delta)&plus;\frac{1-\lambda_{N&space;N}}{\lambda_{P&space;P}}(1-\delta)}&plus;\frac{1-\delta}{(1-\delta)&plus;\frac{1-\lambda_{P&space;P}}{\lambda_{N&space;N}}(1&plus;\delta)}\right))|
| | |

**Table 3** Classification performance metrics as a function of balance. (Luque, 2019)
|Metrics             | ![](https://latex.codecogs.com/svg.latex?\boldsymbol{\mu}(\lambda_{P&space;P},&space;\lambda_{N&space;N)}) |
|----------------| ----------------------------------------------------| 
|**PRC(PPV)**: | ![](https://latex.codecogs.com/svg.latex?\frac{\lambda_{P&space;P}}{\lambda_{P&space;P}&plus;(1-\lambda_{N&space;N)}})|
|**NPV**: | ![](https://latex.codecogs.com/svg.latex?\frac{\lambda_{N&space;N}}{\lambda_{N&space;N}&plus;(1-\lambda_{P&space;P})})|
|**ACC**: | ![](https://latex.codecogs.com/svg.latex?\frac{\lambda_{P&space;P}&plus;\lambda_{N&space;N}}{2})|
|**F1**: | ![](https://latex.codecogs.com/svg.latex?\frac{2&space;\lambda_{P&space;P}}{2&plus;\lambda_{P&space;P}-\lambda_{N&space;N}})|
|**GM**: |  ![](https://latex.codecogs.com/svg.latex?\sqrt{\lambda_{P&space;P}&space;\cdot&space;\lambda_{N&space;N}})|
|**MCCn**: | ![](https://latex.codecogs.com/svg.latex?\frac{1}{2}\left(\frac{\lambda_{P&space;P}&plus;\lambda_{N&space;N}-1}{\sqrt{\left[\lambda_{P&space;P}&plus;\left(1-\lambda_{N&space;N}\right)\right]\left[\lambda_{N&space;N}&plus;\left(1-\lambda_{P&space;P}\right)\right]}}&plus;1\right))|
|**MKn**: | ![](https://latex.codecogs.com/svg.latex?\frac{1}{2}\left(\frac{1}{1&plus;\frac{1-\lambda_{N&space;N}}{\lambda_{P&space;P}}}&plus;\frac{1}{1&plus;\frac{1-\lambda_{P&space;P}}{\lambda_{N&space;N}}}\right))|
| | |

We lead this approach to the multi-class case, starting from the multi-class confusion matrix, generating multiple binary confusion matrices ![](https://latex.codecogs.com/svg.latex?\mathbf{M}_{c}) using the equations 9a to 9d. It is important to highlight that the results given by the transformation of the multi-class confusion matrix to binary confusion matrices are not necessarily equal to the results when using multiple binary classifiers.

<p align="center">
<img src=https://latex.codecogs.com/svg.latex?\begin{aligned}&space;&\mathrm{TP}_{c}&space;\cong&space;m_{c&space;c}&space;\\&space;&\mathrm{FP}_{c}&space;\cong&space;\sum_{j=1}^{C}&space;m_{c&space;j}&space;\text&space;{&space;for&space;}&space;j&space;\neq&space;c&space;\\&space;&\mathrm{FN}_{c}&space;\cong&space;\sum_{i=1}^{C}&space;m_{i&space;c}&space;\text&space;{&space;for&space;}&space;i&space;\neq&space;c&space;\\&space;&\mathrm{TN}_{c}&space;\cong&space;\sum_{i=1}^{C}&space;\sum_{j=1}^{C}&space;m_{i&space;j}&space;\text&space;{&space;for&space;}&space;i&space;\neq&space;c&space;\text&space;{&space;and&space;}&space;j&space;\neq&space;c&space;\end{aligned} />
</p>


Hence, the equations for each metric presented in Tables 2 y 3 can be used to evaluate classification performance given a class of reference.  In this analysis, only one of the four metrics that do not present bias, studied in (Luque, 2019), was considered for comparison purposes.  In addition, an overall performance can be given by the average of the metric for each class.

The classification metrics studied under this approach are: Precision (**PRC**), Negative Predictive Value (**NPV**), Accuracy (**ACC**), F1  score (**F1**) , Geometric Mean (**GM**), the Normalized versions of Matthews Correlation Coefficient (**MCCn**), and Markedness (**MKn**).<br><br><br>



>**REFERENCES** <br> [Luque, A. & Carrasco, Alejandro & Martín, Alejandro & de Las Heras, Ana. (2019). The impact of class imbalance in classification performance metrics based on the binary confusion matrix. Pattern Recognition. 91. 6829. 10.1016/j.patcog.2019.02.023.](https://www.researchgate.net/publication/331402961_The_impact_of_class_imbalance_in_classification_performance_metrics_based_on_the_binary_confusion_matrix) <br><br>
[Chicco, Davide & Jurman, Giuseppe. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics. 21. 10.1186/s12864-019-6413-7.](https://www.researchgate.net/publication/338351315_The_advantages_of_the_Matthews_correlation_coefficient_MCC_over_F1_score_and_accuracy_in_binary_classification_evaluation)<br><br>
[Grandini, Margherita & Bagli, Enrico & Visani, Giorgio. (2020). Metrics for Multi-Class Classification: an Overview.](https://www.researchgate.net/publication/343649058_Metrics_for_Multi-Class_Classification_an_Overview)<br><br>
[P. (2021, 6 enero). How to Improve Class Imbalance using Class Weights in Machine Learning. Analytics Vidhya.]( https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/)
<br><br>
[Boughorbel, S. & Jarray, Fethi & El-Anbari, Mohammed. (2017). Optimal classifier for imbalanced data using Matthews Correlation Coefficient metric. PLOS ONE. 12. e0177678. 10.1371/journal.pone.0177678.](https://www.researchgate.net/publication/317321000_Optimal_classifier_for_imbalanced_data_using_Matthews_Correlation_Coefficient_metric) <br><br>
[Krawczyk, B. (2016). Learning from imbalanced data: open challenges and future directions. Progress in Artificial Intelligence, 5, 221-232.](https://www.semanticscholar.org/paper/Learning-from-imbalanced-data%3A-open-challenges-and-Krawczyk/f537f1bc527bf33cc5fd8da34275106329de1802) <br><br>
[Chawla, Nitesh. (2005). Data Mining for Imbalanced Datasets: An Overview. 10.1007/0-387-25465-X_40.](https://www.researchgate.net/publication/226755026_Data_Mining_for_Imbalanced_Datasets_An_Overview) <br><br>