# Class imbalance in classification performance metrics

The [imbalance_metrics](https://github.com/andyeli/Metrics_functions/blob/main/imbalance_metrics.py) is a function with contain of set of performance evaluation metrics, as can be seen in the next table.

|Function            | Description | Returns |
|----------------| ---------|-----|
|**confusion_matrix_to_vectors**(_Confusion_matrix_) |   Convert matrix confusion to vectors|y_true,y_pred |
|**sns**(_Confusion_matrix_) | Compute the sensitivity metric| {'Balanced': , 'Imbalanced': ,<br>'Bias': }|
|**spc**(_Confusion_matrix_) | Computed specificity metric| {'Balanced': , 'Imbalanced': ,<br>'Bias': } |
|**prc**(_Confusion_matrix_) | Compute the precision|{'Balanced': , 'Imbalanced': ,<br>'Bias': } |
|**npv**(_Confusion_matrix_) |   Compute the Negative Predictive Value |{'Balanced': , 'Imbalanced': ,<br>'Bias': } |
|**acc**(_Confusion_matrix_) | Compute Accuracy| {'Classic':<br>'Balanced':<br>'Imbalanced': <br>'Bias': <br>'Average_Acc': <br>'Average_Acc_Balanced': <br>'Overall_Acc': <br>'Balanced_Acc': <br>'Balanced_Acc_weighted': |
|**f1**(_Confusion_matrix_) |   Compute the F1 score |{'Balanced': , 'Imbalanced': ,<br>'Bias': } |
|**gm**(_Confusion_matrix_) |   Compute the Geometric Mean |{'Balanced': , 'Imbalanced': ,<br>'Bias': } |
|**mccn**(_Confusion_matrix_) |   Compute the Normalized versions of Matthews Correlation Coefficient  |{'Balanced': , 'Imbalanced': ,<br>'Bias': } |
|**bmn**(_Confusion_matrix_) |   Compute the Bookmaker Informedness |{'Balanced': , 'Imbalanced': ,<br>'Bias': } |
|**mkn**(_Confusion_matrix_) | Compute the Markedness|{'Balanced': , 'Imbalanced': ,<br>'Bias': } |
| | | |

>## Mathematical definition
Given a multi-class confusion matrix $\mathbf{M}∈\mathbb{R}^{(C×C)}$,  obtained from the true labels $\mathbf{Y}$  and the prediction  $\mathbf{\hat{Y}}$, where $C$ denotes the number of classes of interest, and the element $\mathbf{m}_{ij}∈\mathbf{M}$ is the number of samples that belong to class i-th, but that are classified as members of class j-th, i.e., $y_{n}=i$ and $\hat{y}_{n}=j$, for $n=1,…,N$. 

In this work, we use the mathematical definition of imbalance given by (Luque, 2019). The imbalance coefficient $\delta_{c}$ , for a given class $c$ , is featured with a value in the $[-1,1]$ range, where $0$ means that classes are perfectly balanced. The $c$-class imbalance coefficient can be computed by

$$\delta_{c}=2 \frac{\sum_{j=1}^{C} m_{c j}}{\sum_{i=1}^{C} \sum_{j=1}^{C} m_{i j}}-1 .$$

A performance evaluation metric $\mu$ is a function that assigns to each confusion matrix, a real value on the set $\mathbb {R}$ i.e.,

$$\mu: Y \times \hat{Y} \rightarrow \mathbb{R}$$

In general, the result of this function is in the range $[0,1]$, where $0$ means the perfect misclassification, and $1$ the perfect classification.

>## Metrics for multi-class classification
### **Overall Accuracy**
The Overall Accuracy $\mathrm{(OA)}$ is the ratio between the number of correctly classified elements and the overall number of samples, and it is computed from multi-class confusion matrix by 

$$ \mathrm{OA}=\frac{\sum_{i=1}^{C} m_{i i}}{\sum_{i=1}^{C} \sum_{j=1}^{C} m_{i j}} .$$

This metric is one of the most used for classification performance evaluation. Nevertheless, if the dataset is imbalanced, the $\mathrm{OA}$ is not a reliable measure, as it produces optimistic results (Chicco, 2020).

### **Balanced Accuracy**
An alternative metric to reduce the impact of imbalanced classes in performance evaluation is Balanced Accuracy $\mathrm{(BA)}$, computed by the average of another well-known metric, the Sensitivity (SNS) per class, i.e.,

$$\mathrm{BA}=\frac{\sum_{c=1}^{C} \mathrm{SNS}_{c}}{c},$$

where $\mathrm{SNS}_{C}=\frac{m_{c c}}{\sum_{j=1}^{C} m_{c j}}$ . $\mathrm{SNS}$ measures the proportion of the number of elements correctly classified from an individual class.


### **Balanced Accuracy Weighted**
The Balanced Accuracy Weighted $\mathrm{(BAW)}$ is a metric used for imbalanced classes, in which the $\mathrm{SNS}_{C}$ of each class is weighted by its relative frequency $w_{c}$. The formula for $\mathrm{BAW}$ is

$$\mathrm{BAW}=\frac{\sum_{i=1}^{C} \mathrm{SNSw}_{C}}{\sum_{c=1}^{C} w_{c}} ,$$

where $\mathrm{SNSw}_{C}=\frac{m_{cc}}{\sum_{j=1}^{C} m_{c j}} w_{c}$ and $w_{c}=\frac{\sum_{i=1}^{C} \sum_{j=1}^{C} m_{i j}}{C \sum_{j=1}^{C} m_{c j}}$ (P, 2021). This metric is an efficient performance indicator, since the recalls are weighted by a relative frequency to the classes size (P, 2021, Grandini, 2020).

### **Cohen’s Kappa Coefficient**
The Kappa Coefficient $\mathrm{(K)}$ is currently one of the most popular metrics in machine learning for classification performance evaluation (Chicco, 2020). This metric measures the inter-rater concordance, as the degree of agreement among raters. It is computed by

$$\mathrm{K}=\frac{q s-\sum_{c=1}^{C} p_{c} t_{c}}{s^{2}-\sum_{c=1}^{C} p_{c} t_{c}},$$

where $q=\sum_{c=1}^{C} m_{c c}$ denotes the overall number of elements correctly predicted, $s=$ $\sum_{i=1}^{C} \sum_{j=1}^{C} m_{i j}$ is the total number of samples, $p_{c}=\sum_{j=1}^{C} m_{c j}$ the number of times class $c$ was predicted, and $t_{c}=\sum_{i=1}^{C} m_{i c}$ the number of times class $c$ truly occurs (Grandini, 2020). Kappa coefficient is high sensitive to the marginal totals (Chicco, 2020).

### **Matthews Correlation Coefficient**
The Matthews Correlation Coefficient (MCC) is generally considered a balanced performance evaluation metric. All the elements of the confusion matrix are included in the numerator and denominator of its formula, so this metric is less biased by imbalanced datasets than other metrics (Chicco, 2020). 

$$ \mathrm{MCC}=\frac{q s-\sum_{c=1}^{C} p_{c} t_{c}}{\sqrt{\left(s^{2}-\sum_{c=1}^{C} p_{c}^{2}\right)\left(s^{2}-\sum_{c=1}^{C} t_{c}^{2}\right)}} $$

The main disadvantage is that $\mathrm{MCC}$ is undefined for extreme cases, for instance, when a whole row or column of the confusion matrix is zero. Generally MCC and Kappa are used in their normalized version to be in the range $[0,1]$, computed by $\mathrm{MCCn}=\frac{\mathrm{MCC}+1}{2}$ and $\mathrm{Kn}=\frac{\mathrm{K}+1}{2}$ respectively.

>## Metrics based on the binary confusion matrix
For the particular case of binary classification, the elements of the confusion matrix are defined as the True Positives $\mathrm{(TP)}$, is the case where the prediction is true and the actual output is also true, $\mathrm{TP}=m_{11}$, the False Negatives $(\mathrm{FN})$, is the case where the prediction is false and the actual output is true, $\mathrm{FN}=m_{12}$, the False Positives $(\mathrm{FP})$, is when the prediction is true and the actual output is false, $\mathrm{FP}=m_{21}$, and the True Negatives $\mathrm{(TN)}$, is the cases where the prediction is false and the actual output is false., $\mathrm{TN}=m_{22}$.  


In the work of Luque (Luque, 2019), a classical metric based on the binary confusion matrix $\mu(\mathrm{TP}, \mathrm{TN}, \mathrm{FP}, \mathrm{FN})$ is expressed as a function $\mu\left(\lambda_{P P}, \lambda_{N N}, \delta\right)$, where $\lambda_{P P}$ represents the ratio of the correctly classified positive elements $m_{P P}=$ $\mathrm{TP}$ and the total of truly positive elements $m_{P}=$ $\mathrm{TP}+\mathrm{FN}$, i.e., $\lambda_{P P}=\frac{m_{P P}}{m_{P}}$, and $\lambda_{N N}$ denotes the ratio of the correctly classified negative $m_{N N}=$ FP and the total of truly negative elements $m_{N}=$ $\mathrm{FP}+\mathrm{TN}$, i.e., $\lambda_{N N}=\frac{m_{N N}}{m_{N}}$. Considering the balanced case, $\mu_{b}=\mu\left(\lambda_{P P}, \lambda_{N N}, 0\right)$ it is possible to define the impact of imbalance by the bias of a metric $B_{\mu}$ as  
$$
B_{\mu}=\mu-\mu_{b}
$$


**Table 1** Classification performance metrics as classical metric. (Luque, 2019)
|Metrics             | $\mu(\mathrm{TP}, \mathrm{TN}, \mathrm{FP}, \mathrm{FN})$                                         |
|----------------| ----------------------------------------------------| 
|**PRC(PPV)**: |   $\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$|
|**NPV**: |   $\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FN}}$|
|**ACC**: |   $\frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{FN}+\mathrm{TN}+\mathrm{FP}}$|
|**F1**: |   $2 \frac{\mathrm{PRC} \cdot \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}}{\mathrm{PRC}+\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}}$|
|**GM**: |   $\sqrt{\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}} \cdot \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}}$|
|**MCCn**: |  $\frac{\mathrm{TP} \cdot \mathrm{TN}-\mathrm{FP} \cdot \mathrm{FN}}{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}}$|
|**MKn**: |  $\mathrm{PPV}+\mathrm{NPV}-1$<br>|
| | |

**Table 2** Classification performance metrics as a function of imbalance. (Luque, 2019)
|Metrics             | $\boldsymbol{\mu}\left(\lambda_{P P}, \lambda_{N N}, \boldsymbol{\delta}\right)$                                         |
|----------------| ----------------------------------------------------| 
|**PRC(PPV)**: |   $\frac{\lambda_{P P}(1+\delta)}{\lambda_{P P}(1+\delta)+\left(1-\lambda_{N N}\right)(1-\delta)}$|
|**NPV**: |   $\frac{\lambda_{N N}(1-\delta)}{\lambda_{N N}(1-\delta)+\left(1-\lambda_{P P}\right)(1+\delta)}$|
|**ACC**: |   $\lambda_{P P} \frac{1+\delta}{2}+\lambda_{N N} \frac{1-\delta}{2}$|
|**F1**: |   $\frac{2 \lambda_{P P}(1+\delta)}{\left(1+\lambda_{P P}\right)(1+\delta)+\left(1-\lambda_{N N}\right)(1-\delta)}$|
|**GM**: |   $\sqrt{\lambda_{P P} \cdot \lambda_{N N}}$|
|**MCCn**: |  $\frac{1}{2}\left(\frac{\lambda_{p p}+\lambda_{N N}-1}{\sqrt{\left[\lambda_{p p}+\left(1-\lambda_{N N}\right) \frac{1-\delta}{1+\delta}\right]\left[\lambda_{N N}+\left(1-\lambda_{p P}\right) \frac{1+\delta}{1-\delta}\right]}}+1\right)$|
|**MKn**: |  $\frac{1}{2}\left(\frac{1+\delta}{(1+\delta)+\frac{1-\lambda_{N N}}{\lambda_{P P}}(1-\delta)}+\frac{1-\delta}{(1-\delta)+\frac{1-\lambda_{P P}}{\lambda_{N N}}(1+\delta)}\right)$<br>|
| | |

**Table 3** Classification performance metrics as a function of balance. (Luque, 2019)
|Metrics             | $\boldsymbol{\mu}\left(\lambda_{P P}, \lambda_{N N}\right)$                                         |
|----------------| ----------------------------------------------------| 
|**PRC(PPV)**: |   $\frac{\lambda_{P P}}{\lambda_{P P}+\left(1-\lambda_{N N}\right)}$|
|**NPV**: |   $\frac{\lambda_{N N}}{\lambda_{N N}+\left(1-\lambda_{P P}\right)}$|
|**ACC**: |   $\frac{\lambda_{P P}+\lambda_{N N}}{2}$|
|**F1**: | $\frac{2 \lambda_{P P}}{2+\lambda_{P P}-\lambda_{N N}}$|
|**GM**: |   $\sqrt{\lambda_{P P} \cdot \lambda_{N N}}$|
|**MCCn**: | $\frac{1}{2}\left(\frac{\lambda_{P P}+\lambda_{N N}-1}{\sqrt{\left[\lambda_{P P}+\left(1-\lambda_{N N}\right)\right]\left[\lambda_{N N}+\left(1-\lambda_{P P}\right)\right]}}+1\right)$|
|**MKn**: | $\frac{1}{2}\left(\frac{1}{1+\frac{1-\lambda_{N N}}{\lambda_{P P}}}+\frac{1}{1+\frac{1-\lambda_{P P}}{\lambda_{N N}}}\right)$<br>|
| | |

We lead this approach to the multi-class case, starting from the multi-class confusion matrix, generating multiple binary confusion matrices $\mathbf{M}_{c}$ using the equations 9a to 9d. It is important to highlight that the results given by the transformation of the multi-class confusion matrix to binary confusion matrices are not necessarily equal to the results when using multiple binary classifiers.

$
\begin{aligned}
&\mathrm{TP}_{c} \cong m_{c c} \\
&\mathrm{FP}_{c} \cong \sum_{j=1}^{C} m_{c j} \text { for } j \neq c \\
&\mathrm{FN}_{c} \cong \sum_{i=1}^{C} m_{i c} \text { for } i \neq c \\
&\mathrm{TN}_{c} \cong \sum_{i=1}^{C} \sum_{j=1}^{C} m_{i j} \text { for } i \neq c \text { and } j \neq c
\end{aligned}
$
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