# Credit Risk Analysis

An analysis using Machine Learning algorithms to identify credit card risk using a dataset from LendingClub.

# Overview

The purpose of this analysis is to understand how to utilize `Machine Learning` statistical algorithms to make predictions based on data patterns provided. In this challenge, we focus on **Supervised Learning** using a free dataset from **LendingClub**, a P2P lending service company to evaluate and predict credit risk. This reason why this is called **"Supervised Learning"** is because the data includes a labeled outcome. 

To complete this analysis, we use different `Machine Learning` techniques to train and evaluate the data with unbalanced classes. The dataset from the **LendingClub** has an unbalanced classification problem due to the number of good loans outweighing the amount of risky loans. In order balance out the classifications to allow for more meaningful predictions and improve the accuracy score, we needed to employ various `Machine Learning` algorithms to resample the data. These algorithms include `RandomOverSampler`, `SMOTE`, `ClusterCentroids`, `SMOTEENN`, `BalancedRandomForestClassifier`, and `EasyEnsembleClassifier`.

# Results

As mentioned in the overview, we use `Machine Learning` to resample the dataset using `Python` libraries: `scikit-learn` and `imbalanced-learn` evaluate the results and provide a comparison for our analysis. 

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk". 

<img width="263" alt="Screen Shot 2022-08-28 at 8 42 09 PM" src="https://user-images.githubusercontent.com/104115586/187115543-b33ef22f-9b14-4307-aec4-404bdd389063.png">


Using the 75/25% method to split the data for training vs. testing, 51,352 "low risk" and 260 "high risk" applications were categorized into the training set.   


<img width="734" alt="Screen Shot 2022-08-28 at 8 42 27 PM" src="https://user-images.githubusercontent.com/104115586/187115562-1a204a77-43bf-4113-b52a-6be2d66e8128.png">

## Deliverable 1: Use Resampling Models to Predict Credit Risk

### Oversampling

**`RandomOverSampler Model`** randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,352 records each as High Risk and Low Risk.

<img width="520" alt="Screen Shot 2022-08-28 at 8 43 09 PM" src="https://user-images.githubusercontent.com/104115586/187115572-b01231de-5d44-4e70-9a90-0ed16880dc2b.png">


  * Balanced accuracy score: 65%.

  <img width="440" alt="Screen Shot 2022-08-28 at 8 58 22 PM" src="https://user-images.githubusercontent.com/104115586/187115586-d937bd8c-3deb-4bbe-8ef2-a06782c35c54.png">


  * The "High Risk" precision rate was only 1% with the recall at 61% giving this model an F1 score of 2%.
  * "Low Risk" had a precision rate of 100% and recall at 69%.  
  
  <img width="370" alt="Screen Shot 2022-08-28 at 8 58 29 PM" src="https://user-images.githubusercontent.com/104115586/187115602-00707d14-965a-4d65-975a-e66111478e00.png">
  
  <img width="696" alt="Screen Shot 2022-08-28 at 8 58 40 PM" src="https://user-images.githubusercontent.com/104115586/187115611-c0677fe5-cbed-40e8-9039-896373725a92.png">



**`SMOTE (Synthetic Minority Oversampling Technique) Model`**, like `RandomOverSampler` increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection. 

  <img width="419" alt="Screen Shot 2022-08-28 at 9 15 36 PM" src="https://user-images.githubusercontent.com/104115586/187116022-36e0d606-2b10-4611-b310-56baaacfb767.png">

  * The balanced accuracy score decreased slightly to 62%.

  
  <img width="357" alt="Screen Shot 2022-08-28 at 8 59 09 PM" src="https://user-images.githubusercontent.com/104115586/187115658-fc859c8d-746e-4bf1-b482-2c8290215d96.png">



  * Like `RandomOverSampler`, the "High Risk" precision rate again was only 1% with the recall degraded to 59% giving this model an F1 score of 2%.
  * "Low Risk" had a precision rate of 100% and an improved recall at 65%.  

  <img width="372" alt="Screen Shot 2022-08-28 at 8 59 15 PM" src="https://user-images.githubusercontent.com/104115586/187115674-0a6f6542-d478-4e10-aec2-df569ccef72b.png">

  <img width="695" alt="Screen Shot 2022-08-28 at 8 59 20 PM" src="https://user-images.githubusercontent.com/104115586/187115677-b401977d-2815-4ae4-a942-624f822b5768.png">


### Undersampling

**`ClusterCentroids Model`**, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 260 records each as High Risk and Low Risk.

<img width="387" alt="Screen Shot 2022-08-28 at 8 59 34 PM" src="https://user-images.githubusercontent.com/104115586/187115635-8619f1e6-24e6-49ca-babc-24780be1db86.png">

  * Balanced accuracy score was lower than the oversampling models at 52%.

  <img width="348" alt="Screen Shot 2022-08-28 at 9 23 19 PM" src="https://user-images.githubusercontent.com/104115586/187116771-6bed19fd-73ea-4095-a3d8-564be3cc4528.png">


  * The "High Risk" precision rate again was only at 1% with the recall at 60% giving this model an F1 score of 1%.
  * "Low Risk" had a precision rate of 100% and with a lower recall at 44% compared to the oversampling models.  

  <img width="371" alt="Screen Shot 2022-08-28 at 9 24 01 PM" src="https://user-images.githubusercontent.com/104115586/187116860-1a27c33b-4c4d-48e6-9c79-42b8009b1a70.png">

  <img width="703" alt="Screen Shot 2022-08-28 at 9 24 05 PM" src="https://user-images.githubusercontent.com/104115586/187116876-d39dc4f0-da3a-4255-bfee-b10701472046.png">




## Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

### Combination Sampling

**`SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model`** combines aspects of both oversampling and undersampling. The model classified 51,351 records as High Risk and 46,389 as Low Risk.

<img width="411" alt="Screen Shot 2022-08-30 at 4 23 03 PM" src="https://user-images.githubusercontent.com/104115586/187554249-4cc27f59-ae88-43f1-9538-0227c6ccae64.png">

  * The balanced accuracy score improved to 62% when using a combined sampling model.

  <img width="341" alt="Screen Shot 2022-08-30 at 4 23 08 PM" src="https://user-images.githubusercontent.com/104115586/187554266-62ee97dd-d8d4-4043-ba6e-ad1c51bab109.png">



  * The "High Risk" precision rate did not improve was only 1%, however the recall increased to 70% giving this model an F1 score of 2%.
  * "Low Risk" still showed a precision rate of 100% with the recall at 54%.  
  
  <img width="376" alt="Screen Shot 2022-08-30 at 4 23 13 PM" src="https://user-images.githubusercontent.com/104115586/187554275-d6dad9a5-0abf-4df5-9467-ef9fd9d13f37.png">


  <img width="696" alt="Screen Shot 2022-08-30 at 4 23 19 PM" src="https://user-images.githubusercontent.com/104115586/187554281-e870df65-6894-4395-9cc4-a172e3910ed3.png">


## Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

Compare two new `Machine Learning` models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.

 <img width="398" alt="Screen Shot 2022-08-28 at 9 27 07 PM" src="https://user-images.githubusercontent.com/104115586/187117626-183460ea-e03b-4524-909f-51a8e2e92f0a.png">


**`BalancedRandomForestClassifier Model`**, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class. 

  * The balanced accuracy score increased to 79% for this model.

 <img width="361" alt="Screen Shot 2022-08-28 at 9 27 18 PM" src="https://user-images.githubusercontent.com/104115586/187117684-b40b6a8e-746c-4e93-9747-cf9c1ac8d536.png">


  * The "High Risk precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
  * "Low Risk" still had a precision rate of 100% with the recall at 87%.  
  *
<img width="474" alt="Screen Shot 2022-08-28 at 9 27 23 PM" src="https://user-images.githubusercontent.com/104115586/187117714-a8c27162-38fe-4d6c-bb71-84d7d58ef947.png">

<img width="699" alt="Screen Shot 2022-08-28 at 9 27 28 PM" src="https://user-images.githubusercontent.com/104115586/187117724-322663dd-bda9-446f-9ae7-ceff9e3d507d.png">

 The top feature by importance was "total_rec_prncp" at 7.9% of the total.
  
<img width="708" alt="Screen Shot 2022-08-28 at 9 27 40 PM" src="https://user-images.githubusercontent.com/104115586/187553658-91b60f98-90ff-4038-8852-283880c5b453.png">

**`EasyEnsembleClassifier Model`**, a set of classifiers where individual decisions are combined to classify new examples.

  * The balanced accuracy score increased to 93% with this model.

  <img width="356" alt="Screen Shot 2022-08-28 at 9 27 54 PM" src="https://user-images.githubusercontent.com/104115586/187117763-3abc87cd-2374-4bbb-a51b-1a05c89b3025.png">


  * The "High Risk precision rate increased to 9% with the recall at 92% giving this model an F1 score of 16%.
  * "Low Risk" still had a precision rate of 100% with the recall now at 94%.  

  <img width="483" alt="Screen Shot 2022-08-28 at 9 27 59 PM" src="https://user-images.githubusercontent.com/104115586/187117779-afda92f5-5b09-4d41-a50e-96e71eb95fae.png">

<img width="712" alt="Screen Shot 2022-08-28 at 9 28 04 PM" src="https://user-images.githubusercontent.com/104115586/187117787-bc99842c-dfd0-4dd1-90fd-069039749312.png">


# Summary

In reviewing all six models, the `EasyEnsembleClassifer` model yielded the best results with an accuracy rate of 93% and a 9% precision rate when predicting "High Risk candidates. The sensitivity rate (aka recall) was also the highest at 92% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, then this one would be the clear choice.

**Ranking of models in descending order based on "High Risk" results:**
* `EasyEnsembleClassifer`: 93% accuracy, 9% precision, 92% recall, and 16% F1 Score
* `BalancedRandomForestClassifer`: 79% accuracy, 3% precision, 70% recall and 6% F1 Score
* `SMOTEENN`: 62% accuracy, 1% precision, 70% recall and 2% F1 Score
* `SMOTE`: 62% accuracy, 1% precision, 59% recall and 2% F1 Score
* `RandomOverSampler`: 65% accuracy, 1% precision, 61% recall and 2% F1 Score
* `ClusterCentroids`: 52% accuracy, 1% precision, 60% recall and 1% F1 Score

A side note that should be considered is that original dataset had 99% of the applications classified as "Low Risk" with only 1% of the data classified in the "High Risk" category. This may skew the results greatly as there is a risk that the `Machine Learning` algorithms are creating clusters drawing from too small of a dataset of actual "High Risk" applications. This margin of risk might not be something that banks would be comfortable accepting.


