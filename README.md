# Alzheimer-s-Disease-and-Machine-Learning-Paper
## 1.	Abstract
The project aims to apply supervised machine learning to an Alzheimer’s data set of 1149 patients who are aged between 60 and 97 to support Alzheimer’s research and diagnosis. Using a synthetic dataset derived from a real-world dataset, this work focuses on classifying demented and nondemented groups with demographic data, cognitive assessment scores, and MRI-derived data. The goal is to develop and evaluate the machine learning models and demonstrate how machine learning can support early diagnosis of Alzheimer’s disease. 
The primary objectives of the project are to preprocess the data, implement multiple machine learning models, and evaluate the performance of these models using different metrics. The outcome of the project will indicate which features or combinations of features are the best for classifying Alzheimer’s disease. This project is driven by the urgent need for an efficient and accurate method for diagnosing Alzheimer’s disease.
## 2.	Introduction

There are many types of dementia, including Vascular dementia, Lewy body dementia, frontotemporal dementia, and Alzheimer’s disease, which is the most common cause of dementia in the world. According to the World Alzheimer Report 2018, around 50 million people worldwide were living with dementia, and the number was forecast to triple to 152 million by 2050. The development of Alzheimer’s Disease involves in 2 abnormal levels of proteins, which are beta-amyloid and tau. If the beta-amyloid reaches abnormal levels, it forms plaques that collect between neurons and disrupt cell function. Also, for tau, it forms neurofibrillary tangles inside neurons, which block the neuron’s transport system. These 2 key factors depend on risk factors such as genetics, age, lifestyle, and environment. (Patterson, 2018)
	There is currently no cure for Alzheimer’s disease, but there are treatments and therapies to manage and slow down the progression and reduce symptoms. Additionally, early diagnosis of Alzheimer’s disease is crucial because it enables the patient to reduce risks by modifying lifestyle, which is the most effective. Identifying the disease in the early stage allows for the implementation optimal strategy to preserve the patient’s level longer. (Jill Rasmussen, Haya Langerman, 2019)
	Machine learning is a subset of artificial intelligence (AI) that focuses on algorithms that can learn from patterns of data and predict and make decisions.  (Crabtree, 2024) Machine learning can significantly assist in the research and diagnosis of the disease since it is capable of learning from data, identifying patterns, and making decisions or predictions. Utilizing machine learning improved the diagnosis of Alzheimer’s disease by enhancing sensitivity and specificity. Machine learning algorithms such as support vector machine (SVM), logistic regression, linear regression, and random forest are used to build an optimal predictive model for distinguishing mild cognitive impairment (MCI) and Alzheimer’s disease (AD) patients by using data such as Clinical Dementia Rating (CDR) and Mini-Mental State Examination (MMSE). Another advantage of using machine learning is that it reduces cost by using widely available and cost-effective data, like cognitive assessment, instead of testing expensive tests like Positron Emission Tomography (PET) scan. Overall, machine learning supports early detection, scalable diagnosis, and clinical decision-making. (Chun-Hung Chang, Chieh-Hsin Lin, Hsien-Yuan Lane, 2021)
## 3.	Literature Review 
Alzheimer’s disease is a non-reversible neurodegenerative disease that is characterized by cognitive decline, memory loss, and brain structure changes. Conventional diagnosis depends on clinical evaluation, cognitive assessments, and neuroimaging; however, these methods are not able to identify early stages of disease. Therefore, researchers lately increasingly tried to apply machine learning (ML) to support the early detection of the disease. 
	Several studies have shown that the effectiveness of using machine learning in diagnosing Alzheimer’s disease using cognitive scores and neuroimaging. Suk et al (2014) demonstrated the application of structural MRI data and Gray Matter Tissue Densities to deep learning models and reported high accuracy in classifying Alzheimer’s disease and Mild Cognitive Impairment (MCI). (Heung-Il Suk, Seong-Whan Lee, Dinggang Shen, 2014) Similarly, Lodha (2018) was able to accurately distinguish Alzheimer’s disease with a high accuracy in the Neural Network model and the Random Forest model, primarily using neuroimaging and brain structure data.  (Priyanka Lodha, Ajay Talele and Kishori Degaonkar, 2018) Most recently, Antor et al (2021) used Support Vector Machine (SVM), logistic Regression, Decision Tree, and Random Forest with demographic data, cognitive data, and MRI-Derived data. The researchers were able to demonstrate the high performance of SVM after fine-tuning. Compared to other Random Forest, Decision Tree, and Logistic Regression models, SVM scored the highest across the key metrics. (Morshedul Bari Antor, A.H.M.Shafayet Jamil, Maliha Mamtaz, Mohammad Monirujjaman Khan, Sultan Aljahdali, Manjit Kaur, Parminder Singh, and Mehedi Masud, 2021) 
Despite the promising results, researchers faced several challenges, including a small dataset size and data imbalance. Machine learning models generally require large training samples to get reliable results. Data imbalance between classes can result in poor performance on the minority classes and introduce bias. This project addresses these gaps by applying a machine learning algorithm to the Alzheimer’s dataset. With careful preprocessing and class balancing, this project will evaluate to access effectiveness in support diagnosis of Alzheimer’s disease. 

## 4.	Methodology	
### 4.1 Data Preprocessing 

<img width="975" height="198" alt="image" src="https://github.com/user-attachments/assets/e1c5fdd2-0567-4bf6-a987-0492c90adfd5" />
Figure (1): Alzheimer’s disease dataset 

Data preprocessing was performed to ensure data equality and prevent unnecessary bias in results. Initially, the dataset was inspected for missing values and duplicate rows. As a result, there were 162 missing values in the M/F categorical data, 110 in EDUC, and 44 in NIS, which are continuous variables. Missing categorical values were imputed by the most frequent value, and missing continuous values were replaced by the average value of that variable. 
The class distribution was examined, and the major class was undersampled using a random undersampling technique. This prevents model bias in predictions. 
Categorical variables were encoded into numerical forms for the machine learning algorithm to understand and process. M/F and Group were converted to binary variables. 

<img width="975" height="446" alt="image" src="https://github.com/user-attachments/assets/f2d1287e-a911-4108-bf63-3a5a095b4e04" />
Figure (2): Gender class distribution before resampling

<img width="975" height="446" alt="image" src="https://github.com/user-attachments/assets/f4f9ecc4-36a6-4207-baed-327bf1218b07" />
Figure (3): Gender class distribution after resampling

### 4.2 Model Development
<img width="975" height="480" alt="image" src="https://github.com/user-attachments/assets/b38816db-5060-4d4b-b27e-60732eef69c1" />
Figure (4): Flowchart of Linear Regression

Linear regression was chosen to predict a linear relationship between one or more independent variables and a dependent variable.
Figure 4 illustrates the flow diagram of the linear regression machine learning model. The data is split into 70/30, which is appropriate for a large dataset. Default parameters were used as it does not require extensive tuning.
Linear regression model performance is measured using
Mean Squared Error (MSE): average of the squared difference between predicted values and actual values; the mean absolute error 
•	Mean Absolute Error (MAE): average absolute difference between predicted values and actual values
•	R2 score: the measure of how well the model’s predictions match the real data. 

<img width="982" height="481" alt="image" src="https://github.com/user-attachments/assets/985669f0-7587-40a4-a957-a3c2ece3932f" />
Figure (5): Random Forest Classifier Flowchart
Random Forest was selected to handle complex and non-linear relationships. For example, Alzheimer’s biomarkers such as MMSE, CDR, and MRI features often interact nonlinearly. Unlike linear models like Logistic Regression and Linear Regression, it does not make a linear assumption. In addition, one of the advantages of Random Forest is that the model is robust to noise and outliers. Especially in the medical field, the dataset may contain errors and variability.
Parameter used:
•	Number of Trees (n_estimators=100): 100 is the balance between performance and computations. The model might underfit if there are too few trees and overfit if there are too many trees.
•	Maximum tree depth (max_depth = 20): 20 is relatively deep to capture complex patterns in the data, but not too many to overfit the training data.
Figure 5 is a representation flow diagram of Random Forest. Same as linear regression, it begins with data loading and preprocessing. Unlike linear regression and other models, the Random Forest Classifier does not require feature scaling. Like other models, this model was also implemented with a 70/30 train-test split.

<img width="975" height="577" alt="image" src="https://github.com/user-attachments/assets/f4956ed9-3d1c-47b0-8cbf-6cce88d4121a" />
Figure (6): Logistic Regression Flowchart

Logistic Regression was used to classify patients into the demented and nondemented group. This model is simple and easy to interpret, allowing the feature to contribute through coefficients. It provides probabilities that can be used for decision-making in clinical settings, giving the probability of having Dementia. Penalty (penalty=l2) was applied to prevent overfitting.
The flow of the Logistic Regression was shown in the figure 6. Similar to other models, it begins with data preprocessing and splitting the train data and the test data using a 70:30 ratio. However, the Logistic Regression model can benefit from scaling. By scaling, all features are treated equally, and the model coverage is faster and more reliable. Finally, the model is trained, and predictions are made on test data.
 
<img width="975" height="565" alt="image" src="https://github.com/user-attachments/assets/b2aa42ee-3966-4084-af0f-adc256992ebe" />
Figure (7): Support Vector Machine Flowchart 

A Support Vector Machine (SVM) was employed for classification and the non-linear relationship between features. This model can strongly classify the demented group and nondemented which is a binary value since it can handle high-dimensional data, and it finds the most optimal hyperplane which can maximize the margin between the two classes. SVM was selected to effectively handle high-dimensional data. The flow of the model is also the same as the Logistic Regression model. 
### 4.3 Model Evaluation

Model performance was assessed using several metrics for comprehensive analysis. Except for the Linear Regression, the Random Forest Classifier, the Logistic Regression, and the Support Vector Machine, which were measured using accuracy to measure overall performance, precision to measure the correctness of model predictions, recall to measure the rate of missed predictions, and F1 to measure overall reliability of the model diagnosis. In addition, ROC accuracy score was evaluated to measure how well the classifier model perform distinguishing between the demented group and nondemented group.
 

## 5.	Results and Discussion

The results of this study mainly focus on the performance of the four machine learning models – Linear Regression, Random Forest Classifier, Logistic Regression, and Support Vector Machine (SVM) in Alzheimer’s disease prediction. 

### 5.1	Linear Regression 

<img width="646" height="492" alt="image" src="https://github.com/user-attachments/assets/a022c519-6d29-4819-8691-14dc2815c342" />
Figure (7): Linear Regression Visualization 

The Linear Regression Model shows a great performance in predicting ASF using eTIV, with the performance metrics on the test set: 
•	Mean Squared Error (MSE): 0.0008417328919800539
•	Mean Absolute Error (MAE): 0.024676864872618897
•	R² Score: 0.9760905010871366
These results indicate the model was able to predict very close to the actual values with very low average error. The high R² Score of 0.976 shows that the selected features fit well. Although the Linear Regression demonstrates a strong linear fit as shown in Figure 7, it has no ability to classify Alzheimer’s disease, which is non-linear. In addition, the Linear Regression can only explore a linear relationship between continuous features.

### 5.2	Random Forest Classifier

<img width="535" height="374" alt="image" src="https://github.com/user-attachments/assets/d5e16fc5-1f58-40c1-932b-50e2d46f1f18" />
Figure (8) Confusion Matrix of Random Forest Classifier using MMSE

<img width="529" height="276" alt="image" src="https://github.com/user-attachments/assets/c858a68a-567b-401c-bae2-5fa719b796cf" />
Figure (9) Random Forest Model Performance using MMSE

<img width="532" height="382" alt="image" src="https://github.com/user-attachments/assets/a26287b0-1772-46e4-bed1-dff6923b5fa1" />
Figure (10) Confusion Matrix of Random Forest Classifier using CDR

<img width="591" height="306" alt="image" src="https://github.com/user-attachments/assets/de19ce40-6383-4c47-924c-99c2d0a93b63" />
Figure (11) Random Forest Model Performance using CDR

The Random Forest model achieved strong classification performance. The model gives 303 correct predictions and 79 wrong predictions with the MMSE feature, and gives 359 correct predictions and 23 wrong predictions with the feature. The model effectively handled complex and non-linear interactions and was robust to noise and outliers. The CDR and MMSE features are the most influential variables for the model.

### 5.3	Logistic Regression
<img width="656" height="303" alt="image" src="https://github.com/user-attachments/assets/ddced127-6239-4fa2-9a2d-11d013b38104" />
Figure (12) Logistic Regression Performance using CDR

<img width="684" height="495" alt="image" src="https://github.com/user-attachments/assets/710bd67e-a9fd-4704-adc6-145e5c21179c" />
Figure (13) Confusion Matrix using CDR

<img width="738" height="331" alt="image" src="https://github.com/user-attachments/assets/e0e65cbb-6606-4436-b93b-967e5d7b0787" />
Figure (14) Logistic Regression Performance using MMSE

<img width="590" height="416" alt="image" src="https://github.com/user-attachments/assets/54bdeaf3-f525-47ef-9035-2f5dee28ea20" />
Figure (15) Confusion Matrix using MMSE

The Logistic Regression also got a high score on multiple metrics. Similar to the Random Forest Classifier, MMSE and CDR score were the most influential predictors. The model was able to predict a large number of patients with dementia, but showed limitations in predicting non-demented patients. The model gives 535 correct predictions out of 573 with CDR features and 449 correct predictions out of 573 with the MMSE feature.

### 5.4	Support Vector Machine

<img width="638" height="310" alt="image" src="https://github.com/user-attachments/assets/a5255d4c-3fc8-47c0-9f25-e616494043ed" />
Figure (16) SVM Performance using MMSE

<img width="870" height="614" alt="image" src="https://github.com/user-attachments/assets/fb4815c3-b217-473e-b812-c79c2129ed16" />
Figure (17) SVM Confusion Matrix using MMSE

<img width="654" height="328" alt="image" src="https://github.com/user-attachments/assets/428a62f8-5c94-4948-9a42-55349d48d3cd" />
Figure (18) SVM Performance using CDR

<img width="571" height="413" alt="image" src="https://github.com/user-attachments/assets/41aed7e3-318a-4f79-bc6c-a6d0d89ba49c" />
Figure (19) Confusion Matrix using CDR

The SVM model showed strong classification performance. The model achieved an overall high performance. The SVM model predictions also performed better with CDR and MMSE over other features. It yields 459 correct predictions and 114 incorrect predictions using the MMSE features. In addition, it gives 535 correct predictions and 38 incorrect predictions using the CDR feature, which is the same as the Logistic Regression. The model predicts fewer mistakes with the CDR feature. 

### 5.5	Discussion

<img width="975" height="355" alt="image" src="https://github.com/user-attachments/assets/22c0ba18-4404-4aef-8063-ac7ccdb2f046" />
Figure (20) ROC AUC Score Comparison 

<img width="975" height="379" alt="image" src="https://github.com/user-attachments/assets/e9a64b1b-af41-48c6-93fb-767db7767049" />
Figure (21) Accuracy Score Comparison

<img width="1498" height="766" alt="image" src="https://github.com/user-attachments/assets/c5f25938-9403-4cea-af38-59d5fab33c84" />

By looking at the accuracy score comparison and the ROC AUC score comparison, the Random Forest Classifier and SVM models capture complex relationships between cognitive scores and the demented or nondemented group. However, the Random Forest Classifier was able to predict nondemented group perfectly, and got 1.00 in F1 score (shown in figures 9 and 11) with both CDR and MMSE features. Logistic Regression has lower accuracy compared to more advanced models due to its sensitivity to the linear assumption done by Logistic Regression features and outcome.
The Logistic Regression remains valuable due to its superior interpretability and probabilistic outputs. It outperformed in Although Linear Regression was able to perform well in predicting ASF using eTIV, it has limitations in binary classification. 

<img width="975" height="542" alt="image" src="https://github.com/user-attachments/assets/dd590f16-9351-43e1-9c6b-085d7e71596a" />
Figure (22): Feature Importance Bar Graph

Feature importance analysis indicates that the cognitive scores (MMSE and CDR) are the most predictive features. Among MRI-derived features, such as nWBV, eTIV, and ASF, are not predictive features since the features extracted are limited and the dimensions are high. Demographic variables like age and education played an important role, but they are not the best for the model to predict on their own. 

<img width="701" height="259" alt="image" src="https://github.com/user-attachments/assets/9836ba8e-b047-4d58-ade1-c2617582c860" />
Figure (23) SVM performance with ASF, MMSE, CDR, and eTIV

Even though eTIV, AST, and nWBV cannot perform predictions on their own, when combined with CDR and MMSE, the accuracy seems promising.   

Early detection of Alzheimer’s disease is crucial for care planning and diagnosing symptoms early and accurately. Machine learning models such as the Random Forest Classifier and Support Vector Machine can faster way to detect Alzheimer’s disease in a more cost-effective and efficient way. Moreover, machine learning can scale large amounts of datasets as it increases.
