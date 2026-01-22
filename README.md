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
 


