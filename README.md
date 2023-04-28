# A-Short-Cut-To-Mushrooms
Repository for Data Mining Final Project

# Table Of Contents
1. [Project Summary](#project-summary)  
2. [Expectations](#expectations)
3. [Pipeline Info](#pipeline-info)
4. [Dataset and Features](#dataset-and-features)
5. [Technologies](#technologies)
6. [Machine Learning Models](https://github.com/manhtiendoan/CSC4850-Machine-Learning-Project/edit/main/README.md#machine-learning-models)
7. [Citations](#citations)   

# Project Summary  
>> This project covers an assessment of five Machine Learning models used to analyze the UCI Mushroom dataset which contains identifying data on mushrooms from the Agaricus and Lepiota Families. This study aims to evaluate the performance of these models and determine which is most effective at classifying a mushroom’s ‘Edibility’, or whether a mushroom is toxic to humans. ‘Edibility. The performance of each model was evaluated based on metrics including accuracy, precision, recall, F1 score, and associated Receiver Operating Characteristic and learning curves. Model selection, for each algorithm, compared three independent train/test splits (50-50, 70-30, and 80-20) before undergoing three-fold cross-validation. The results of which were compared and the best models (by metrics) for each were selected based on their best scores automatically with a simple selection algorithm. The findings of this study provide insight into creating an effective pipeline for data analysis and model selection using real-world data.

# Expectations  

# Pipeline Info  
  >>  This project revolves around a custom Machine Learning pipeline built for the Google Colab platform using Python, Scikit-learn, Pandas and Matplotlib. The special GUI overlay field options for Colab were utilized to make adjusting global parameters, feature selection, encoding type (ordinal and One-Hot), models assessed easy and convenient. Models built into the Scikit learn library are easy to implement and will function given they have typical methods of classifiers like score() and fit(). After the data set is set up and the user selects a feature to classify (via drop down menu) the pipeline will evaluate with n-fold cross validation (set by user) and return the best model. The best models from cross validation are run for each Train/Test split designated in the code (50-50, 60-40, and 70-30). From these splits the ultimate best model is returned for each model and displayed in a table at the end along with all relevant metrics. ROC curve and Learning Curve plots are also displayed for the cross validated models.   

# Dataset and Features
### About the Dataset  
Dataset obtained from [UCI Machine Learning Repository]([https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles](https://archive.ics.uci.edu/ml/datasets/Mushroom))
>> This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.

### Raw Data  
* age, status, sex, orientation, body_type, diet, drinks, drugs, education, ethnicity, height, income, job, last_online, location, offspring, pets, religion, sign, smokes, speaks, essay0, essay1, essay2, essay3, essay4, essay5, essay6, essay7, essay8, essay9
* 59,949 raw entries 
* .csv format

For training and predicting, all used features were converted to numeric or binary data. These features are labeled with '_data' for use as an official for use in training and testing the models. 

### Feature Selection
 The following fields were used for classification: 
 >> edible,	gill-size, gill-color, bruises	ring-type, stalk-root, gill-spacing, stalk-surface-above-ring, stalk-surface-below-ring, population, habitat, ring-number, cap-surface, spore-print-color,	 stalk-color-above-ring,	stalk-color-below-ring,	veil-color, gill-attachment, stalk-shape, odor, cap-shape, cap-color, veil-type

### Edible
The binary field 'Edible' labels weather or not a mushroom is toxic to humans. 'p' for poisonous and 'e' for edible

# Technologies  
### [Python](https://www.python.org/) <img src="https://user-images.githubusercontent.com/60898339/222571123-81f8e8e4-b183-4f92-a4bc-95d9d3e9f007.png" width=25 height=25>

### [Google Colab](https://colab.research.google.com/) <img src="https://user-images.githubusercontent.com/60898339/233802082-d2c46791-530f-4c95-9bd0-0b0889f8a601.png" width=25 height=25>

## Libraries
### [scikit-learn](https://scikit-learn.org/) <img src="https://user-images.githubusercontent.com/60898339/233802426-495b6620-22ba-4910-a63c-fec3d4843210.png" width=5% height=5%>
### [NumPy](https://numpy.org/) <img src="https://user-images.githubusercontent.com/60898339/233802193-1a22a918-5a56-4e45-8f09-77f58d65629d.svg" width=25 height=25>
### [Pandas](https://pandas.pydata.org/) <img src="https://user-images.githubusercontent.com/60898339/233802257-a731902d-9557-4707-bfae-2ea0dfb3bf4b.svg" width=55 height=35>
### [Matplotlib](https://matplotlib.org/) <img src="https://user-images.githubusercontent.com/60898339/233802324-53ef5e2f-c190-43b1-a763-6c889f8d87cb.svg" width=65 height=45>

# Machine Learning Models 
* [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [Support Vector Machine (Linear Kernel)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
* [Support Vector Machine (RBF Kernel)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [Naive Bayes Classification](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [K-Nearest Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

# Citations

D. Dua and C. Graff, UCI Machine Learning Repository. University of California, Irvine, School of Information, 2017. [Online]. Available: http://archive.ics.uci.edu/ml

# My Info
<div align="">
	<tr>
		<td>
		<td>
    <b>&nbsp Robert Tognoni:</b>
		<a href="https://github.com/rtogn"><img src="https://user-images.githubusercontent.com/60898339/222575865-617bc990-796a-4e29-834e-b30762f11526.png" width=25 height=25></a>
		<a href="https://www.linkedin.com/in/robert-tognoni-9a4795b0"><img  src="https://user-images.githubusercontent.com/60898339/222576175-1d3213f8-a001-4e7e-bb75-046fe5951fe3.png" width=25 height=25></a> 
		</td>  
	</tr>
</div>  


