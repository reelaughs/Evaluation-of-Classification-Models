**Assignment for #Data-Mining module, NTU MSc Information Systems**

The task was to compare two existing implementations of classifiers; firstly training and testing them on a dataset provided followed by comparing the two models using techniques for classification model comparison. This included a full performance comparison for the two models including the time it took to train and apply the model, as well as justification for all decision making e.g., the fine-tuning of model hyper-parameters, handling of missing values, stopping criterion, etc as well as underlying thought processes underlying any data pre-processing.

For this task, I chose to compare knn classifiers, traditionally known as lazy learners that are simple in nature but comptationally expensive, with decision tree classifiers, that are eager learners in contrast. A comparison paper was written which sought to understand this contrast as well as any differences of note in terms of performance with regards to executing a classification task on the same dataset.

_Data Preprocessing_
The following key transformations were applied to handle missing values, encode categorical features, and standardize numerical data: 
- Prior to the test dataset being loaded, the first line in the dataset “|1x3 Cross validator” was removed for easier processing. 
- Also, due to the target variable, gross annual income, being a categorical variable; LabelEncoder() was required in order to transform the target variable, gross annual income
- For the KNN classifier, KNNImputation is available in the scikit-learn library for handling numeric features, where the missing values were replaced with the average of the nearest neighbors.
- For categorical features, SimpleImputer was used instead as KNNImputer only handles numerical variables.
- For the knn classifier, the numeric features were standardized using StandardScaler
- For both the knn and decision tree classifiers, OneHotEncoder was also used to encode categorical features as a one-hot numeric array, or conversion into binary vectors for easier processing in the pipeline

_Tuning of Hyperparameters_
For the KNN model, a few potential n_neighbour values were tested via GridSearchCV, starting from 5 (the default value in scikit-learn), 9, 13, 16, 17 and 21. GridSearchCV was used due to the simple nature of the model and also due to the low number of parameters varied, which made a more exhaustive search possible
  - For the next steps tuning the weights metric (both uniform and distance were varied within this parameter to see which was optimal) as well as        distance (which tested minkowski, Euclidean and Manhattan), GridSearchCV was used again to determine the optimal hyperparameters
  - Along each step of determining the hyperparameters, cross validation was also evaluated to help select the best hyperparameters and avoid overfitting, via evaluating performance across different training data splits. This helps to estimate the model’s ability to generalize to unseen data.
    
For the decision tree classifier, Optuna was used rather than GridSearchCV as a search strategy due to the increased number of parameters. In contrast, GridSearchCV would have been too time and computationally intensive for the decision tree classifier.The hyperparameters being varied during the search were max_depth, criterion (gini and entropy), max_features, min_sample_split, min_samples_leaf. 

It is also to be noted that cv=3 as a parameter for the was kept for analysis across both knn and decision tree analysis to reduce computational cost.

For the decision tree classifier, cost complexity pruning was also performed to simplify the decision tree by reducing its size without significantly affecting accuracy. The best ccp_alpha yielding the highest test accuracy is selected for retraining to obtain the optimal pruned tree. A final round of cross validation is also done to ensure that the pruned model can generalise well

_Model Evaluation_
After obtaining the training accuracy score, the models were evaluated on the test set, which had not been used during training or cross-validation. A confusion matrix was also used to visualize the performance, showing how many instances were correctly or incorrectly classified.
Additional evaluation metrics were calculated via deriving a classification report from the data, including precision, recall, and F1-score, which provide insights into the model's performance across different classes.
