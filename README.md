studentDropoutPredictor:
This project develops a machine learning model that predicts whether a high-school student is at high or low risk of dropping a class. Instead of relying on grades, the model uses behavioral, demographic, and lifestyle information, which makes it more suitable for early intervention. The work uses the Mathematics portion of the UCI Student Performance dataset, which provides a broad set of variables describing each student.

Project Motivation:
Schools often attempt to identify struggling students early, but these efforts usually depend on exam results or mid-term grades. This creates a delay because many of the warning signs appear long before grades are available. The idea behind this project is to see whether non-grade factors such as study habits, absences, prior failures, family support, alcohol consumption, health, and free-time behavior can provide a meaningful signal about academic risk. The dataset is imbalanced, so SMOTE is used to help the model learn patterns for the smaller high-risk group.

Dataset:
The project uses the student-mat.csv dataset from the UCI Machine Learning Repository. It contains 395 samples and 33 input features that include lifestyle habits, family structure, personal characteristics, and academic behavior. The target variable is created from the final grade, where students with a grade below 10 are labeled as high-risk and students with a grade of 10 or above are labeled as low-risk, based on the fact that the grading system is out of 20. The grade columns G1, G2, and G3 are removed to prevent the model from learning directly from data that determines the target.

Methodology:
The dataset is loaded, cleaned, and prepared for modeling. Categorical features are one-hot encoded to convert them into numerical form. The data is split into training and testing portions using stratified sampling so the proportion of high and low risk students is preserved. Several models are trained, starting with a logistic regression baseline, followed by a random forest classifier, a gradient boosting classifier, and a logistic regression model trained after applying SMOTE. Model performance is evaluated using accuracy, precision, recall, and F1-score. Particular attention is given to the ability of each model to correctly identify high-risk students. The final chosen model is saved for future use.

Model Performance:
The baseline logistic regression and random forest models achieved acceptable accuracy but performed poorly on the high-risk class. Gradient boosting performed slightly better but still struggled with the minority class. The most effective results came from applying SMOTE to balance the training data and then fitting a logistic regression model. This approach increased recall for high-risk students from 0.31 in the baseline model to 0.50. Although total accuracy decreased slightly, the improvement in detecting at-risk students is more valuable for an early-warning setting.

Final Model:
The final selected model is the SMOTE-balanced logistic regression model. It reaches an accuracy of about 0.66, a recall of 0.50 for the high-risk class, and an F1-score of 0.49 for the same class. These results show that the model is significantly more effective at identifying students who may struggle academically. Since the purpose of the project is early detection rather than perfect accuracy, the chosen model suits the goals of the project.

Running the Project:
To run the project, install the required dependencies using pip. After installing the packages, run the files. These scripts handle preprocessing, model training, model comparison, feature importance generation, and saving of the final trained model.

Future Work:
There are other possible directions for future improvement. These include testing gradient boosting frameworks such as XGBoost or LightGBM, applying hyperparameter tuning to further improve recall, and integrating new features that capture additional aspects of student behavior or self-reported information.

Mohammad Jeneidi
Florida State University
Computer Science