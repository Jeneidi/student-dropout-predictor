from trainModel import train_baseline_model
from trainRandomForest import train_rf_model
from trainSmoteModel import train_smote_model
from trainGradientBoost import train_gb_model

def get_baseline_scores():
    print("\nBASELINE LOGISTIC REGRESSION")
    train_baseline_model()

def get_rf_scores():
    print("\nRANDOM FOREST")
    train_rf_model()

def get_smote_scores():
    print("\nSMOTE + LOGISTIC REGRESSION")
    train_smote_model()

def get_gb_scores():
    print("\nGRADIENT BOOSTING")
    train_gb_model()

if __name__ == "__main__":
    get_baseline_scores()
    get_rf_scores()
    get_smote_scores()
    get_gb_scores()
