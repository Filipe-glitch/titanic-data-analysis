from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TitanicModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = RandomForestClassifier(random_state=42)

    def treinar(self):
        self.model.fit(self.X_train, self.y_train)
        return self

    def avaliar(self):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        
        print(f"Acurácia do Modelo: {acc:.4f}\n")
        print("Relatório de Classificação:")
        print(classification_report(self.y_test, y_pred))
        
        return acc