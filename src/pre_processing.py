import pandas as pd
from sklearn.model_selection import train_test_split

class TitanicPreProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def carregar_dados(self):
        self.df = pd.read_csv(self.filepath, sep=',', decimal='.')
        return self

    def engenharia_de_recursos(self):
        self.df['familia_total'] = self.df['irmaos_conjuges'] + self.df['pais_filhos']
        self.df['sozinho'] = (self.df['familia_total'] == 0).astype(int)    
        self.df['sexo_binario'] = self.df['sexo'].map({'feminino': 1, 'masculino': 0}) 
        self.df = pd.get_dummies(self.df, columns=['classe', 'embarque'], drop_first=True)
        
        return self

    def separar_dados(self):
        cols_drop = ['nivel_cabine', 'sexo', 'sobreviveu']
        X = self.df.drop(columns=[c for c in cols_drop if c in self.df.columns])
        y = (self.df['sobreviveu'] == 'sim').astype(int)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)