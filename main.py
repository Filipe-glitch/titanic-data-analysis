from src.pre_processing import TitanicPreProcessor
from src.models import TitanicModelTrainer

def main():
    print("Iniciando o Pipeline do Titanic...")
    
    # Carregamento e Pré-processamento
    caminho = 'data/titanic.csv'
    processador = TitanicPreProcessor(caminho)
    X_train, X_test, y_train, y_test = (
        processador.carregar_dados()
                  .engenharia_de_recursos()
                  .separar_dados()
    )
    print("Dados limpos e separados com sucesso!")

    # Treinamento e Avaliação de Machine Learning
    print("Treinando o modelo de Machine Learning...")
    treinador = TitanicModelTrainer(X_train, X_test, y_train, y_test)
    treinador.treinar().avaliar()

if __name__ == "__main__":
    main()