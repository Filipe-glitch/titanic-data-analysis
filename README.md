# 🚢 Pipeline de Machine Learning & Análise - Titanic

## 📌 Objetivo
Este projeto tem como objetivo realizar uma análise exploratória de dados (EDA) e construir um pipeline preditivo de Machine Learning para prever a sobrevivência dos passageiros do Titanic

---

## 📊 Tecnologias utilizadas
- Python
- Pandas & NumPy
- Scikit-Learn (Modelagem e Machine Learning)
- Matplotlib & Seaborn

---

## 📁 Estrutura do projeto
```text
titanic-data-analysis/
│
├── data/
│   └── titanic.csv
│
├── notebooks/
│   └── exploracao.ipynb
│
├── outputs/
│
├── src/
│   ├── pre_processing.py
│   └── models.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔍 Análises realizadas

- Distribuição de sobreviventes
- Análise por sexo e classe
- Idade dos passageiros
- Tarifas pagas
- Passageiros sozinhos vs acompanhados

---

## 🔍 Análises e Etapas Realizadas

- Exploração de Dados (EDA): Identificação de distribuições de idade, tarifas, classe e gênero.
- Engenharia de Recursos: Criação de variáveis compostas como o tamanho total da família e indicador de passageiro sozinho.
- Modelagem Preditiva: Implementação de pipeline de classificação utilizando Scikit-Learn para prever a sobrevivência.

---

## 💡 Principais insights

- Mulheres tiveram a maior taxa de sobrevivência
- Passageiros da primeira classe tiveram mais chances de sobreviver
- Idade influencia parcialmente na sobrevivência

---

## ▶️ Como executar

1. Clone o repositório:

```bash
git clone https://github.com/Filipe-glitch/titanic-data-analysis.git
```

2. Acesse a pasta do projeto:

```bash
cd titanic-data-analysis
```

3. Crie um ambiente virtual:

```bash
python -m venv .venv
```

4. Ative o ambiente virtual no Windows:

```bash
.venv\Scripts\activate
```

5. Instale as dependências:

```bash
pip install -r requirements.txt
```

6. Execute o projeto:

```bash
python main.py
```

---

## 📌 Créditos

- Este projeto foi desenvolvido com base em materiais educacionais, incluindo um minicurso da USP.
- Os dados utilizados são amplamente conhecidos e utilizados para fins de estudo em análise de dados (base do Titanic).
- O código foi adaptado, reorganizado e expandido para fins de aprendizado e prática pessoal em análise de dados com Python.
