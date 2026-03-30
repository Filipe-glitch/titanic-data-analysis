#%%  IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% CARREGAMENTO DOS DADOS 
titanic = pd.read_csv('titanic.csv', sep=',', decimal='.')

#%% Organização dos bancos de dados
titanic.info()

#%% Armazenar variaveis num objeto
classe = titanic['classe']
selecao_var = titanic[['sobreviveu', 'classe', 'sexo']]

#%% Removendo variáveis
selecao_var = selecao_var.drop(columns=['classe'])

#%% SELECIONANDO FILTROS E CONSULTAS
# Quem Sobreviveu?
sobreviveu = titanic[titanic['sobreviveu'] == 'sim']

# Quantos adultos do sexo masculino tinha no Titanic?
masc_adultos = (titanic[(titanic['sexo'] == 'masculino') & 
                        (titanic['idade'] >= 18)])

# Quantos eram da segunda classe e estavam no embarque Southampton?
thamp_segunda = (titanic[(titanic['classe'] == 'segunda') &
                         (titanic['embarque'] == 'Southampton')])

# Quantos da segunda classe tinha a bordo
apenas_segunda = (titanic[(titanic['classe'] == 'segunda')])

# Quantas mulheres ou crianças tinham a  bordo?
mulheres_criancas = (titanic[(titanic['sexo'] == 'feminino') |
                             (titanic['idade'] < 18)])

# Quantas pessoas estavam sozinhas (sem parentes)?
sozinhos = (titanic[(titanic['irmaos_conjuges'] == 0) &
                    (titanic['pais_filhos'] == 0)])

# Quantas pessoas pagaram mais de $100 pela tarifa e embarcaram em Cherbourg?
tarifas_embarque = (titanic[(titanic['valor_tarifa'] > 100) &
                            (titanic['embarque'] == 'Cherbourg')])

#%% LIMPEZA DE DADOS 
titanic_limpo = titanic.drop(columns=['nivel_cabine']).dropna() 

#%% ANÁLISE
#Frequência de sobrevivientes
titanic['sobreviveu'].value_counts()
titanic['sobreviveu'].value_counts(normalize=True)

#Utilizar os DataFrames filtrados
sozinhos['sobreviveu'].value_counts(normalize=True)
masc_adultos['sobreviveu'].value_counts(normalize=True)

#%% ESTATÍSTICAS DESCRITIVAS
titanic['idade'].count() # Contagem de observações
titanic['idade'].mean() # Média aritmética
titanic['idade'].median() # Mediana
titanic['idade'].max() # Máximo
titanic['idade'].min() # Mínimo
titanic['idade'].quantile(q = 0.25) # Primeiro Quartil
titanic['idade'].quantile(q = 0.75) # Terceiro Quartil
titanic['idade'].var() # Variância
titanic['idade'].std() # Desvio padrão

# ANÁLISE AGRUPADA
titanic['idade'].describe()
titanic[['idade', 'valor_tarifa']].describe()
titanic.describe()

#%% ANÁLISE AGRUPADA
media_tarifa = titanic[['classe', 'valor_tarifa']].groupby(by=['classe']).mean()
sobrevivencia_sexo = titanic[['sobreviveu', 'sexo']].groupby('sexo')['sobreviveu'].value_counts(normalize=True)
sobrevivencia_classe = titanic[['sobreviveu', 'classe']].groupby('classe')['sobreviveu'].value_counts(normalize=True)

print(media_tarifa)
print(sobrevivencia_sexo)
print(sobrevivencia_classe)
#%% GRÁFICO DE BARRAS PARA CONTAGEM
plt.figure(figsize=(15,9), dpi=600)
sns.countplot(data=titanic, x='sobreviveu')
plt.show()

# Adicionando formatações ao gráfico
plt.figure(figsize=(15,9), dpi=600)
sns.countplot(data=titanic, x='sobreviveu', color='purple')
plt.title('Titanic',fontsize=20)
plt.xlabel('Sobreviveu?',fontsize=15)
plt.ylabel('Contagem',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Adicionando as contagens
plt.figure(figsize=(15,9), dpi=600)
ax = sns.countplot(data=titanic, x='sobreviveu', color='purple')
for container in ax.containers: ax.bar_label(container, fontsize=12)
plt.title('Titanic',fontsize=20)
plt.xlabel('Sobreviveu?',fontsize=15)
plt.ylabel('Contagem',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Detalhando as informações entre grupos
plt.figure(figsize=(15,9), dpi=600)
ax = sns.countplot(data=titanic, x='sobreviveu', hue='sexo', palette='viridis', legend=True)
for container in ax.containers: ax.bar_label(container, fontsize=12)
plt.title('Titanic',fontsize=20)
plt.xlabel('Sobreviveu?',fontsize=15)
plt.ylabel('Contagem',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#%% GRÁFICO DE BARRAS GERAL
# Valor médio da tarifa em cada local de embarque 
dados = titanic[['embarque', 'valor_tarifa']].groupby(by=['embarque']).mean()

# Gerando o gráfico de barras
plt.figure(figsize=(15,9), dpi=600)
ax1 = sns.barplot(data=dados, x=dados.index, y='valor_tarifa', hue=dados.index, palette='bright')
for container in ax1.containers: ax1.bar_label(container, fmt='%.2f', padding=3, fontsize=12)
plt.title("Tarifa Média",fontsize=20)
plt.xlabel('Cidade de Embarque',fontsize=15)
plt.ylabel('Valor',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#%% HISTOGRAMA
# Histograma da idade \ Variável métrica
plt.figure(figsize=(15,9), dpi=600)
sns.histplot(data=titanic, x='idade', color='lightgreen')
plt.show()

# Aprimorando a apresentação do gráfico
plt.figure(figsize=(15,9), dpi=600)
sns.histplot(data=titanic, x='idade', bins=range(0,85,5), kde=True, color='lightgreen')
plt.xlabel('Idade',fontsize=15)
plt.ylabel('Frequência',fontsize=15)
plt.xticks(ticks=np.arange(0,85,5), fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#%% GRÁFICO DE PONTOS (SCATTERPLOT)
# Objetivo de analisar passageiros que pagaram menos de $100
plt.figure(figsize=(15,9), dpi=600)
sns.scatterplot(data=titanic[titanic['valor_tarifa']<100], x='idade', y='valor_tarifa')
plt.show()

# Existe alguma relação entre as variáveis? 
plt.figure(figsize=(15,9), dpi=600)
sns.regplot(data=titanic[titanic['valor_tarifa']<100], x='idade', y='valor_tarifa')
plt.show()

# Adicionando outras variáveis
plt.figure(figsize=(15,9), dpi=600)
sns.scatterplot(data=titanic[titanic['valor_tarifa']<100], x='idade', y='valor_tarifa', 
                hue='classe', hue_order=['primeira', 'segunda', 'terceira'])
plt.title('Titanic',fontsize=20)
plt.xlabel('Idade',fontsize=15)
plt.ylabel('Valor da Tarifa',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#%% GRÁFICO DE LINHAS
# Vamos separar cada classe por meio da cor da linha
dados_linha = titanic[['classe', 'valor_tarifa']].groupby(by=['classe']).mean()

plt.figure(figsize=(15,9), dpi=600)
sns.lineplot(data=dados_linha, x=dados_linha.index, y='valor_tarifa', marker='o', linewidth=3, color='purple')
plt.title('Titanic',fontsize=20)
plt.xlabel('Classes',fontsize=15)
plt.ylabel('Tarifa Média',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#%% BOXPLOT
# Gráfico para representar as medidas de posição de variáveis métricas
plt.figure(figsize=(15,9), dpi=600)
sns.boxplot(y=titanic['idade'], color='orange')

# Calcula as estatísticas
minimo = titanic['idade'].min()
q1 = titanic['idade'].quantile(q = 0.25)
q2 = titanic['idade'].median()
q3 = titanic['idade'].quantile(q = 0.75)
maximo = titanic['idade'].max()

# Adiciona os textos no gráfico
plt.text(0, minimo, f'Mín = {minimo}', ha='center', va='top', fontweight='bold')
plt.text(0, q1, f'Q1 = {q1:.1f}', ha='center', va='top', fontweight='bold')
plt.text(0, q2, f'Q2 = {q2:.1f}', ha='center', va='center', fontweight='bold')
plt.text(0, q3, f'Q3 = {q3:.1f}', ha='center', va='bottom', fontweight='bold')
plt.text(0, maximo, f'Máx = {maximo}', ha='center', va='bottom', fontweight='bold')

plt.title('Boxplot da Idade',fontsize=20)
plt.show()