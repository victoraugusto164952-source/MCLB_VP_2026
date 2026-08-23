# ============================================================
#                 RESULTADOS LAB 01 - AULA 03 
# ============================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Dataset de Atendimento Financeiro
dados = {
    'mensagem': [
        'Como posso emitir a segunda via do meu boleto?',
        'Preciso da 2a via da minha fatura atrasada',
        'Quero negociar o pagamento da minha dívida',
        'Como fazer um acordo para pagar o débito em aberto?',
        'Gostaria de alterar meu endereço de cadastramento',
        'Onde atualizo os meus dados residenciais no app?'
    ],
    'intencao': [
        'segunda_via', 'segunda_via',
        'negociar_divida', 'negociar_divida',
        'atualizar_cadastro', 'atualizar_cadastro'
    ]
}

df1 = pd.DataFrame(dados)

# Criando lista de Stopwords personalizadas em Português
stopwords_pt = [
    'de', 'da', 'do', 'dos', 'das', 'a', 'o', 'as', 'os', 'em', 'para',
    'com', 'por', 'meu', 'minha', 'meus', 'minhas', 'como', 'quero', 'preciso'
]

# Vetorização TF-IDF aplicando stopwords e n-grams
vectorizer = TfidfVectorizer(stop_words=stopwords_pt, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(df1['mensagem'])
y = df1['intencao']

modelo = LogisticRegression()
modelo.fit(X_vec, y)

# Teste com nova mensagem
frase_teste = ["Preciso urgente da segunda via da fatura"]
frase_vec = vectorizer.transform(frase_teste)
predicao = modelo.predict(frase_vec)[0]

print("--- RESULTADOS DO LAB 01 (AULA 03) ---")
print(f"Mensagem: '{frase_teste[0]}'")
print(f"Intenção Predita: [{predicao}]")
print(f"Vocabulário Filtrado (sem stopwords): {list(vectorizer.get_feature_names_out())}")

#========== PRODUÇÃO DO RELATÓRIO:==============
# 1 - Qual o impacto da remoção de stopwords no tamanho do vocabulário do modelo?
As stopwords tiram as palavras desnecessárias para a interpretação do texto, se retirar ela
interfere na otimização da máquina.

# 2 - O que significa a configuração ngram_range=(1, 2) no TfidfVectorizer?
significa que ela vai aceitar uma certa combinação de palavras, exemplo nos parâmetros
minimo e máximo, no minimo ele vai aceitar uma quantidade de palavras combinadas, a mesma 
coisa para o parâmetro maximo, o que interfere diretamento no sentido da intenção.


# 3 - Como a remoção de palavras genéricas ajuda a evitar classificações incorretas?
Ajuda o algoritmo a focar somente nas palavras chaves que indicam mais a intenção 
do usuário, evitando confusão com as preposições que alteram o sentido da intenção

# Todos os resultados devem ser inseridos no arquivo resultados_aula03.md
#========== FIM ==============



--- RESULTADOS DO LAB 02 (AULA 03) ---



#========== PRODUÇÃO DO RELATÓRIO:==============
# 1 - O que representam as métricas Precision, Recall e F1-Score no relatório?
Elas representam a garantia dos resultados das instancias preditas, cada uma com
uma forma de chegar no resultado esperrado.

# 2 - Como interpretar a diagonal principal da Matriz de Confusão?
Ele registra as interações do chatbot e verifica quantas dessas interações estavam corretas
de acordo com a intenção di cliente.

# 3 - Por que a acurácia isolada pode ser enganosa quando temos classes desbalanceadas?
Ela pode ser enganosa por conta das previsões, se um conjunto de dados for majoritariamente
de uma classe, ela vai prever com base nesses dados, podendo haver erros em classes menores



--- Relatório de Classificação ---
                     precision    recall  f1-score   support

horario_atendimento       0.50      1.00      0.67         1
        localizacao       0.00      0.00      0.00         1
    troca_devolucao       0.00      0.00      0.00         1

           accuracy                           0.33         3
          macro avg       0.17      0.33      0.22         3
       weighted avg       0.17      0.33      0.22         3

--- Matriz de Confusão ---
[[1 0 0]
 [1 0 0]
 [0 1 0]]
