--- RESULTADOS DO LAB 01 (AULA 03) ---


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



Mensagem: 'Preciso urgente da segunda via da fatura'
Intenção Predita: [segunda_via]
Vocabulário Filtrado (sem stopwords): ['2a', '2a via', 'aberto', 'acordo', 'acordo pagar', 'alterar', 'alterar endereço', 'app', 'atrasada', 'atualizo', 'atualizo dados', 'boleto', 'cadastramento', 'dados', 'dados residenciais', 'débito', 'débito aberto', 'dívida', 'emitir', 'emitir segunda', 'endereço', 'endereço cadastramento', 'fatura', 'fatura atrasada', 'fazer', 'fazer um', 'gostaria', 'gostaria alterar', 'negociar', 'negociar pagamento', 'no', 'no app', 'onde', 'onde atualizo', 'pagamento', 'pagamento dívida', 'pagar', 'pagar débito', 'posso', 'posso emitir', 'residenciais', 'residenciais no', 'segunda', 'segunda via', 'um', 'um acordo', 'via', 'via boleto', 'via fatura']
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


--- RESULTADOS DO LAB 03 (AULA 03) ---

#========== PRODUÇÃO DO RELATÓRIO:==============

# 1 - Cole o código corrigido e a acurácia obtida.

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dados_rh = {
    'mensagem': [
        'Como solicitar minhas ferias?', 'Quero agendar meu periodo de ferias',
        'Queria marcar minhas férias para o próximo mês.', 'Como faço o pedido de férias?',
        'Preciso de informações sobre o processo de férias.', 'Qual é o prazo para solicitar as férias?',
        'Onde baixo meu holerite do mes?', 'Preciso do comprovante de rendimentos',
        'Não consigo ver meu holerite, podem me ajudar?', 'Gostaria de pegar meu contracheque',
        'Preciso do meu comprovante de pagamento.', 'Onde encontro os meus demonstrativos de pagamento?',
        'Como cadastrar meu atestado medico?', 'Onde envio o atestado de consulta?',
        'Tive uma consulta e preciso entregar o atestado.', 'Como envio o meu atestado médico para a empresa?',
        'Qual é o procedimento para envio de atestado?', 'Preciso justificar minha ausência com um atestado.'
    ],
    'intencao': [
        'solicitar_ferias', 'solicitar_ferias',
        'solicitar_ferias', 'solicitar_ferias',
        'solicitar_ferias', 'solicitar_ferias',
        'obter_holerite', 'obter_holerite',
        'obter_holerite', 'obter_holerite',
        'obter_holerite', 'obter_holerite',
        'enviar_atestado', 'enviar_atestado',
        'enviar_atestado', 'enviar_atestado',
        'enviar_atestado', 'enviar_atestado'
    ]
}

df3 = pd.DataFrame(dados_rh)


X = df3['mensagem']
y = df3['intencao']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=['de', 'o', 'meu', 'minhas'])),
    ('classifier', LogisticRegression())
])


pipeline.fit(X_train, y_train)


predicoes = pipeline.predict(X_test)
print(f"Acuracia via Pipeline: {accuracy_score(y_test, predicoes) * 100:.2f}%")


frase_teste_individual = 'cadê meu holerite?'
predicao_frase_teste = pipeline.predict([frase_teste_individual])
print(f"Predição para '{frase_teste_individual}': {predicao_frase_teste[0]}")

--- SAIDA DO CÓDIGO ---

Acuracia via Pipeline: 33.33%
Predição para 'cadê meu holerite?': obter_holerite

---------------------------------------------------

# 2 - Qual é a grande vantagem de utilizar o objeto Pipeline no Scikit-Learn?
o Pipeline permite que você monte seu código de forma resumida, escrevendo um código 
mais pequeno que funciona e mitiga chance de erros e evita vazamento de dados.


# 3 - Por que o Pipeline evita que erros de pré-processamento ocorram entre treino e teste?
o Pipeline isola as etapas de ajuste do pré-processamento aos dados de treino, evitando que 
características do conjunto de teste influenciem o treinamento do modelo e do pré-processador, 
o que levaria a uma avaliação superestimada do desempenho do modelo
