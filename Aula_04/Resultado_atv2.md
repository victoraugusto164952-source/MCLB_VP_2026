# ==============================================================================
# ATIVIDADE 2: CHATBOT VERSÃO 2 (DECISION TREE) - RESOLUÇÃO COMPLETA
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Carregar o CSV
df = pd.read_csv('dataset_moveis_100.csv')

# 2. Divisão estratificada de treino/teste (30% teste)
X_train, X_test, y_train, y_test = train_test_split(
    df['texto'], 
    df['intencao'], 
    test_size=0.30, 
    random_state=42, 
    stratify=df['intencao']
)

# 3. Construir e treinar a Pipeline utilizando TF-IDF + DecisionTreeClassifier
pipeline_dt = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Treinamento do modelo
pipeline_dt.fit(X_train, y_train)

# 4. Calcular e imprimir Matriz de Confusão e Relatório de Classificação
y_pred = pipeline_dt.predict(X_test)

print("\n" + "="*60)
print("=== MATRIZ DE CONFUSÃO ===")
print("="*60)
print(confusion_matrix(y_test, y_pred))

print("\n" + "="*60)
print("=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print("="*60)
print(classification_report(y_test, y_pred))

# Configuração do limiar de confiança para o Fallback
LIMIAR_CONFIANCA = 0.50

# 5. Estrutura interativa (8 frases via input)
print("\n" + "="*60)
print("=== INICIANDO BATERIA DE TESTES (8 INPUTS OBRIGATÓRIOS) ===")
print("="*60)

for i in range(1, 9):
    print(f"\n[Teste {i}/8]")
    
    # Solicitação manual da frase
    frase = input("Digite a frase do cliente: ").strip()
    
    # Tratamento para string vazia (força o fallback de segurança)
    if not frase:
        print("Desculpe, não entendi sua solicitação. Encaminhando você para um atendente humano...")
        continue
    
    # Extrair probabilidades e classe prevista
    probs = pipeline_dt.predict_proba([frase])
    maior_prob = np.max(probs)
    intencao = pipeline_dt.predict([frase])[0]
    
    # 6. Lógica de Fallback (Limiar = 50%)
    if maior_prob >= LIMIAR_CONFIANCA:
        print(f"Intenção: '{intencao}' detectada com {maior_prob * 100:.1f}% de certeza.")
    else:
        print("Desculpe, não entendi sua solicitação. Encaminhando você para um atendente humano...")

SAIDA


============================================================
=== MATRIZ DE CONFUSÃO ===
============================================================
[[4 0 0 0 2]
 [1 4 1 0 0]
 [0 0 6 0 0]
 [0 0 1 5 0]
 [0 0 0 1 5]]

============================================================
=== RELATÓRIO DE CLASSIFICAÇÃO ===
============================================================
                    precision    recall  f1-score   support

logistica_entregas       0.80      0.67      0.73         6
       reclamacoes       1.00      0.67      0.80         6
           suporte       0.75      1.00      0.86         6
 trocas_devolucoes       0.83      0.83      0.83         6
            vendas       0.71      0.83      0.77         6

          accuracy                           0.80        30
         macro avg       0.82      0.80      0.80        30
      weighted avg       0.82      0.80      0.80        30


============================================================
=== INICIANDO BATERIA DE TESTES (8 INPUTS OBRIGATÓRIOS) ===
============================================================

[Teste 1/8]
Digite a frase do cliente: meu celular quebrou
Intenção: 'logistica_entregas' detectada com 100.0% de certeza.

[Teste 2/8]
Digite a frase do cliente: onde está meu produto?
Intenção: 'logistica_entregas' detectada com 100.0% de certeza.

[Teste 3/8]
Digite a frase do cliente: qual cor tem do celular
Intenção: 'logistica_entregas' detectada com 100.0% de certeza.

[Teste 4/8]
Digite a frase do cliente: quanto est´o guarda roupa?
Intenção: 'trocas_devolucoes' detectada com 100.0% de certeza.

[Teste 5/8]
Digite a frase do cliente: vim devolver isso
Intenção: 'trocas_devolucoes' detectada com 100.0% de certeza.

[Teste 6/8]
Digite a frase do cliente: meu pedido está como?
Intenção: 'logistica_entregas' detectada com 100.0% de certeza.

[Teste 7/8]
Digite a frase do cliente: preciso arrumar meu computador
Intenção: 'logistica_entregas' detectada com 100.0% de certeza.

[Teste 8/8]
Digite a frase do cliente: 
Desculpe, não entendi sua solicitação. Encaminhando você para um atendente humano...
