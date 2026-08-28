 1. Desempenho dos ModelosKNN (K=3): 100% de Acurácia e F1-Score. Classificou perfeitamente todas as amostras de teste, sem nenhuma confusão.Decision Tree: 80% de Acurácia (F1: 79.7%).
 2.  Errou ao confundir as intenções de vendas, logistica_entregas, reclamacoes e suporte.

 3. 2. Comportamento nos Testes Reais (input())KNN: Mostrou-se estável e seguro. Interpretou variações de frases por proximidade semântica. O Fallback funcionou perfeitamente (confiança caiu em frases estranhas).
    3. Decision Tree: Mostrou-se instável e com falsa certeza. Classificou frases erradas com 100% de confiança por causa de palavras isoladas (como "com" ou "o"),
    4. quebrando a lógica do Fallback.
   
    5. 3. Veredito: KNN (K=3) venceuO KNN é o melhor modelo para este projeto. Ele analisa o contexto geral do texto (via métrica cosine) em vez de regras rígidas de palavras soltas.
       4.  Além disso, gera probabilidades reais que fazem o limiar de Fallback funcionar com segurança em produção.
