# Aplicação das regressões de Ridge, Lasso e Elastic net a modelos de crédito

O crédito impulsiona a capacidade de consumo e o poder de compra dos indivíduos, gerando com isso uma economia mais dinâmica e fluida. Desse modo, de forma bastante simplista, o chamado ciclo do crédito inicia quando a instituição financeira empresta capital ao consumidor e este o investe na economia, retornando a instituição o valor empretado mais juros, após um período acordado. 
A quebra do ciclo ocorre quando o indíviduo se torna inadimplente, isto é, não realizada o pagamento previsto de seus empréstimos, no período estipulado. Para mensurar o risco de inadimplência  as intituições financeiras se amparam em ferramentas de Credit e Behavior scoring, utilizadas para mensurar o risco de um indivíduo tornar-se inadimplente dada suas características econômicas e comportamentais no mercado, uma vez que, o grande volume de solicitações impede aprovações exclusivamente qualitativas.
Assim, um recurso bastante utilizado são os modelos de crédito quantitativos empregados comumente para metigar e predizer o risco de inadimplência, tais modelos em sua maioria são desenvolvidos com a metodologias estatística de regressão linear e principalmente logística, apesar do advento dos algortimos de Machine Learning, os modelos de principalmente de Regressào Logística ainda mostram-se ferramente de bastante interesse pela sua facilidade de interpretação e implementação.

# Regressão Logística comum

Vamos relembrar alguns aspectos importantes do modelo de Regressão Logística, seja a variável resposta Y binária, temos:

- <img src="https://latex.codecogs.com/gif.latex?Y_i" title="Y_i" /> = 1 a ocorrência do evento de interesse;

- <img src="https://latex.codecogs.com/gif.latex?Y_i" title="Y_i" /> = 0 ausência do evento de interesse ou referência.

- <img src="https://latex.codecogs.com/gif.latex?X=(x_1,\ldots,&space;x_p)^\top" title="X=(x_1,\ldots, x_p)^\top" /> é um vetor de variáveis exploratórias, que podem ser discretas, continuas ou categóricas. De tal forma que, as variáveis categóricas quando não ordinais podem ser incorporadas ao modelo por meio da matriz dummy.

- O componente sistemático do modelo é dado por:

<a href="https://www.codecogs.com/eqnedit.php?latex=\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})}{1&plus;\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})},&space;\\&space;logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)=&space;\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})}{1&plus;\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})},&space;\\&space;logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)=&space;\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik}" title="\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik})}{1+\exp(\beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik})}, \\ logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)= \beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik}" /></a>

em que <img src="https://latex.codecogs.com/gif.latex?\pi_i" title="\pi_i" /> denota a probabilidade de ocorrência do evento de interesse.

- Os parâmetros estimados são obtidos a partir da função de Máxima Verossimilhança, dada por:

<a href="https://www.codecogs.com/eqnedit.php?latex=l(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[y_i&space;log(\pi_i)&plus;(1-y_i)log(1-\pi_i))\right&space;]=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[y_i&space;log(\pi_i)&plus;(1-y_i)log(1-\pi_i))\right&space;]=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]" title="l(\beta) = \sum^{n}_{i=1}\left [y_i log(\pi_i)+(1-y_i)log(1-\pi_i))\right ]= \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ]" /></a>

Apesar de ser um algoritmo amplamente utilizado, os modelos de regressão logística tem algumas limitações, principalmente quando precisamos considerar um volume bastante grande de covariáveis para sua construção. Tal questão levanta problemas como o de multicolinearidades, isto é, variáveis preditoras com alto nível de correlação no modelo resultando em inferências errôneas ou pouco confiáveis.
Outro ponto de atenção refere-se a seleção das variáveis preditoras, na grande maioria dos casos a seleção é dada pelas técnicas de Stepwise, Forward, Backward ou mesmo o Teste da Razão de Verossimilhança algoritmos baseados em testes de distribuições de probabilidade, que para amostras muito grandes tendem a ser significativos em algum grau. 

A tarefa de determinar quais preditores estão associados a variável target não é simples, ao selecionar as variáveis para um modelo, geralmente se observa p-value individual, contudo se as covariáveis estiverem altamente correlacionadas, os p-values tendem a ser significativos, levando a resultados equivovados e por consequência incluindo informações irrelevantes no processo, adicionando complexidade e interpretabilidade desnecessárias. Além disso, se o número de observações não for muito maior que o de covariáveis, ou mesmo, modelos hiperparametrizados podem ocorrer problemas de alta variabilidade, resultando em overfitting. 

# Regularização

De modo geral, quando a relação entre a função de ligação e uma resposta dicotômica com os preditores é aproximadamente linear, as estimativas de probabilidade terão um viés baixo, contudo podem ter uma alta variação, quando o número de covariáveis for elevado comparado as observações ou quando existe multicolinearidade nos dados. Dessa forma, as regressões regularizadas podem ser empregadas trocando um pequeno aumento no viés por uma considerável diminuição na variação e consequentemente melhorando a precisão geral do modelo.

Com isso para contornar as limitacões elencados vamos discutir acerca do Ridge, Lasso e Elastic Net. A solução destes algoritmos é adicionar hiperparâmetros regularizadores penalizando os parâmetros da regressão, auxiliando assim na diminuição da variância e do erro do modelo, garantindo a generalização efetiva e com isso, regulando a entrada das covariáveis, através de pesos, impondo uma penalidade na função objetivo.

# Ridge

O estimador de Ridge depende da escolha do hiperparâmetro de tuning <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> > 0 que é acrescido ao estimador de MV, assim temos:

<a href="https://www.codecogs.com/eqnedit.php?latex=l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}\beta^2_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}\beta^2_j" title="l^R_\lambda(\beta) = \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ] - \lambda \sum^{p}_{j=1}\beta^2_j" /></a>

Para qualquer valor fixo de λ, a regressão de Ridge se encaixa apenas em um único modelo, que pode ser executado rapidamente. À medida que a penalidade de contração λ aumenta, as estimativas do coeficiente tendem a se aproximar de zero. No entanto, a penalidade introduzida na função de probabilidade de log reduzirá todos os coeficientes para zero, mas não definirá nenhum deles exatamente como zero. Portanto, a regressão de Ridge tem a desvantagem sobre a seleção do modelo, de incluir todos os preditores no modelo final. Dessa forma, a interpretação do modelo quando o número de variáveis é grande acaba sendo mais problemática. 

# Lasso

O Lasso é outra alternativa de regularização que supera a desvantagem da incapacidade da regressão de Ridge de reduzir o número de preditores no modelo final. A versão penalizada da função de probabilidade de log a ser maximizada assume agora a forma:

<a href="https://www.codecogs.com/eqnedit.php?latex=l^L_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}|\beta_j|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}|\beta_j|" title="l^R_\lambda(\beta) = \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ] - \lambda \sum^{p}_{j=1}|\beta_j|" /></a>

Comparando com a regressão de Ridge, o Lasso usa uma penalidade de L1 em vez de L2. A penalidade de L1 usada é usada para seleção de variáveis, uma vez que tem o efeito quando λ é suficientemente grande, de forçar algumas das estimativas do coeficiente a serem exatamente iguais a zero. O Lasso tem uma vantagem sobre a regressão da Ridge, já que o modelo final pode envolver apenas um subconjunto dos preditores, o que, por sua vez, melhora a interpretabilidade. Com relação à precisão da previsão, geralmente quando apenas um pequeno número de preditores possui coeficientes substanciais, pode-se esperar um desempenho melhor do Lasso, enquanto quando todos os coeficientes têm aproximadamente o mesmo tamanho, espera-se um melhor desempenho da regressão da Ridge. Tanto para a regressão do Lasso quanto de Ridge, geralmente não se penalizam o intercepto. 

# Elastic Net

Outro método de regularização e seleção variável chamado Elastic Net, inclui um parâmetro de ajuste α ≥ 0, sendo a penalidade uma mistura das duas abordagens anteriore

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha\sum_{j=1}^{p}\beta^2_j&plus;(1-\alpha))\sum_{j=1}^{p}|\beta_j|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha\sum_{j=1}^{p}\beta^2_j&plus;(1-\alpha))\sum_{j=1}^{p}|\beta_j|" title="\alpha\sum_{j=1}^{p}\beta^2_j+(1-\alpha))\sum_{j=1}^{p}|\beta_j|" /></a>

Essa abordagem é particularmente útil quando o número de preditores é muito maior que o número de observações.

Escolher um bom valor de λ é uma etapa crítica para os algoritmos por isso é importante a realização de tuning dos hiperparâmetros juntamento com a avaliação do ajuste dos modelos.
