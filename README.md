# Aplicação das regressões de Ridge, Lasso e Elastic net a modelos de crédito

O crédito impulsiona a capacidade de consumo e o poder de compra dos indivíduos, gerando com isso uma economia mais dinâmica e fluida. Desse modo, de forma bastante simplista, o chamado ciclo do crédito inicia quando a instituição financeira empresta capital ao consumidor e este o investe na economia, retornando a instituição o valor empretado mais juros, após um período acordado. 
A quebra do ciclo ocorre quando o indíviduo se torna inadimplente, isto é, não realizada o pagamento previsto de seus empréstimos, no período estipulado. Para mensurar o risco de inadimplência  as intituições financeiras se amparam em ferramentas de Credit e Behavior scoring, utilizadas para mensurar o risco de um indivíduo tornar-se inadimplente dada suas características econômicas e comportamentais no mercado, uma vez que, o grande volume de solicitações impede aprovações exclusivamente qualitativas.
Assim, um recurso bastante utilizado são os modelos de crédito quantitativos empregados comumente para metigar e predizer o risco de inadimplência, tais modelos em sua maioria são desenvolvidos com a metodologias estatística de regressão linear e principalmente logística, apesar do advento dos algortimos de Machine Learning, os modelos de principalmente de Regressào Logística ainda mostram-se ferramente de bastante interesse pela sua facilidade de interpretação e implementação.

# Regressão Logística

Vamos relembrar alguns aspectos importantes do modelo de Regressão Logística, seja a variável resposta Y binária, temos:

- <img src="https://latex.codecogs.com/gif.latex?Y_i" title="Y_i" /> = 1 a ocorrência do evento de interesse;

- <img src="https://latex.codecogs.com/gif.latex?Y_i" title="Y_i" /> = 0 ausência do evento de interesse ou referência.

- <img src="https://latex.codecogs.com/gif.latex?X=(x_1,\ldots,&space;x_p)^\top" title="X=(x_1,\ldots, x_p)^\top" /> é um vetor de variáveis exploratórias, que podem ser discretas, continuas ou categóricas. De tal forma que, as variáveis categóricas quando não ordinais podem ser incorporadas ao modelo por meio da matriz dummy.

- Sendo a função de ligação Logit, o componente sistemático do modelo é dado por:

<a href="https://www.codecogs.com/eqnedit.php?latex=\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})}{1&plus;\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})},&space;\\&space;logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)=&space;\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})}{1&plus;\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})},&space;\\&space;logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)=&space;\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik}" title="\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik})}{1+\exp(\beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik})}, \\ logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)= \beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik}" /></a>

em que <img src="https://latex.codecogs.com/gif.latex?\pi_i" title="\pi_i" /> denota a probabilidade de ocorrência do evento de interesse.

- Os parâmetros estimados são obtidos a partir da função de Log-Verossimilhança, dada por:

<a href="https://www.codecogs.com/eqnedit.php?latex=l(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[y_i&space;log(\pi_i)&plus;(1-y_i)log(1-\pi_i))\right&space;]=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[y_i&space;log(\pi_i)&plus;(1-y_i)log(1-\pi_i))\right&space;]=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]" title="l(\beta) = \sum^{n}_{i=1}\left [y_i log(\pi_i)+(1-y_i)log(1-\pi_i))\right ]= \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ]" /></a>

As variâncias e covariâncias de <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\beta}" title="\boldsymbol{\beta}" /> são dadas pela Matriz de informação Fisher

<img src="https://latex.codecogs.com/gif.latex?-\frac{\partial^2\ell}{\partial\,\beta_j^2}&space;=&space;\sum_{i=1}^{n}\,x_{ij}^2\,\pi(x_i)\,(1&space;-&space;\pi(x_i))&space;\\&space;e&space;\\&space;-\frac{\partial^2\ell}{\partial\,\beta_j\,\beta_l}&space;=&space;\sum_{i=1}^{n}\,x_{ij}\,x_{il}\,\pi(x_i)\,(1&space;-&space;\pi(x_i))" title="-\frac{\partial^2\ell}{\partial\,\beta_j^2} = \sum_{i=1}^{n}\,x_{ij}^2\,\pi(x_i)\,(1 - \pi(x_i)) \\ e \\ -\frac{\partial^2\ell}{\partial\,\beta_j\,\beta_l} = \sum_{i=1}^{n}\,x_{ij}\,x_{il}\,\pi(x_i)\,(1 - \pi(x_i))" />

Sendo a inferência sobre os <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\beta}" title="\boldsymbol{\beta}" />'s dada pelo teste de Wald, no qual testamos a significância dos parâmetros estimados,<img src="https://latex.codecogs.com/gif.latex?\mathcal{H}_0:&space;\boldsymbol{\beta}&space;=&space;0" title="\mathcal{H}_0: \boldsymbol{\beta} = 0" />:

<img src="https://latex.codecogs.com/gif.latex?S_{W}&space;=&space;\left(\hat{\boldsymbol{\beta}}\right)^\top\,\left[I\left(\hat{\boldsymbol{\beta}}\right)\right]^{-1}&space;\,\left(\widehat{\boldsymbol{\beta}}\right)&space;\sim&space;\chi^2_{p&space;-&space;q}" title="S_{W} = \left(\hat{\boldsymbol{\beta}}\right)^\top\,\left[I\left(\hat{\boldsymbol{\beta}}\right)\right]^{-1} \,\left(\widehat{\boldsymbol{\beta}}\right) \sim \chi^2_{p - q}" />

Apesar de ser um algoritmo amplamente utilizado e bastante robusto, os modelos de regressão logística tem algumas limitações, principalmente quando precisamos considerar um volume grande de covariáveis para sua construção. Tal questão levanta problemas como o de multicolinearidades, isto é, variáveis preditoras com alto nível de correlação no modelo, resultando em inferências errôneas ou pouco confiáveis.
Outro ponto de atenção refere-se a seleção das covariáveis, na grande maioria dos casos a seleção é dada pelas técnicas de Stepwise, Forward, Backward ou mesmo pelo Teste da Razão de Verossimilhança, todas baseadas em testes de distribuições de probabilidade, que para amostras muito grandes ou com problemas de multicolinearidade tendem a ser a retornar resultados significativos em algum grau. Tais pontos podem levar a conclusões equivovados e por consequência incluir informações irrelevantes no processo, adicionando complexidade e interpretabilidade desnecessárias ao modelo, resultando em modelos hiperparametrizados com problemas de alta variabilidade e overfitting. 

# Regularização

De modo geral, quando a relação entre a função de ligação e uma resposta dicotômica com os preditores é aproximadamente linear, as estimativas de probabilidade terão um viés baixo, contudo, podem ter uma alta variação, quando o número de covariáveis for elevado comparado as observações ou quando existe multicolinearidade nos dados. Dessa forma, as regressões regularizadas podem ser empregadas trocando um pequeno aumento no viés por uma considerável diminuição na variação e consequentemente melhorando a precisão geral do modelo.

À vista disso, vamos discutir acerca dos modelos regularizados de Ridge, Lasso e Elastic Net. A solução destes algoritmos é adicionar hiperparâmetros regularizadores penalizando os parâmetros da regressão, auxiliando assim na diminuição da variância e do erro do modelo, garantindo a generalização efetiva dos resultados e com isso, regulando a entrada das covariáveis, através de pesos e penalização da função objetivo.

# Ridge

O estimador de Ridge depende da escolha do hiperparâmetro de tuning <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> > 0 que é acrescido ao EMV, assim temos:

<a href="https://www.codecogs.com/eqnedit.php?latex=l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}\beta^2_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}\beta^2_j" title="l^R_\lambda(\beta) = \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ] - \lambda \sum^{p}_{j=1}\beta^2_j" /></a>

À medida que a penalidade de <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> aumenta, as estimativas tendem a se aproximar de zero. Portanto, a regressão de Ridge tem a desvantagem sobre a seleção das covariáveis, uma vez que, inclui todas no modelo final. Logo, a interpretação do modelo quando o número de variáveis é grande torna-se problemática. 

# Lasso

O Lasso é outra alternativa de regularização que supera a desvantagem da regressão de Ridge de reduzir o número de preditores no modelo final. A versão penalizada da função de EMV assume a forma:

<a href="https://www.codecogs.com/eqnedit.php?latex=l^L_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}|\beta_j|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}|\beta_j|" title="l^R_\lambda(\beta) = \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ] - \lambda \sum^{p}_{j=1}|\beta_j|" /></a>

Comparando com a regressão de Ridge, a Lasso usa uma penalidade de L1 em vez de L2. O que permite a seleção de variáveis, uma vez que, quando <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> é suficientemente grande, algumas das estimativas podem ser exatamente iguais a zero. Desse modo, o Lasso tem a vantagem sobre a regressão de Ridge, já que o modelo final envolve apenas um subconjunto dos preditores, que, por sua vez, melhora a interpretabilidade do modelo. Destaca-se que tanto a regressão do Lasso quanto de Ridge, geralmente não se penalizam o intercepto. 

# Elastic Net

Outro método de regularização e seleção variável chamado Elastic Net, inclui um parâmetro de ajuste α ≥ 0, sendo a penalidade uma mistura das duas abordagens anteriore

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha\sum_{j=1}^{p}\beta^2_j&plus;(1-\alpha))\sum_{j=1}^{p}|\beta_j|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha\sum_{j=1}^{p}\beta^2_j&plus;(1-\alpha))\sum_{j=1}^{p}|\beta_j|" title="\alpha\sum_{j=1}^{p}\beta^2_j+(1-\alpha))\sum_{j=1}^{p}|\beta_j|" /></a>

Essa abordagem é particularmente útil quando o número de preditores é muito maior que o número de observações.

Escolher um bom valor de λ é uma etapa crítica para os algoritmos por isso é importante a realização de tuning dos hiperparâmetros juntamento com a avaliação do ajuste dos modelos.
