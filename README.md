# Regressões Logísticas de Ridge, Lasso e Elastic-Net

Um recurso bastante utilizado na análise de crédito, são os modelos quantitativos, empregados comumente para mitigar e predizer o risco de inadimplência dos clientes. Tais modelos, em sua maioria, são desenvolvidos com as metodologias de regressão, principalmente a Logística Binária e visam modelar a probabilidade de um evento ocorrer, segundo um conjunto de fatores. 
Apesar do advento dos algoritmos de Machine Learning, os modelos Logísticos ainda ocupam lugar de bastante relevância no mercado e na academia, pela sua facilidade de interpretação e implementação. Visando avançar na utilização deste tipo de metodologia, vamos expor de forma breve algumas características da Regressão Logística Binária, suas limitações e alguns modelos alternativos baseados em técnicas de regularização para contorná-las.

# Regressão Logística Binária

Tomando como apoio um trabalho anteriormente realizado sobre Modelos de Regressão Logística Binária: https://github.com/WOLFurriell/RegBin/tree/master. 

Seja a variável resposta Y binária, temos:

- <img src="https://latex.codecogs.com/gif.latex?Y_i" title="Y_i" /> = 1 a ocorrência do evento de interesse;

- <img src="https://latex.codecogs.com/gif.latex?Y_i" title="Y_i" /> = 0 ausência do evento de interesse ou referência.

- <img src="https://latex.codecogs.com/gif.latex?X=(x_1,\ldots,&space;x_p)" title="X=(x_1,\ldots, x_p)" /> é um vetor de variáveis exploratórias, que podem ser discretas, continuas ou categóricas. De tal forma que, as variáveis categóricas quando não ordinais podem ser incorporadas ao modelo por meio da matriz dummy.

- Dado uma função de ligação Logit, o componente sistemático do modelo é dado por:

<a href="https://www.codecogs.com/eqnedit.php?latex=\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})}{1&plus;\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})},&space;\\&space;logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)=&space;\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})}{1&plus;\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})},&space;\\&space;logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)=&space;\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik}" title="\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik})}{1+\exp(\beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik})}, \\ logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)= \beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik}" /></a>

em que <img src="https://latex.codecogs.com/gif.latex?\pi_i" title="\pi_i" /> denota a probabilidade de ocorrência do evento de interesse. Ainda, sendo um modelo probabilístico alguns pressupostos devem ser considerados como relação linear entre as covariáveis a target e inexistência de multicolinearidade.

- Os parâmetros estimados são obtidos a partir da função de Log-Verossimilhança, dada por:

<a href="https://www.codecogs.com/eqnedit.php?latex=l(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[y_i&space;log(\pi_i)&plus;(1-y_i)log(1-\pi_i))\right&space;]=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[y_i&space;log(\pi_i)&plus;(1-y_i)log(1-\pi_i))\right&space;]=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]" title="l(\beta) = \sum^{n}_{i=1}\left [y_i log(\pi_i)+(1-y_i)log(1-\pi_i))\right ]= \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ]" /></a>

As variâncias e covariâncias de <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\beta}" title="\boldsymbol{\beta}" /> são dadas pela Matriz informação de Fisher

<img src="https://latex.codecogs.com/gif.latex?-\frac{\partial^2\ell}{\partial\,\beta_j^2}&space;=&space;\sum_{i=1}^{n}\,x_{ij}^2\,\pi(x_i)\,(1&space;-&space;\pi(x_i))&space;\\&space;e&space;\\&space;-\frac{\partial^2\ell}{\partial\,\beta_j\,\beta_l}&space;=&space;\sum_{i=1}^{n}\,x_{ij}\,x_{il}\,\pi(x_i)\,(1&space;-&space;\pi(x_i))" title="-\frac{\partial^2\ell}{\partial\,\beta_j^2} = \sum_{i=1}^{n}\,x_{ij}^2\,\pi(x_i)\,(1 - \pi(x_i)) \\ e \\ -\frac{\partial^2\ell}{\partial\,\beta_j\,\beta_l} = \sum_{i=1}^{n}\,x_{ij}\,x_{il}\,\pi(x_i)\,(1 - \pi(x_i))" />

Sendo a inferência sobre os <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\beta}" title="\boldsymbol{\beta}" />'s dada pelo teste de Wald, no qual, testamos a significância dos parâmetros estimados, <img src="https://latex.codecogs.com/gif.latex?\mathcal{H}_0:&space;\boldsymbol{\beta}&space;=&space;0" title="\mathcal{H}_0: \boldsymbol{\beta} = 0" /> e desejamos rejeitar a hipótese nula:

<img src="https://latex.codecogs.com/gif.latex?S_{W}&space;=&space;\left(\hat{\boldsymbol{\beta}}\right)^\top\,\left[I\left(\hat{\boldsymbol{\beta}}\right)\right]^{-1}&space;\,\left(\widehat{\boldsymbol{\beta}}\right)&space;\sim&space;\chi^2_{p&space;-&space;q}" title="S_{W} = \left(\hat{\boldsymbol{\beta}}\right)^\top\,\left[I\left(\hat{\boldsymbol{\beta}}\right)\right]^{-1} \,\left(\widehat{\boldsymbol{\beta}}\right) \sim \chi^2_{p - q}" />

Apesar de ser um algoritmo amplamente utilizado e bastante robusto, os modelos de Regressão Logística tem algumas limitações, principalmente quando precisamos considerar um volume grande de covariáveis para sua construção. Tal questão levanta problemas como o de multicolinearidade, isto é, variáveis preditoras com alto nível de correlação, resultando em inferências errôneas ou pouco confiáveis.
Outro ponto de atenção refere-se a seleção das covariáveis, na grande maioria dos casos a seleção é dada pelas técnicas de Stepwise, Forward, Backward ou mesmo pelo Teste da Razão de Verossimilhança, todas baseadas em testes de distribuição de probabilidade, que para amostras muito grandes ou com problemas de multicolinearidade tendem a retornar resultados significativos em algum grau. Tais pontos podem levar a conclusões equivocadas e por consequência incluir informações irrelevantes no processo, adicionando complexidade e interpretabilidade desnecessárias, resultando em modelos hiperparametrizados com problemas de alta variabilidade e overfitting. 

# Regularização

As técnicas de regularização podem ser empregadas nos modelos trocando um pequeno aumento no viés por uma considerável diminuição na variação dos resultados e consequentemente melhorando a precisão geral, principalmente quando o número de covariáveis é elevado, ou mesmo, quando existe multicolinearidade nos dados. À vista disso, vamos discutir acerca de algumas técnicas de regularização aplicadas aos modelos conhecidos como Ridge, Lasso e Elastic Net. A solução destes algoritmos é adicionar hiperparâmetros regularizadores, penalizando os parâmetros da regressão e auxiliando na diminuição da variância e do erro do modelo, garantindo a generalização efetiva dos resultados, regulando a entrada das covariáveis, através de pesos e penalização da função objetivo. Além disso, não é possível obter os erros padrão das estimativas baseados nas propriedades de Máxima Verossimilhança, uma vez que, a penalização realiza maximiza um espaço com restrições.

# Ridge

O estimador de Ridge depende da escolha do hiperparâmetro de tuning <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> > 0 que é acrescido ao EMV, assim temos:

<a href="https://www.codecogs.com/eqnedit.php?latex=l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}\beta^2_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}\beta^2_j" title="l^R_\lambda(\beta) = \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ] - \lambda \sum^{p}_{j=1}\beta^2_j" /></a>

À medida que a penalidade de <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> aumenta, as estimativas tendem a se aproximar de zero. Portanto, a regressão de Ridge tem a desvantagem sobre a seleção das covariáveis, uma vez que, inclui todas no modelo final. Logo, a interpretação do mesmo quando o número de variáveis é grande torna-se problemática. 

# Lasso

O Lasso é outra alternativa de regularização que supera a desvantagem da regressão de Ridge ao reduzir o número de preditores no modelo final. A versão penalizada da função de EMV assume a forma:

<a href="https://www.codecogs.com/eqnedit.php?latex=l^L_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}|\beta_j|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}|\beta_j|" title="l^R_\lambda(\beta) = \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ] - \lambda \sum^{p}_{j=1}|\beta_j|" /></a>

Comparando com a regressão de Ridge, a Lasso usa uma penalidade de L1 em vez de L2. O que permite a seleção de variáveis, uma vez que, quando <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> é suficientemente grande, algumas das estimativas podem ser exatamente iguais a zero. Desse modo, o Lasso tem a vantagem sobre a regressão de Ridge, já que o modelo final envolve apenas um subconjunto dos preditores, o que, melhora a interpretabilidade do modelo. Destaca-se que tanto a regressão do Lasso quanto de Ridge, geralmente não se penalizam o intercepto. 

# Elastic Net

Outro método de regularização e seleção de covariáveis é chamado de Elastic Net, que inclui um parâmetro de ajuste <a href="https://www.codecogs.com/eqnedit.php?latex=0&space;\leq&space;\alpha&space;\leq&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?0&space;\leq&space;\alpha&space;\leq&space;1" title="0 \leq \alpha \leq 1" /></a>, sendo a penalidade uma mistura das duas abordagens anteriormente expostas:

<a href="https://www.codecogs.com/eqnedit.php?latex=l^E_\alpha(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda\left&space;[\alpha\sum_{j=1}^{p}\beta^2_j&plus;(1-\alpha))\sum_{j=1}^{p}|\beta_j|&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l^E_\alpha(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda\left&space;[\alpha\sum_{j=1}^{p}\beta^2_j&plus;(1-\alpha))\sum_{j=1}^{p}|\beta_j|&space;\right&space;]" title="l^E_\alpha(\beta) = \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ] - \lambda\left [\alpha\sum_{j=1}^{p}\beta^2_j+(1-\alpha))\sum_{j=1}^{p}|\beta_j| \right ]" /></a>

Esse modelo é bastante útil quando o número de preditores(p) é muito maior que o número de observações(n), isto é, p > n. 

Para os três algoritmos expostos escolher um bom valor de <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> é uma etapa de extrema importância, por isso a realização de tuning deste hiperparâmetro, juntamento com a avaliação do ajuste dos modelos é essencial.

# Aplicação 

Para testar os algoritmos foi utilizada a base BlogFeedback Data Set, disponível no UCI Machine Learning Center: https://archive.ics.uci.edu/ml/datasets/BlogFeedback#. Os resultados foram obtidos com auxílio do software R.
A base conta com 281 variáveis, um volume relativamente alto e passível de seleção. No estudo, comparamos os modelos de Ridge, Lasso, Elastic-Net e Logit sem penalização. No caso dos modelos penalizados, foi realizado o Tuning do parâmetro <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> e de <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> para o Elastic-Net. Quanto ao Logit sem penalização, foi empregado o processo de Stepwise para seleção de variáveis.

No gráfico abaixo temos o tuning do parâmetro <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> para os modeloa de Ridge, Lasso e Elastic-Net, visando a maximização da medida de AUC. Tal processo, permite verificar o conjunto de informações que retornam a melhor discriminação para o modelo.

<img align="center" width="950" height="300"  src="https://github.com/WOLFurriell/credit_Lasso_Ridge/blob/master/plots/ggauc.png">

No que tange o diagnóstico, verificamos a curva ROC, bem como, a medida de AUC, desse modo, avaliamos que os resultados foram bastante similares, sendo o Lasso, o modelo que apresentou a melhor performace. Contudo, o modelo Logit sem penalização, apesar do volume de variáveis, demonstrou uma boa capacidade de discriminação e generalização dos resultados, quando avaliamos as bases de Teste de Validação. 

<img align="center" width="1000" height="450"  src="https://github.com/WOLFurriell/credit_Lasso_Ridge/blob/master/plots/roc0.png">

É válido ressaltar que na aplicação exposta, não ocorreu o problema de p > n, isto é, o volume de variáveis superiores ao de observações. Ponto de bastante destaque para utilização dos modelos penalizados. Além disso, em volumes ainda maiores de variáveis apenalização pode ser utilizada como um bom método para seleção de variáveis.

# Referências 

- Tibshirani, R. (1997). The LASSO method for variable selection in the Cox model. Statistics in Medicine 16 (4), 385–395

- Pereira, Jose Manuel, Mario Basto, and Amelia Ferreira da Silva. "The logistic lasso and ridge regression in predicting corporate failure." Procedia Economics and Finance 39 (2016): 634-641.

- Friedman, Jerome, Trevor Hastie, and Robert Tibshirani. The elements of statistical learning. Vol. 1. No. 10. New York: Springer series in statistics, 2001.

- Penalized Logistic Regression Essentials in R: Ridge, Lasso and Elastic Net: http://www.sthda.com/english/articles/36-classification-methods-essentials/149-penalized-logistic-regression-essentials-in-r-ridge-lasso-and-elastic-net/#:~:text=Penalized%20Logistic%20Regression%20Essentials%20in%20R%3A%20Ridge%2C%20Lasso%20and%20Elastic%20Net,-kassambara%20%7C%2011%2F03&text=Penalized%20logistic%20regression%20imposes%20a,is%20also%20known%20as%20regularization.

- Regularização em regressão com implementação: http://leg.ufpr.br/~walmes/ensino/ML/tutorials/03-regularization-2.html
