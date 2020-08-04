# credit_Lasso_Ridge
Aplicação das regressões de Ridge, Lasso e Elastic net a modelos de crédito

O crédito impulsiona a capacidade de consumo e o poder de compra dos indivíduos, gerando com isso uma economia mais dinâmica e fluida. Desse modo, de forma bastante simplista, o chamado ciclo do crédito inicia quando a instituição financeira empresta capital ao consumidor e este o investe na economia, retornando a instituição o valor empretado mais juros, após um período acordado. 
A quebra do ciclo ocorre quando o indíviduo se torna inadimplente, isto é, não realizada o pagamento previsto de seus empréstimos, no período estipulado. Para mensurar o risco de inadimplência  as intituições financeiras se amparam em ferramentas de Credit e Behavior scoring, utilizadas para mensurar o risco de um indivíduo tornar-se inadimplente dada suas características econômicas e comportamentais no mercado, uma vez que, o grande volume de solicitações impede aprovações exclusivamente qualitativas.
Assim, um recurso bastante utilizado são os modelos de crédito quantitativos empregados comumente para metigar e predizer o risco de inadimplência, tais modelos em sua maioria são desenvolvidos com a metodologias estatística de regressão linear e principalmente logística, apesar do advento dos algortimos de Machine Learning, os modelos de principalmente de Regressào Logística ainda mostram-se ferramente de bastante interesse pela sua facilidade de interpretação e implementação.

# Regressão Logística comum

Seja a variável resposta $Y$ binária tal que
Y_i = 1 a ocorrência do evento de interesse (Evento);
Y_i = 0 ausência do evento de interesse (Referência).
\mathbf{X}=(X_1,\ldots, X_k)^\top$ é um vetor de variáveis exploratórias, que podem ser discretas, continuas ou categóricas;
As variáveis categóricas são incorporadas ao modelo por meio de matrizes de variáveis \emph{dummy.}

<a href="https://www.codecogs.com/eqnedit.php?latex=\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})}{1&plus;\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})},&space;\\&space;logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)=&space;\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})}{1&plus;\exp(\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik})},&space;\\&space;logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)=&space;\beta_0&plus;\beta_1&space;x_{i1}&space;&plus;&space;...&space;&plus;&space;\beta_k&space;x_{ik}" title="\pi_i=Pr(Y_i=1|X_i=x_i)=\frac{\exp(\beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik})}{1+\exp(\beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik})}, \\ logit(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)= \beta_0+\beta_1 x_{i1} + ... + \beta_k x_{ik}" /></a>

em que [latex]\pi_i[/latex] denota a probabilidade de ocorrência do evento de interesse.

Os dados $y_1, y_2, ..., y_n$ são i.i.d. 
$Y_i \sim \mathcal{B}(\pi_i)$ isto é, assume a distribuição bernoulli da resposta.
Não existe uma relação linear entre $Y_i$ e $\boldsymbol{X}$, mas sim, entre a função de ligação e o preditor linear.
Não homogeneidade da variância..
Os erros precisam ser independentes, mas não normalmente distribuídos.
As medidas de qualidade do modelo dependem de amostras suficientemente grandes, evitando assim, subpopulações muito pequenas.
	


Os parâmetros estimados são obtidos a partir da função de Máxima Verossimilhança, dada por:

<a href="https://www.codecogs.com/eqnedit.php?latex=l(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[y_i&space;log(\pi_i)&plus;(1-y_i)log(1-\pi_i))\right&space;]=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[y_i&space;log(\pi_i)&plus;(1-y_i)log(1-\pi_i))\right&space;]=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]" title="l(\beta) = \sum^{n}_{i=1}\left [y_i log(\pi_i)+(1-y_i)log(1-\pi_i))\right ]= \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ]" /></a>

# Ridge

O estimador de Ridge depende da escolha do hiperparâmetro de tuning \lambda > 0 que é acrescido ao estimador de MV, assim temos:

<a href="https://www.codecogs.com/eqnedit.php?latex=l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}\beta^2_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}\beta^2_j" title="l^R_\lambda(\beta) = \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ] - \lambda \sum^{p}_{j=1}\beta^2_j" /></a>

# Lasso

<a href="https://www.codecogs.com/eqnedit.php?latex=l^L_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}|\beta_j|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l^R_\lambda(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]&space;-&space;\lambda&space;\sum^{p}_{j=1}|\beta_j|" title="l^R_\lambda(\beta) = \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ] - \lambda \sum^{p}_{j=1}|\beta_j|" /></a>

# Elastic Net.

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha\sum_{j=1}^{p}\beta^2_j&plus;(1-\alpha))\sum_{j=1}^{p}|\beta_j|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha\sum_{j=1}^{p}\beta^2_j&plus;(1-\alpha))\sum_{j=1}^{p}|\beta_j|" title="\alpha\sum_{j=1}^{p}\beta^2_j+(1-\alpha))\sum_{j=1}^{p}|\beta_j|" /></a>


For  both  lasso  and  ridge  regression,  generally  one  do  not  penalize  the  intercept  term,  and  standardize  the  predictors for the penalty to be meaningful (Hastie et al., 2009). Another  regularization  and  variable  selection  method  proposed  by  Zou  and  Hastie  (2005),  called  elastic  net,  includes a tuning parameter α≥ 0, being the penalty a mixture of the previous two approaches: 
