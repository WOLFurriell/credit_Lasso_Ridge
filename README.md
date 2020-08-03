# credit_Lasso_Ridge
Aplicação das regressões de Ridge, Lasso e Elastic net a modelos de crédito

O crédito impulsiona a capacidade de consumo e o poder de compra dos indivíduos, gerando com isso uma economia mais dinâmica e fluida. Desse modo, de forma bastante simplista, o chamado ciclo do crédito inicia quando a instituição financeira empresta capital ao consumidor e este o investe na economia, retornando a instituição o valor empretado mais juros, após um período acordado. 
A quebra do ciclo ocorre quando o indíviduo se torna inadimplente, isto é, não realizada o pagamento previsto de seus empréstimos, no período estipulado. Para mensurar o risco de inadimplência  as intituições financeiras se amparam em ferramentas de Credit e Behavior scoring, utilizadas para mensurar o risco de um indivíduo tornar-se inadimplente dada suas características econômicas e comportamentais no mercado, uma vez que, o grande volume de solicitações impede aprovações exclusivamente qualitativas.
Assim, um recurso bastante utilizado são os modelos de crédito quantitativos empregados comumente para metigar e predizer o risco de inadimplência, tais modelos em sua maioria são desenvolvidos com a metodologias estatística de regressão linear e principalmente logística, apesar do advento dos algortimos de Machine Learning, os modelos de principalmente de Regressào Logística ainda mostram-se ferramente de bastante interesse pela sua facilidade de interpretação e implementação.

# Regressão Logística comum



Sem perdas de generalidade podemos expressar um modelo de regressão da seguinte forma:
<a href="https://www.codecogs.com/eqnedit.php?latex=Y&space;\mid\mathbf{X}&space;&&space;\sim&space;f(\bm{\theta})&space;\\&space;Q(Y&space;\mid&space;\mathbf{X})&space;&&space;=&space;g(\mathbf{X}&space;\mid&space;\bm{\beta})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y&space;\mid\mathbf{X}&space;&&space;\sim&space;f(\bm{\theta})&space;\\&space;Q(Y&space;\mid&space;\mathbf{X})&space;&&space;=&space;g(\mathbf{X}&space;\mid&space;\bm{\beta})" title="Y \mid\mathbf{X} & \sim f(\bm{\theta}) \\ Q(Y \mid \mathbf{X}) & = g(\mathbf{X} \mid \bm{\beta})" /></a>
em que 
f(\bm{\theta}) $ é a f.d.p. de alguma distribuição indexada por um parâmetro  \bm{\theta} $;
Q(Y \mid \mathbf{X})  é alguma quantidade de interesse (média, quantil, parâmetro) de $ Y $ condiciada as covariáveis; e
g(\mathbf{X} \mid  \bm{\beta}) $ é uma função de ligação utilizada para associar as covariáveis com a quantidade de interesse.




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
