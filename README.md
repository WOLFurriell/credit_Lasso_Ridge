# credit_Lasso_Ridge
Aplicação das regressões de Ridge, Lasso e Elastic net a modelos de crédito


Os parâmetros estimados são obtidos a partir da função de Máxima Verossimilhança, dada por:

<a href="https://www.codecogs.com/eqnedit.php?latex=l(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[y_i&space;log(\pi_i)&plus;(1-y_i)log(1-\pi_i))\right&space;]=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l(\beta)&space;=&space;\sum^{n}_{i=1}\left&space;[y_i&space;log(\pi_i)&plus;(1-y_i)log(1-\pi_i))\right&space;]=&space;\sum^{n}_{i=1}\left&space;[&space;y_ix_i\beta-log(1&plus;e^{x_i\beta}))&space;\right&space;]" title="l(\beta) = \sum^{n}_{i=1}\left [y_i log(\pi_i)+(1-y_i)log(1-\pi_i))\right ]= \sum^{n}_{i=1}\left [ y_ix_i\beta-log(1+e^{x_i\beta})) \right ]" /></a>


O estimador de Ridge depende da escolha do hiperparâmetro de tuning <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" /></a>

The  logistic  ridge  regression  estimator  depends  on  the  choice  of  a  tuning  parameter  λ≥  0,  to  be  determined  separately.  The  coefficients  estimates  are  the  values  that  maximize  the  following  slightly  different  log-likelihood function  where  aL2ridge  penalty  is  added  to  the  function  (Duffy  and  Santner,  1989;  Cessie  and  Houwelingen,  1992), resulting in the following constrained maximization equation:
