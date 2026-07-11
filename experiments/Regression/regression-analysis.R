library(dplyr)


df <- "master_results.csv" |>
  read.csv() |>
  select(train_strategy,
         test_strategy,
         augmentation,
         in_domain,
         n_train_simulations,
         relative_MAE_I) |>
  rename(train = train_strategy,
         test = test_strategy,
         aug = augmentation,
         train_size = n_train_simulations,
         y = relative_MAE_I) |>
  mutate(train = as.factor(train),
         test = as.factor(test),
         aug = as.factor(aug))

lm(formula = y ~ train + test + aug, data = df) |> summary() |> print()
## Call:
## lm(formula = y ~ train + test + aug, data = df)
## Residuals:
##      Min       1Q   Median       3Q      Max
## -1.72948 -0.66712 -0.09346  0.62780  1.90840
## Coefficients:
##                     Estimate Std. Error t value Pr(>|t|)
## (Intercept)          7.93447    0.17171  46.208  < 2e-16 ***
## trainMCMC           -1.06997    0.17722  -6.037 1.45e-08 ***
## trainUNIFORM_RANDOM -0.08825    0.21182  -0.417   0.6776
## testMCMC             0.34568    0.17722   1.951   0.0532 .
## testUNIFORM_RANDOM  -0.29603    0.21182  -1.398   0.1646
## aug1                -2.00399    0.13867 -14.452  < 2e-16 ***
## ---
## Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
## Residual standard error: 0.8204 on 134 degrees of freedom
## Multiple R-squared:  0.6818,	Adjusted R-squared:  0.6699
## F-statistic: 57.42 on 5 and 134 DF,  p-value: < 2.2e-16


lm(formula = y ~ in_domain + test + aug, data = df) |> summary() |> print()
## Call:
## lm(formula = y ~ in_domain + test + aug, data = df)
## Residuals:
##     Min      1Q  Median      3Q     Max
## -1.7295 -0.6514 -0.1082  0.6345  1.8643
## Coefficients:
##                    Estimate Std. Error t value Pr(>|t|)
## (Intercept)          7.9124     0.1628  48.590  < 2e-16 ***
## in_domain           -1.0258     0.1417  -7.242 3.05e-11 ***
## testMCMC             0.3236     0.1686   1.919   0.0570 .
## testUNIFORM_RANDOM  -0.3402     0.1829  -1.860   0.0651 .
## aug1                -2.0040     0.1382 -14.496  < 2e-16 ***
## ---
## Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
## Residual standard error: 0.8179 on 135 degrees of freedom
## Multiple R-squared:  0.6814,	Adjusted R-squared:  0.6719
## F-statistic: 72.17 on 4 and 135 DF,  p-value: < 2.2e-16
