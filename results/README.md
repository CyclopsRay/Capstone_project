# results

Made by `pixi run reproduce`, not committed:

- `scores.csv`: PCC_DEG by fold, split (train, valid, test) and stage of the model; stage `ceiling` is the simulated
  response without its noise
- `view_weights.csv`: the weight of each compound view, learned and simulated, by fold
- `summary.json`: means and standard deviations over folds, data sizes and settings
- `folds/fold_<f>/`: per fold, the score of every sample, the settings chosen on the validation compounds and the
  test predictions
- `logs/`: one log per step
