# results

Outputs of the demo run (`pixi run reproduce`), regenerated on every run and not
committed:

- `demo/scores.csv`: PCC_DEG and PCC_plate of each method on each test fold
- `demo/weights.csv`: the view weights UnDetect learned, next to the simulated ones
- `demo/summary.json`: five-fold means, data sizes and settings, read by the report
- `demo/logs/`: one log file per workflow step

The full paper run writes its results under `$UNDETECT_DATA/results` instead
(README.md).
