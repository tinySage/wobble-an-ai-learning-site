# Part 3 source check

| Topic | Sources compared | Difference found | What the lesson does |
|---|---|---|---|
| Data leakage | scikit-learn common pitfalls, UBC CPSC 330 | Agree on definition and on pipelines as the fix | Demonstrates leaky feature selection on random labels |
| Metrics | Google MLCC, scikit-learn | Agree on definitions | Shows both ROC and PR views |
| Imbalance | Google MLCC, Machine Learning Mastery | MLCC says ROC/AUC suit roughly balanced data; MLM says ROC AUC can flatter imbalanced data but one of its own three examples had PR AUC about equal to ROC AUC | Shows both numbers and says it depends on the dataset |
| Calibration | scikit-learn examples (older vs newer) | v1.1 text says calibration does not significantly alter precision, recall, F1; dev text says it can significantly alter discrete-prediction metrics. Both agree ROC AUC is unchanged by a strictly monotonic transform; isotonic can nudge it | Says AUC is unchanged under a stretch, accuracy at 0.5 may shift after recalibration |
| Hyperparameter search | Bergstra and Bengio, scikit-learn | Paper: random beats grid in high dimensions, grid is reliable in 1-2 dimensions; random matched or beat grid on 5 of 7 datasets, not all | Toggle lets you see grid hold up when both settings matter |

URLs were found by search and read as excerpts, not in full.
