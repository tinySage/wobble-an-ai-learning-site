# Part 2 research: classify things

| Topic | Sources compared | Where they differ | What we did |
|---|---|---|---|
| Logistic regression | Google MLCC, scikit-learn | MLCC frames a penalty and early stopping; sklearn uses C as inverse strength | Show C directly, explain the inverse |
| kNN | sklearn, howaiworks, Let's Data Science | sqrt(n) rule is a start, not a law | Slider for k, no rule imposed |
| Naive Bayes | sklearn (two pages) | Good classifier, poor probabilities | Widget shows confidence next to accuracy |
| Trees | sklearn, MachineLearningMastery, MLU-Explain | Control by depth versus leaf size | Depth slider only, noted in lesson |
| Forests | sklearn, Breiman | sklearn averages probabilities; Breiman votes | Averaged, flagged in maths tab |
| Boosting | sklearn | Learning rate trades against rounds | Both sliders exposed |
| SVM | sklearn RBF example | Too large a gamma cannot be rescued by C | Both sliders, break-it uses gamma |
