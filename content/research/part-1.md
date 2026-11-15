# Part 1 source check

Each lesson was written after comparing at least two sources. Where they differ, the lesson says so.

| Topic | Sources compared | Difference found | What the lesson does |
|---|---|---|---|
| Linear regression | CS229 notes, scikit-learn, Google MLCC | CS229 writes loss with a ½ factor and theta; MLCC says weight and bias; sklearn says coef_ and intercept_ | Uses w and b, mean squared error |
| Solving it | CS229 normal equations vs scikit-learn | CS229 shows (XᵀX)⁻¹Xᵀy; sklearn solves with a least-squares routine | Notes inversion is avoided in practice; engine uses QR |
| Learning rate | Google MLCC, CS229 | MLCC's 0.03 and 1 are for its dataset only | Shows the stability limit 2/λmax computed from the actual data |
| Feature scaling | Let's Data Science, howaiworks, Kanaries | All agree descent and penalties need it, OLS does not; StandardScaler uses population std | Notes both; says fit scaler on training data only |
| Ridge strength | sklearn vs mean-based textbook forms | sklearn Ridge uses summed error, so α is not comparable | Warns in The maths tab |
| Overfitting | MLU-Explain, Kanaries, interactive explorers | Explorers allow degree up to 20; small samples are very unstable | Caps degree at 9, log-scale error, says results vary by dataset |
| Bias and variance | MLU-Explain, CS229, UW CSE 163 | Texts agree on the decomposition; U-shape is not universal (double descent) | States decomposition, notes real data lacks known truth |

URLs were found by search; page contents were read as search excerpts, not in full. KaTeX rendering of maths is planned for the polish stage.
