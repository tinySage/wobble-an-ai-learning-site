# Part 4 research: find structure

| Topic | Sources compared | Where they differ | What we did |
|---|---|---|---|
| k-means | scikit-learn, DataCamp | DataCamp treats silhouette as a guide, sklearn warns about local minima | Show inertia and silhouette side by side, offer random vs k-means++ starts |
| Clustering methods | sklearn comparison table, DataCamp, The Neural Base | One source says DBSCAN needs no assumptions about the data, but it still needs an eps that suits the density | "Uneven" dataset shows one eps cannot fit tight and loose groups at once |
| PCA scaling | scikit-learn, EPFL, a mailing-list thread, an astronomy paper | sklearn does not scale; EPFL says standardise unless units agree; the thread notes spread can carry meaning | Standardise toggle plus the units break-it |
| t-SNE and UMAP | Distill, Kobak and Linderman, MetricGate, TheoremPath, Datanovia | Some guides say UMAP keeps more global structure, Kobak and Linderman show the gap closes with the same initialisation | Lesson calls the claim contested. The widget implements t-SNE only |
| Gaussian mixtures | scikit-learn | Likelihood always improves with more components | Show BIC for three covariance shapes |
| Anomaly detection | scikit-learn, Liu et al. | "auto" threshold vs a numeric contamination | Slider is a flag rate, and the break-it shows false alarms |

Engine checks against scikit-learn 1.9.1: k-means (inertia, labels), silhouette, DBSCAN labels, four linkages, PCA variances and axes, Gaussian mixture log-likelihood, BIC and means for all three covariance shapes, and t-SNE affinities.
