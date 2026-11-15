/** The whole planned course. A lesson is "live" when content/lessons/<slug>.json exists. */
export interface Planned { slug: string; title: string; verb: string }
export interface Part { id: number; slug: string; title: string; blurb: string; lessons: Planned[] }

const L = (slug: string, title: string, verb: string): Planned => ({ slug, title, verb });

export const PARTS: Part[] = [
  { id: 1, slug: "fit-and-overfit", title: "Fit and overfit", blurb: "Draw a line through data, then see how it goes wrong.", lessons: [
    L("linear-regression", "Linear regression", "make it"), L("gradient-descent", "Gradient descent", "make it"),
    L("learning-rate", "Learning rate", "break it"), L("feature-scaling", "Feature scaling", "fix it"),
    L("polynomial-features", "Polynomial features", "make it"), L("overfitting", "Overfitting", "break it"),
    L("regularisation", "Regularisation", "fix it"), L("bias-and-variance", "Bias and variance", "read it") ] },
  { id: 2, slug: "classify", title: "Classify", blurb: "Draw boundaries between groups.", lessons: [
    L("logistic-regression", "Logistic regression", "make it"), L("decision-boundaries", "Decision boundaries", "read it"),
    L("k-nearest-neighbours", "k-nearest neighbours", "break it"), L("naive-bayes", "Naive Bayes", "make it"),
    L("decision-trees", "Decision trees", "break it"), L("random-forests", "Random forests", "fix it"),
    L("gradient-boosting", "Gradient boosting", "make it"), L("support-vector-machines", "Support vector machines", "break it") ] },
  { id: 3, slug: "trust-the-score", title: "Trust the score", blurb: "Learn when a number can and cannot be believed.", lessons: [
    L("train-and-test", "Train and test", "trust it"), L("cross-validation", "Cross-validation", "fix it"),
    L("data-leakage", "Data leakage", "break it"), L("metrics", "Precision, recall and ROC", "read it"),
    L("imbalanced-classes", "Imbalanced classes", "break it"), L("calibration", "Calibration", "read it"),
    L("hyperparameter-search", "Hyperparameter search", "make it") ] },
  { id: 4, slug: "find-structure", title: "Find structure", blurb: "Group and compress data with no labels.", lessons: [
    L("k-means", "k-means", "break it"), L("clustering-methods", "Hierarchical and DBSCAN", "make it"),
    L("pca", "PCA", "make it"), L("t-sne-and-umap", "t-SNE and UMAP", "break it"),
    L("gaussian-mixtures", "Gaussian mixtures", "make it"), L("anomaly-detection", "Anomaly detection", "make it") ] },
  { id: 5, slug: "neural-networks", title: "Neural networks", blurb: "Build the idea up from one neuron.", lessons: [
    L("perceptron", "The perceptron", "make it"), L("multilayer-networks", "Multilayer networks", "make it"),
    L("backpropagation", "Backpropagation", "read it"), L("activations", "Activation functions", "break it"),
    L("optimisers", "SGD and Adam", "make it"), L("vanishing-gradients", "Vanishing gradients", "break it"),
    L("convolutions", "Convolutions", "make it"), L("dropout", "Dropout", "fix it") ] },
  { id: 6, slug: "advanced", title: "Advanced", blurb: "Sequences, attention, agents and fairness.", lessons: [
    L("time-series", "Time series and walk-forward testing", "break it"), L("hidden-markov-models", "Hidden Markov models", "make it"),
    L("embeddings", "Embeddings", "read it"), L("attention", "Attention and transformers", "read it"),
    L("bandits-and-q-learning", "Bandits and Q-learning", "make it"), L("autoencoders", "Autoencoders", "make it"),
    L("diffusion", "Diffusion, the intuition", "read it"), L("interpretability", "Interpretability and SHAP", "read it"),
    L("fairness", "Fairness", "break it") ] },
];

export const allPlanned = () => PARTS.flatMap((p) => p.lessons.map((l) => ({ ...l, part: p })));
