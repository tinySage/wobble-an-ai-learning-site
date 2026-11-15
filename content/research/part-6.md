# Part 6 research notes: sequences, language, agents, generation, explanation, fairness

Sources were searched and compared per lesson. Links were opened or appeared in search results. The Bergmeir and Benitez paper was dropped from the time-series sources because its DOI could not be opened from here.

## Where sources disagree
- **Time series.** Everyone agrees on walk-forward testing. They differ on window shape (expanding for stable processes, sliding for drift) and on gaps. One source warns a gap cannot repair a feature that already looks ahead; the lesson says so. Shuffled folds flatter even honest features (0.205 against 0.307 here), and a peeking feature fools walk-forward too (0.133).
- **HMMs.** Rabiner and the course notes agree on the three problems. Baum-Welch is only a local search; the widget shows a start with identical states never separating.
- **Embeddings.** Levy and Goldberg (2014) find SVD matches skip-gram on similarity but trails it on analogies. Arora et al. give a theory for why analogies can work. The widget reports analogies as measured: the royal one lands in the top three, the capital one sometimes misses.
- **Attention scaling.** Vaswani et al. say they suspect large dot products push softmax into flat regions. A later paper notes dimension is not the only cause of large scores. The lesson states the variance argument for independent unit-variance entries and nothing stronger.
- **Bandits.** Textbooks favour UCB and Thompson sampling by regret bounds. In these runs plain epsilon-greedy (0.1) beat UCB (c = 2) at 600 pulls, so the lesson says it depends on the constant. Q-learning versus SARSA on the cliff: both are right, for different goals.
- **Autoencoders.** Linear autoencoder equals PCA in subspace (Baldi and Hornik 1989), not in basis. Sources agree.
- **Diffusion.** The exact score stands in for the network, so the lesson makes no claim about learned networks. Fewer sampling steps are cited from Nichol and Dhariwal.
- **Shapley values.** Real disagreement: interventional (independent background) versus conditional value functions. Kumar et al. say both have serious problems; Janzing et al. argue for unconditional; Aas-style conditional is argued for by others. The lesson presents both.
- **Fairness.** Kleinberg et al. and Chouldechova agree on the trade-off. Bell et al. argue relaxed versions are achievable in practice. Pleiss et al. show calibration fits with at most one error-rate constraint.

## Checked against reference code or maths
Time-series splits match scikit-learn's TimeSeriesSplit; HMM forward equals listing all paths and Viterbi equals brute-force search; PPMI rank-k error equals the dropped eigenvalues; attention matches a hand calculation; Q-learning's best return equals value iteration (-13); a linear autoencoder reaches the PCA error; noising in one step matches the iterated chain; Shapley credits add up and match the linear formula; the PPV identity matches measured precision.
