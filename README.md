# Wobble

Learn machine learning by breaking it. Interactive lessons from linear regression to transformers: each one has a model you can drag and a button that makes it fail.

Work in progress. Built in stages, each on its own branch:

1. `stage-1-scaffold`: site shell, design tokens, fonts, lesson content format
2. `stage-2-engine`: the TypeScript ML engine, checked against scikit-learn
3. `stage-3-foundations`: Part 1 lessons (fit and overfit)
4. `stage-4-trust`: Part 3 lessons (trust the score)
5. `stage-5-ship`: home, progress, About, accessibility, deploy

`design/` holds the design canvas files the site is built from (blush and matcha).

Code is released under the [MIT License](LICENSE).

## The ML engine

`lib/ml/` is a small machine learning library written in TypeScript so every widget computes in your browser: least squares by Householder QR, ridge, lasso, polynomial fits, gradient descent, scaling, train/test and k-fold splits, walk-forward windows, classification metrics and L2 logistic regression.

It is checked against scikit-learn. `scripts/make_fixtures.py` writes reference numbers and `npm test` compares the engine to them. Time-ordered splits are also tested so that training never includes the future.

Known behaviour that the lessons account for: a high-degree polynomial on a few points can overfit by a huge and unstable margin, so the overfitting lesson caps the degree and uses a log scale for the error.
