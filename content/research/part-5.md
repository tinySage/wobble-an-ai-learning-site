# Part 5 research notes: neural networks

Sources compared per lesson: scikit-learn user guide, Goodfellow/Bengio/Courville *Deep Learning*, Stanford CS231n notes, the original papers (Rumelhart et al. 1986, Kingma and Ba 2015, Srivastava et al. 2014).

## Where sources disagree
- **Sigmoid in hidden layers.** CS231n says never use it there. Other texts keep it for probability outputs. Lesson says: avoid in hidden layers, fine at the output.
- **Adam vs SGD generalisation.** Papers disagree on whether SGD generalises better. Marked contested. The widget shows optimisation speed only.
- **Residual connections and sigmoid.** They only partly help. Reported as measured in the widget.
- **UMAP / global structure** (Part 4) is carried over as contested.

## Checked against reference code
Perceptron vs scikit-learn, backprop gradients vs a numpy MLP, convolution vs SciPy, gradient check catches deliberately planted bugs.
