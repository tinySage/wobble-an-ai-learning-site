"""Writes tests/fixtures/sklearn.json. The TypeScript engine is checked against these numbers.
Run: python scripts/make_fixtures.py   (needs numpy and scikit-learn)"""
import json, numpy as np, sklearn
from sklearn.linear_model import Ridge, LinearRegression, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

rng = np.random.default_rng(11)
out = {"sklearn": sklearn.__version__, "numpy": np.__version__}

x = np.sort(rng.uniform(0, 1, 20)); y = np.sin(2*np.pi*x)*0.9 + rng.normal(0, .2, 20)
grid = np.linspace(0, 1, 15)
def poly(v, d): return PolynomialFeatures(d, include_bias=False).fit_transform(v.reshape(-1, 1))
ols = LinearRegression().fit(poly(x, 3), y)
rdg = Ridge(alpha=0.1).fit(poly(x, 5), y)
out["poly"] = dict(x=x.tolist(), y=y.tolist(), grid=grid.tolist(),
  ols3=dict(coef=ols.coef_.tolist(), intercept=float(ols.intercept_), pred=ols.predict(poly(grid, 3)).tolist()),
  ridge5=dict(alpha=0.1, coef=rdg.coef_.tolist(), intercept=float(rdg.intercept_), pred=rdg.predict(poly(grid, 5)).tolist()))

X = rng.normal(size=(50, 6)); w = np.array([2.0, 0, -1.5, 0, 0, 0.8]); yl = X @ w + rng.normal(0, .3, 50)
las = Lasso(alpha=0.05, tol=1e-12, max_iter=200000).fit(X, yl)
out["lasso"] = dict(X=X.tolist(), y=yl.tolist(), alpha=0.05, coef=las.coef_.tolist(), intercept=float(las.intercept_))

gx = rng.uniform(-1, 1, 12); gy = 1.5*gx + 0.4 + rng.normal(0, .1, 12)
def gd(lr, steps):
    w_, b_ = 0.0, 0.0; tr = []
    for _ in range(steps):
        p = w_*gx + b_; loss = float(np.mean((p-gy)**2)); tr.append([w_, b_, loss])
        w_, b_ = w_ - lr*2*np.mean((p-gy)*gx), b_ - lr*2*np.mean(p-gy)
    return tr
out["gd"] = dict(x=gx.tolist(), y=gy.tolist(), ok=dict(lr=0.1, steps=25, traj=gd(0.1, 25)), diverge=dict(lr=1.6, steps=12, traj=gd(1.6, 12)))

S = rng.normal(5, 3, size=(30, 2)); sc = StandardScaler().fit(S)
out["scale"] = dict(X=S.tolist(), mean=sc.mean_.tolist(), std=sc.scale_.tolist(), z=sc.transform(S).tolist())
out["kfold"] = dict(n=23, k=5, tests=[t.tolist() for _, t in KFold(5).split(np.arange(23))])

yt = rng.integers(0, 2, 80); ys = np.round(np.clip(yt*0.25 + rng.normal(.4, .25, 80), 0, 1), 2)
yp = (ys >= .5).astype(int)
out["metrics"] = dict(y=yt.tolist(), score=ys.tolist(), auc=float(roc_auc_score(yt, ys)), precision=float(precision_score(yt, yp)),
  recall=float(recall_score(yt, yp)), f1=float(f1_score(yt, yp)), confusion=confusion_matrix(yt, yp).tolist())

LX = rng.normal(size=(60, 3)); ly = (LX @ np.array([1.2, -0.8, 0.0]) + rng.normal(0, .8, 60) > 0).astype(int)
out["logistic"] = dict(X=LX.tolist(), y=ly.tolist(), fits=[])
for C in (1.0, 0.1):
    m = LogisticRegression(C=C, tol=1e-12, max_iter=10000).fit(LX, ly)
    out["logistic"]["fits"].append(dict(C=C, coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]), proba=m.predict_proba(LX)[:, 1].tolist()))
json.dump(out, open("tests/fixtures/sklearn.json", "w"))
print("ok", out["sklearn"])
