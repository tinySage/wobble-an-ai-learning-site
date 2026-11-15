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
MX = np.column_stack([rng.uniform(0, 1, 25), rng.uniform(0, 300, 25)]); my = 2*MX[:, 0] + 0.01*MX[:, 1] + 1 + rng.normal(0, .1, 25)
Z = np.column_stack([MX, np.ones(25)]); H = 2/25 * Z.T @ Z
def gdm(lr, steps):
    wb = np.zeros(3); losses = []
    for _ in range(steps):
        r = Z @ wb - my; losses.append(float(np.mean(r**2))); wb = wb - lr * 2/25 * Z.T @ r
    return losses
lmax = float(np.linalg.eigvalsh(H)[-1])
out["multi"] = dict(X=MX.tolist(), y=my.tolist(), lmax=lmax, lr=0.5/lmax, losses=gdm(0.5/lmax, 40), div_lr=1.2*2/lmax, div_losses=gdm(1.2*2/lmax, 40), ok_lr=0.95*2/lmax, ok_losses=gdm(0.95*2/lmax, 40))
from sklearn.metrics import brier_score_loss, average_precision_score
from sklearn.calibration import calibration_curve
cy = (rng.uniform(size=300) < 0.2).astype(int); cs = np.clip(cy*0.3 + rng.normal(0.2, 0.2, 300), 0.001, 0.999)
fp_, mp_ = calibration_curve(cy, cs, n_bins=10, strategy="uniform")
out["calib"] = dict(y=cy.tolist(), score=cs.tolist(), brier=float(brier_score_loss(cy, cs)), ap=float(average_precision_score(cy, cs)), frac=fp_.tolist(), mean=mp_.tolist())
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
CX = np.vstack([rng.normal([.3,.35], .12, (40,2)), rng.normal([.65,.62], .12, (40,2))]); cy = np.array([0]*40 + [1]*40)
gx, gy = np.meshgrid(np.linspace(0,1,15), np.linspace(0,1,15)); G = np.column_stack([gx.ravel(), gy.ravel()])
out["cls"] = dict(X=CX.tolist(), y=cy.tolist(), G=G.tolist(),
  knn=KNeighborsClassifier(5).fit(CX, cy).predict_proba(G)[:,1].tolist(),
  nb=GaussianNB().fit(CX, cy).predict_proba(G)[:,1].tolist(),
  tree=DecisionTreeClassifier(max_depth=3, random_state=0).fit(CX, cy).predict_proba(G)[:,1].tolist(),
  gb=GradientBoostingClassifier(n_estimators=15, learning_rate=0.3, max_depth=2, random_state=0).fit(CX, cy).predict_proba(G)[:,1].tolist(),
  svm_rbf=SVC(C=1.0, gamma=3.0).fit(CX, cy).decision_function(G).tolist(), svm_n=int(SVC(C=1.0, gamma=3.0).fit(CX, cy).n_support_.sum()))

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.manifold._t_sne import _joint_probabilities
from scipy.spatial.distance import squareform, pdist
UX = np.vstack([rng.normal([.2,.3], .06, (25,2)), rng.normal([.7,.3], .06, (25,2)), rng.normal([.5,.8], .06, (25,2))])
init = UX[[0, 30, 55]]; km = KMeans(3, init=init, n_init=1, algorithm="lloyd").fit(UX)
th = np.linspace(0, np.pi, 40); MN = np.vstack([np.column_stack([.3+.3*np.cos(th), .42+.22*np.sin(th)]), np.column_stack([.7-.3*np.cos(th), .58-.22*np.sin(th)])]) + rng.normal(0, .02, (80,2))
Z5 = rng.normal(size=(100,5)) @ rng.normal(size=(5,5)); pc = PCA().fit(Z5)
def gm(cov):
    base = GaussianMixture(3, covariance_type=cov, random_state=0).fit(UX)
    return base
gms = {}
for cv in ["full", "diag", "spherical"]:
    mu0 = init.copy(); w0 = np.ones(3)/3
    c0 = np.array([np.cov(UX.T) for _ in range(3)])
    if cv == "full": p0 = np.array([np.linalg.inv(c) for c in c0])
    elif cv == "diag": p0 = np.array([1/np.diag(c) for c in c0])
    else: p0 = np.array([2/np.trace(c) for c in c0])
    g = GaussianMixture(3, covariance_type=cv, tol=1e-9, max_iter=500, weights_init=w0, means_init=mu0, precisions_init=p0).fit(UX)
    gms[cv] = dict(loglik=float(g.score(UX)*len(UX)), bic=float(g.bic(UX)), aic=float(g.aic(UX)), means=g.means_.tolist(), weights=g.weights_.tolist(), cov0=c0.tolist())
HX = rng.normal(size=(40,6)); D2 = (squareform(pdist(HX, "sqeuclidean"))).astype(np.float32)
Pj = squareform(_joint_probabilities(D2, 12.0, 0))
out["unsup"] = dict(X=UX.tolist(), init=init.tolist(), km_inertia=float(km.inertia_), km_labels=km.labels_.tolist(), km_centers=km.cluster_centers_.tolist(), sil=float(silhouette_score(UX, km.labels_)),
  moons=MN.tolist(), db=DBSCAN(eps=0.07, min_samples=4).fit(MN).labels_.tolist(),
  agg={l: AgglomerativeClustering(3, linkage=l).fit(UX).labels_.tolist() for l in ["single","complete","average","ward"]},
  pca_X=Z5.tolist(), pca_values=pc.explained_variance_.tolist(), pca_ratio=pc.explained_variance_ratio_.tolist(), pca_abs=np.abs(pc.components_).tolist(),
  gmm=gms, tsne_X=HX.tolist(), tsne_P=Pj.tolist())
json.dump(out, open("tests/fixtures/sklearn.json", "w"))
print("ok", out["sklearn"])
