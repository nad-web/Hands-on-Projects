import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import kendalltau

# ─── 1. CAV Direction ────────────────────────────────────────────────────────

def compute_cav(acts_pos, acts_neg):
    """Compute Concept Activation Vector from positive/negative activations.
    
    CAV direction: v_hat = w / ||w||  where w is the LinearSVC weight vector.
    """
    X = np.vstack([acts_pos, acts_neg])
    y = np.concatenate([np.ones(len(acts_pos)), np.zeros(len(acts_neg))])
    svm = LinearSVC(C=1.0, max_iter=10000, dual='auto')
    svm.fit(X, y)
    w = svm.coef_[0]
    return w / np.linalg.norm(w)

# ─── 2. Linear Probing ───────────────────────────────────────────────────────

def linear_probe(acts, labels, n_splits=5):
    """Train linear probe with stratified k-fold cross-validation.
    
    Equation: f_probe^(l): h^(l)(x) -> y_hat_c ∈ {0,1}
    """
    clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='lbfgs')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, acts, labels, cv=skf, scoring='accuracy', n_jobs=-1)
    return scores.mean(), scores.std()

# ─── 3. RSA: RDM Computation ─────────────────────────────────────────────────

def compute_rdm(acts):
    """Compute Representational Dissimilarity Matrix.
    
    Equation: RDM_ij = 1 - corr(h(x_i), h(x_j))
    """
    rdm = 1 - np.corrcoef(acts)
    np.fill_diagonal(rdm, 0)
    return rdm

def make_concept_rdm(labels):
    """Construct concept-model RDM: 0 for same-concept pairs, 1 for different."""
    n = len(labels)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] != labels[j]:
                rdm[i, j] = 1.0
    return rdm

# ─── 4. RSA: Kendall's tau ───────────────────────────────────────────────────

def rsa_correlation(rdm_net, rdm_concept):
    """Compute Kendall's tau-b between lower triangles of two RDMs."""
    idx = np.tril_indices_from(rdm_net, k=-1)
    tau, pvalue = kendalltau(rdm_net[idx], rdm_concept[idx])
    return tau, pvalue

def rsa_bootstrap_ci(rdm_net, rdm_concept, n_bootstrap=1000, confidence=0.95):
    """Bootstrap confidence interval for RSA correlation."""
    n = rdm_net.shape[0]
    taus = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        boot_net = rdm_net[np.ix_(indices, indices)]
        boot_concept = rdm_concept[np.ix_(indices, indices)]
        tau, _ = rsa_correlation(boot_net, boot_concept)
        taus.append(tau)
    alpha = 1 - confidence
    tau_obs, _ = rsa_correlation(rdm_net, rdm_concept)
    return (tau_obs,
            np.percentile(taus, 100 * alpha / 2),
            np.percentile(taus, 100 * (1 - alpha / 2)))

# ─── 5. Targeted Ablation ────────────────────────────────────────────────────

def targeted_ablation(model, images, labels, concept_mask, layer_idx,
                      top_pct=0.10, device='cpu'):
    """Ablate top concept-aligned units and measure accuracy change.
    
    Returns: (acc_before, acc_after, delta_acc)
    """
    model.eval()
    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1)
        acc_before = (preds.cpu() == labels).float().mean().item()

    acts = []
    def hook_fn(module, input, output): acts.append(output)
    handle = model.conv_layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad(): model(images.to(device))
    handle.remove()

    activation = acts[0].mean(dim=(2, 3))
    concept_labels_t = concept_mask.float().to(device)
    correlations = [
        torch.corrcoef(torch.stack([activation[:, c], concept_labels_t]))[0, 1].abs().item()
        for c in range(activation.shape[1])
    ]
    top_indices = np.argsort(correlations)[-max(1, int(top_pct * len(correlations))):]

    def ablation_hook(module, input, output):
        output[:, top_indices, :, :] = 0
        return output

    handle = model.conv_layers[layer_idx].register_forward_hook(ablation_hook)
    with torch.no_grad():
        preds_abl = model(images.to(device)).argmax(dim=1)
        acc_after = (preds_abl.cpu() == labels).float().mean().item()
    handle.remove()
    return acc_before, acc_after, acc_after - acc_before

# ─── 6. Counterfactual Injection ─────────────────────────────────────────────

def counterfactual_injection(model, images, labels, cav, layer_idx,
                             alpha=1.0, device='cpu'):
    """Inject concept direction into activations and measure logit change.
    
    Equation: h_mod = h + alpha * v_hat
    """
    model.eval()
    cav_t = torch.tensor(cav, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits_orig = model(images.to(device))

    def injection_hook(module, input, output):
        b, c, h, w = output.shape
        gap = output.mean(dim=(2, 3))
        gap_modified = gap + alpha * cav_t.unsqueeze(0)
        diff = (gap_modified - gap).view(b, c, 1, 1)
        return output + diff.expand(b, c, h, w)

    handle = model.conv_layers[layer_idx].register_forward_hook(injection_hook)
    with torch.no_grad():
        logits_mod = model(images.to(device))
    handle.remove()

    target_orig = logits_orig.gather(1, labels.view(-1, 1).to(device))
    target_mod  = logits_mod.gather(1, labels.view(-1, 1).to(device))
    logit_change = (target_mod - target_orig).squeeze()
    return logit_change.mean().item(), logit_change.std().item()
