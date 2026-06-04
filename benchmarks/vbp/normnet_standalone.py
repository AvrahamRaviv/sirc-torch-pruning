#!/usr/bin/env python
"""
Normalized-Net pruning — SELF-CONTAINED demo (single GPU).

One file, end to end:
    train  ->  normalize/reparam (calibrate sigma, fold)  ->  score (NCI / rel / nonrel)
           ->  prune (torch_pruning)  ->  fine-tune  ->  report.

Imports ONLY: torch, torchvision, torch_pruning (the published package) + stdlib.
No project-internal imports — give this to anyone, they can run + debug it alone.

------------------------------------------------------------------------------
THE THREE SCORES (per INPUT channel of each prunable layer l)
------------------------------------------------------------------------------
Normalized net: before each target layer, input is normalized to zero-mean/unit-var
by its measured (mu, sigma); the weight is re-expressed as  v~ = sigma * W  (fold).
The fold is function-preserving (asserted below). On the normalized net the weight
magnitude IS the contribution score.

  M[i,j] = sigma_in[i] * || reduce_kernel(W[j,i,:]) ||           (in x out)

  NCI    : one hop, no propagation.    NCI[i] = sum_j M[i,j]^2   ( = sigma_i^2 ||W_i||^2 )
  rel    : I^l = Wbar @ I^{l+1},  Wbar = M^p / colsum(M^p)       (column-stochastic)
           -> mass-preserving, LOCAL (compare channels within a layer only)
  nonrel : same Wbar, but first multiply downstream importance by the per-hop
           transfer  sigma_post^p  (the measured activation gain Sigma^{l+1}).
           -> tries to be GLOBAL (cross-layer); see the spectral caveat in the docstring
              of compute_propagation() below.

------------------------------------------------------------------------------
CHANNEL DIRECTION (the subtle part)
------------------------------------------------------------------------------
torch_pruning prunes a COUPLED group: a producer's OUTPUT channel k and every
consumer's INPUT channel k are the SAME physical channel. Our score is INPUT-side
(per consumer-input channel). NormScoreImportance therefore reads the score off the
group's IN-side member (prune_*_in_channels) -> tp maps it to the right physical
channels via root_idxs. Attaching an input-score to an output dim would prune the
WRONG layer (off by one). assert_alignment() checks we got it right.
"""
import argparse, copy, time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch_pruning as tp
from torch_pruning.pruner import function as tp_fn

IN_FNS  = (tp_fn.prune_conv_in_channels,  tp_fn.prune_linear_in_channels)
OUT_FNS = (tp_fn.prune_conv_out_channels, tp_fn.prune_linear_out_channels)


# ----------------------------------------------------------------------------- device / data / model
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_data(args):
    """CIFAR-10 (downloaded, resized) or FakeData fallback (always runs offline)."""
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_set = val_set = None
    if args.dataset == "cifar10":
        try:
            train_set = torchvision.datasets.CIFAR10(args.data_path, train=True,  download=True, transform=tf)
            val_set   = torchvision.datasets.CIFAR10(args.data_path, train=False, download=True, transform=tf)
            n_classes = 10
        except Exception as e:                              # offline cluster -> fake
            print(f"[data] CIFAR-10 unavailable ({e}); using FakeData.")
            args.dataset = "fake"
    if args.dataset == "fake":
        n_classes = args.num_classes
        train_set = torchvision.datasets.FakeData(2000, (3, args.img_size, args.img_size), n_classes, tf)
        val_set   = torchvision.datasets.FakeData(500,  (3, args.img_size, args.img_size), n_classes, tf)
    pin = args.device.type == "cuda"
    nw  = args.workers if args.device.type == "cuda" else 0   # MPS/CPU: keep 0
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                                               num_workers=nw, pin_memory=pin)
    return train_loader, val_loader, n_classes


def build_model(args, n_classes):
    name = args.model
    weights = "DEFAULT" if args.pretrained else None
    try:
        model = getattr(torchvision.models, name)(weights=weights)
    except Exception as e:
        print(f"[model] pretrained weights unavailable ({e}); random init.")
        model = getattr(torchvision.models, name)(weights=None)
    # swap classifier head to n_classes
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        first_conv, classifier = model.conv1, model.fc
    elif name.startswith("convnext"):
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, n_classes)
        first_conv, classifier = model.features[0][0], model.classifier[2]
    else:
        raise ValueError(f"unsupported model {name}")
    return model, first_conv, classifier


# ----------------------------------------------------------------------------- train / eval
def run_epoch(model, loader, device, opt=None, limit=0):
    train = opt is not None
    model.train(train)
    tot = correct = 0
    loss_sum = 0.0
    for bi, (x, y) in enumerate(loader):
        if limit and bi >= limit:
            break
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = F.cross_entropy(out, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * y.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        tot      += y.size(0)
    return loss_sum / max(tot, 1), 100.0 * correct / max(tot, 1)


def train_model(model, loader, device, epochs, lr, limit=0, tag="train"):
    if epochs <= 0:
        return
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        t = time.time()
        loss, acc = run_epoch(model, loader, device, opt, limit)
        sched.step()
        print(f"[{tag}] epoch {ep+1}/{epochs}  loss {loss:.3f}  acc {acc:5.2f}  "
              f"lr {sched.get_last_lr()[0]:.4f}  ({time.time()-t:.1f}s)")


# ----------------------------------------------------------------------------- normalize / calibrate
def select_prunable(model, first_conv, classifier):
    """Plain Conv2d (groups==1, in>1) + Linear, excluding stem conv and classifier.
    Depthwise convs are intentionally skipped (groups!=1) -> not scored, not pruned."""
    P = OrderedDict()
    for name, m in model.named_modules():
        if m is first_conv or m is classifier:
            continue
        if isinstance(m, nn.Conv2d) and m.groups == 1 and m.in_channels > 1:
            P[name] = m
        elif isinstance(m, nn.Linear):
            P[name] = m
    return P


@torch.no_grad()
def calibrate(model, prunable, loader, device, n_batches, save_input_for):
    """One pass: per prunable layer capture input mean/std (mu_in, sigma_in) per input
    channel, and forward FIRING ORDER (a valid topo sort of the prunable DAG).
    `save_input_for` modules also stash one input batch (for the fold sanity check)."""
    stats = {m: {"sum": None, "sqsum": None, "n": 0} for m in prunable.values()}
    fire_order = []
    seen = set()
    saved_inputs = {}

    def pre_hook(mod, inp):
        x = inp[0].detach()
        if mod not in seen:
            fire_order.append(mod); seen.add(mod)
        # channel axis depends on LAYER TYPE, not tensor rank: Conv2d is NCHW (dim 1);
        # Linear is channels-LAST (dim -1) — and in ConvNeXt its input is 4-D [N,H,W,C].
        if isinstance(mod, nn.Conv2d):
            s  = x.sum(dim=(0, 2, 3)); sq = (x * x).sum(dim=(0, 2, 3)); cnt = x.numel() // x.size(1)
        else:                   # Linear: [..., C]
            xf = x.reshape(-1, x.size(-1)); s = xf.sum(0); sq = (xf * xf).sum(0); cnt = xf.size(0)
        st = stats[mod]
        st["sum"]   = s  if st["sum"]   is None else st["sum"]   + s
        st["sqsum"] = sq if st["sqsum"] is None else st["sqsum"] + sq
        st["n"]    += cnt
        if mod in save_input_for and mod not in saved_inputs:
            saved_inputs[mod] = x[:4].clone()       # tiny slice for the assert

    handles = [m.register_forward_pre_hook(pre_hook) for m in prunable.values()]
    model.eval()
    bi = 0
    for x, _ in loader:
        model(x.to(device))
        bi += 1
        if bi >= n_batches:
            break
    for h in handles:
        h.remove()

    mu, sigma = {}, {}
    for m, st in stats.items():
        mean = st["sum"] / st["n"]
        var  = st["sqsum"] / st["n"] - mean * mean
        mu[m]    = mean
        sigma[m] = var.clamp_min(1e-8).sqrt()
    return mu, sigma, fire_order, saved_inputs


def assert_fold_preserving(layer, mu, sigma, x):
    """v~ = sigma*W, m = b + W@mu  reproduces  W x + b  on normalized input. Function-preserving."""
    W = layer.weight.data
    b = layer.bias.data if layer.bias is not None else torch.zeros(W.size(0), device=W.device)
    if isinstance(layer, nn.Linear):
        y0 = F.linear(x, W, b)
        xn = (x - mu) / sigma
        v  = W * sigma[None, :]
        m  = b + W @ mu
        y1 = F.linear(xn, v, m)
    else:  # Conv2d
        y0 = F.conv2d(x, W, b, layer.stride, layer.padding, layer.dilation, layer.groups)
        xn = (x - mu[None, :, None, None]) / sigma[None, :, None, None]
        v  = W * sigma[None, :, None, None]
        m  = b + W.sum(dim=(2, 3)) @ mu
        y1 = F.conv2d(xn, v, m, layer.stride, layer.padding, layer.dilation, layer.groups)
        # fold is exact on the INTERIOR; zero-padding borders see 0 not mu, so crop them
        ph, pw = (layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding))
        if ph or pw:
            y0 = y0[:, :, ph:y0.size(2) - ph if ph else None, pw:y0.size(3) - pw if pw else None]
            y1 = y1[:, :, ph:y1.size(2) - ph if ph else None, pw:y1.size(3) - pw if pw else None]
    return (y0 - y1).abs().max().item()


# ----------------------------------------------------------------------------- scores
@torch.no_grad()
def layer_M(layer, sigma_in):
    """M[in,out] = sigma_in[i] * ||reduce_kernel(W[out,in,:])||  (frobenius over kernel)."""
    W = layer.weight.data
    if W.dim() == 4:
        w_red = W.flatten(2).norm(p=2, dim=2)      # [out,in]
    else:
        w_red = W.abs()                            # [out,in]
    M = w_red.t().abs() * sigma_in[:, None]        # [in,out]
    return M


def discover_edges(model, example_inputs, prunable):
    """For each prunable layer L, downstream prunable consumers (channel-aligned)
    via tp's dependency group on L's OUTPUT dim. (DG tracing needs autograd -> no no_grad.)"""
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    Pset = set(prunable.values())
    edges = {}
    for L in prunable.values():
        out_fn = tp_fn.prune_conv_out_channels if isinstance(L, nn.Conv2d) else tp_fn.prune_linear_out_channels
        n_out  = L.out_channels if isinstance(L, nn.Conv2d) else L.out_features
        try:
            grp = DG.get_pruning_group(L, out_fn, idxs=list(range(n_out)))
        except Exception:
            edges[L] = []; continue
        cons = []
        for dep, _idxs in grp:
            lay, fn = dep.layer, dep.pruning_fn
            if fn in IN_FNS and lay in Pset and lay is not L and lay not in cons:
                cons.append(lay)
        edges[L] = cons
    return edges


@torch.no_grad()
def compute_scores(prunable, sigma, mu, edges, fire_order, p=2):
    """Returns dict mode -> {layer: per-input-channel score}.

    nonrel spectral caveat: propagation is an inhomogeneous product of per-layer
    operators. rel's Wbar is column-stochastic (total mass conserved -> bounded, but
    cannot rank ACROSS layers -> local only). nonrel injects sigma_post^p to break that
    stochasticity for cross-layer scale, but the per-hop mass-gain = E_{propagated
    dist}[sigma_post^p] != 1 in general -> geometric drift over depth. So nonrel is
    cross-layer in intent; verify its dynamic range before trusting it globally
    (a within-layer ranker via 'mean' normalization is the safe use).
    """
    def n_out(L): return L.out_channels if isinstance(L, nn.Conv2d) else L.out_features

    M = {L: layer_M(L, sigma[L]) for L in prunable.values()}
    sigma_post = {}     # per-OUTPUT-channel post-act std = a consumer's input std (Sigma^{l+1})
    for L in prunable.values():
        cons = edges[L]
        s = None
        for C in cons:
            if sigma[C].numel() == n_out(L):
                s = sigma[C] if s is None else s + sigma[C]
        sigma_post[L] = (s / len(cons)) if (s is not None and cons) else torch.ones(n_out(L), device=M[L].device)

    nci = {L: (M[L].pow(2)).sum(dim=1) for L in prunable.values()}   # one hop

    def propagate(nonrel):
        I_in = {}
        for L in reversed(fire_order):
            no = n_out(L)
            cons = [C for C in edges[L] if C in I_in and I_in[C].numel() == no]
            if cons:
                I_next = torch.zeros(no, device=M[L].device)
                for C in cons:
                    I_next = I_next + I_in[C]
            else:                                   # terminal -> uniform seed
                I_next = torch.full((no,), 1.0 / no, device=M[L].device)
            if nonrel:
                I_next = I_next * sigma_post[L].pow(p)        # per-hop transfer
            Mp = M[L].pow(p)
            colsum = Mp.sum(dim=0).clamp_min(1e-8)
            Wbar = Mp / colsum[None, :]                       # column-stochastic
            I_in[L] = Wbar @ I_next                           # [in]
        return I_in

    return {"nci": nci, "rel": propagate(False), "nonrel": propagate(True)}


# ----------------------------------------------------------------------------- importance adapter
class NormScoreImportance(tp.importance.GroupMagnitudeImportance):
    """Feed our per-INPUT-channel scores to tp. Reads the IN-side group member so the
    score lands on the correct physical channels (see CHANNEL DIRECTION note up top)."""
    def __init__(self, in_scores, normalizer="mean"):
        super().__init__(p=2, group_reduction="mean", normalizer=normalizer)
        self.in_scores = in_scores      # {module: 1-D tensor over its input channels}

    @torch.no_grad()
    def __call__(self, group):
        gi, gx = [], []
        for i, (dep, idxs) in enumerate(group):
            lay, fn = dep.layer, dep.pruning_fn
            if fn in IN_FNS and lay in self.in_scores:
                s = self.in_scores[lay].detach().float().cpu()
                gi.append(s[list(idxs)])
                gx.append(group[i].root_idxs)
        if not gi:
            return None                 # unscored group (e.g. depthwise) -> pruner skips
        return self._normalize(self._reduce(gi, gx), self.normalizer)


def assert_alignment(model, example_inputs, prunable, scores):
    """Sanity: lowest-score input channel of a layer is the one tp would drop first.
    Confirms input-score -> correct physical channel (no off-by-one-layer).
    (DG tracing needs autograd -> no no_grad here.)"""
    L = next(iter(prunable.values()))
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    s = scores[L]
    n_in = L.in_channels if isinstance(L, nn.Conv2d) else L.in_features
    in_fn = tp_fn.prune_conv_in_channels if isinstance(L, nn.Conv2d) else tp_fn.prune_linear_in_channels
    grp = DG.get_pruning_group(L, in_fn, idxs=list(range(n_in)))
    imp = NormScoreImportance({L: s}, normalizer=None)(grp)   # raw, preserves order
    # imp (per group channel) must match our input-side score -> same argmin = same channel
    return imp is not None and imp.numel() == n_in and int(imp.argmin()) == int(s.argmin())


# ----------------------------------------------------------------------------- prune one config
def prune_and_eval(base_model, example_inputs, prunable_names, first_conv_name, classifier_name,
                   mode_scores, args, train_loader, val_loader, device):
    model = copy.deepcopy(base_model).to(device)
    name2mod = dict(model.named_modules())
    ignored = [name2mod[first_conv_name], name2mod[classifier_name]]
    if args.mode == "magnitude":
        # stock tp baseline: plain group weight-magnitude (output-side, no normalization).
        # Magnitude is meaningful cross-layer -> GLOBAL ranking (the standard, fair baseline;
        # local fixed-ratio magnitude is a strawman).
        imp = tp.importance.MagnitudeImportance(p=2)
        global_pruning = True
    else:
        in_scores = {name2mod[n]: mode_scores[n] for n in prunable_names if n in mode_scores}
        # rel is LOCAL -> per-layer ratio; nci/nonrel may be global
        global_pruning = args.global_prune and args.mode != "rel"
        imp = NormScoreImportance(in_scores, normalizer="mean")

    macs0, params0 = tp.utils.count_ops_and_params(model, example_inputs)
    pruner = tp.pruner.MetaPruner(model, example_inputs, importance=imp,
                                  global_pruning=global_pruning,
                                  pruning_ratio=args.pruning_ratio,
                                  ignored_layers=ignored)
    pruner.step()
    macs1, params1 = tp.utils.count_ops_and_params(model, example_inputs)

    _, acc_pre = run_epoch(model, val_loader, device, limit=args.limit_batches)
    train_model(model, train_loader, device, args.epochs_ft, args.lr_ft, args.limit_batches, tag=f"ft-{args.mode}")
    _, acc_ft = run_epoch(model, val_loader, device, limit=args.limit_batches)
    return dict(mode=args.mode, params=params1 / 1e6, params_drop=100 * (1 - params1 / params0),
                macs=macs1 / 1e6, macs_drop=100 * (1 - macs1 / macs0),
                acc_pre_ft=acc_pre, acc_ft=acc_ft)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="resnet50", choices=["resnet18", "resnet50", "convnext_tiny"])
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "fake"])
    ap.add_argument("--data_path", default="./data")
    ap.add_argument("--num_classes", type=int, default=10)        # used by fake
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--epochs", type=int, default=2, help="base train epochs (0 = skip)")
    ap.add_argument("--epochs_ft", type=int, default=2, help="post-prune fine-tune epochs")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr_ft", type=float, default=0.005)
    ap.add_argument("--pruning_ratio", type=float, default=0.5)
    ap.add_argument("--global_prune", action="store_true", default=False)
    ap.add_argument("--calib_batches", type=int, default=10)
    ap.add_argument("--p", type=int, default=2, choices=[1, 2])
    ap.add_argument("--modes", default="magnitude,nci,rel,nonrel")
    ap.add_argument("--limit_batches", type=int, default=0, help="cap batches/epoch (smoke test)")
    args = ap.parse_args()

    args.device = device = pick_device()
    print(f"== device {device} | model {args.model} | dataset {args.dataset} | img {args.img_size} ==")

    train_loader, val_loader, n_classes = build_data(args)
    args.num_classes = n_classes
    model, first_conv, classifier = build_model(args, n_classes)
    model = model.to(device)
    example_inputs = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    # 1) train base
    train_model(model, train_loader, device, args.epochs, args.lr, args.limit_batches, tag="base")
    _, acc0 = run_epoch(model, val_loader, device, limit=args.limit_batches)
    macs0, params0 = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"[base] acc {acc0:.2f} | {params0/1e6:.2f}M params | {macs0/1e6:.1f}M MACs")

    # 2) normalize / calibrate
    prunable = select_prunable(model, first_conv, classifier)
    name2mod = dict(model.named_modules())
    mod2name = {m: n for n, m in name2mod.items()}
    fold_check = list(prunable.values())[:2]
    mu, sigma, fire_order, saved = calibrate(model, prunable, train_loader, device,
                                             args.calib_batches, save_input_for=fold_check)
    print(f"[calib] {len(prunable)} prunable layers | fire-order len {len(fire_order)}")

    # 3) fold function-preservation check
    for L in fold_check:
        if L in saved:
            d = assert_fold_preserving(L, mu[L], sigma[L], saved[L])
            print(f"[fold] {mod2name[L]:<40s} max|Wx+b - normalized| = {d:.2e}  "
                  f"{'OK' if d < 1e-3 else 'FAIL'}")

    # 4) scores
    edges = discover_edges(model, example_inputs, prunable)
    scores_by_mode = compute_scores(prunable, sigma, mu, edges, fire_order, p=args.p)
    align_ok = assert_alignment(model, example_inputs, prunable, scores_by_mode["nci"])
    print(f"[align] input-score -> physical channel mapping: {'OK' if align_ok else 'CHECK'}")
    for mode, sc in scores_by_mode.items():
        # cross-LAYER drift = spread of per-layer MEAN score across depth (the nonrel
        # concern). Per-channel min is dominated by dead channels -> uninformative.
        means = torch.tensor([v.float().mean().item() for v in sc.values()])
        drift = (means.max() / means.clamp_min(1e-20).min()).item()
        print(f"[score] {mode:<7s} cross-layer drift (max/min of per-layer mean) = {drift:.2e}")

    # 5) prune + ft per mode  (scores keyed by NAME -> survive deepcopy)
    scores_named = {mode: {mod2name[L]: v for L, v in sc.items()} for mode, sc in scores_by_mode.items()}
    prunable_names = list(prunable.keys())
    rows = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        args.mode = mode
        glob = True if mode == "magnitude" else (args.global_prune and mode != "rel")
        print(f"\n=== mode: {mode} (pruning_ratio={args.pruning_ratio}, global={glob}) ===")
        row = prune_and_eval(model, example_inputs, prunable_names, mod2name[first_conv],
                             mod2name[classifier], scores_named.get(mode), args,
                             train_loader, val_loader, device)
        rows.append(row)

    # 6) report
    print("\n" + "=" * 78)
    print(f"RESULTS  {args.model} / {args.dataset}  | base acc {acc0:.2f} | "
          f"{params0/1e6:.2f}M params {macs0/1e6:.1f}M MACs")
    print("-" * 78)
    print(f"{'mode':<8}{'params(M)':>11}{'-%':>7}{'MACs(M)':>10}{'-%':>7}"
          f"{'acc(noFT)':>11}{'acc(FT)':>10}")
    for r in rows:
        print(f"{r['mode']:<8}{r['params']:>11.2f}{r['params_drop']:>7.1f}"
              f"{r['macs']:>10.1f}{r['macs_drop']:>7.1f}{r['acc_pre_ft']:>11.2f}{r['acc_ft']:>10.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
