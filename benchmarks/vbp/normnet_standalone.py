#!/usr/bin/env python
"""
Normalized-Net pruning — SELF-CONTAINED, ImageNet-1k, single GPU.

One file, end to end (weights from --ckpt; NO hub download):
    load ckpt  ->  normalize/reparam (calibrate sigma, fold)  ->  score (magnitude / NCI /
    rel / nonrel)  ->  prune (torch_pruning)  ->  [optional fine-tune]  ->  dump + report.

------------------------------------------------------------------------------
THE THREE SCORES (per INPUT channel of each prunable layer l)
------------------------------------------------------------------------------
Normalized net: before each target layer, input is normalized to zero-mean/unit-var
by its measured (mu, sigma); the weight is re-expressed as  W' = sigma * W  (fold).
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
import argparse, copy, json, os, sys, time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# repo root on sys.path so the in-repo torch_pruning/ package imports even when this file is
# run directly (python adds only the script dir, not cwd; run_ddp gets cwd via `python -m`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch_pruning as tp
from torch_pruning.pruner import function as tp_fn

IN_FNS  = (tp_fn.prune_conv_in_channels,  tp_fn.prune_linear_in_channels)
OUT_FNS = (tp_fn.prune_conv_out_channels, tp_fn.prune_linear_out_channels)

_LOG_FH = None   # set in main() -> <save_dir>/<tag>_progress.log


def log(msg):
    """Timestamped line to stdout (docker job log) AND <tag>_progress.log, flushed."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n"); _LOG_FH.flush()


# ----------------------------------------------------------------------------- device / data / model
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _FastImageNet(torch.utils.data.Dataset):
    """Mirror of vbp_common.FastImageNet: samples = list[(path, label)], PIL default_loader."""
    def __init__(self, samples, transform):
        from torchvision.datasets.folder import default_loader
        self.samples, self.transform, self.loader = samples, transform, default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(self.loader(path)), label


def build_data(args):
    """ImageNet-1k only. Cached-pickle FastImageNet (<data_path>/{train,val}_samples.pkl,
    matches vbp_common.build_dataloaders), ImageFolder (<data_path>/val) fallback. The val
    pipeline (resize <val_resize> / center-crop 224) serves both eval and sigma calibration
    (= build_calib_loader: val transform on TRAIN samples, no aug)."""
    norm = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.val_resize),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(), norm])
    train_pkl = os.path.join(args.data_path, "train_samples.pkl")
    val_pkl   = os.path.join(args.data_path, "val_samples.pkl")
    if os.path.isfile(val_pkl):                       # cluster fast path (cached pickle)
        import pickle
        log(f"data: loading cached pickle val={val_pkl}")
        with open(val_pkl, "rb") as f:
            val_set = _FastImageNet(pickle.load(f), tf)
        calib_pkl = train_pkl if os.path.isfile(train_pkl) else val_pkl
        log(f"data: loading cached pickle calib={calib_pkl}")
        with open(calib_pkl, "rb") as f:
            train_set = _FastImageNet(pickle.load(f), tf)
    elif os.path.isdir(os.path.join(args.data_path, "val")):   # ImageFolder fallback
        log(f"data: ImageFolder scan {args.data_path}/val (slow; no pkl found)")
        val_set = torchvision.datasets.ImageFolder(os.path.join(args.data_path, "val"), tf)
        cdir = "train" if os.path.isdir(os.path.join(args.data_path, "train")) else "val"
        train_set = torchvision.datasets.ImageFolder(os.path.join(args.data_path, cdir), tf)
    else:
        raise SystemExit(f"[data] no imagenet data at {args.data_path}: expected "
                         f"val_samples.pkl (cached) or val/ (ImageFolder)")
    log(f"data: imagenet ready (calib {len(train_set)} / val {len(val_set)})")
    pin = args.device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=pin)
    return train_loader, val_loader


def build_model(args):
    name = args.model
    # NEVER download from the hub (offline cluster -> hangs). Build the imagenet-1k arch
    # (1000-way head, no swap) with random weights; real weights come from --ckpt. Same as
    # vbp_common.load_model (model_fn(pretrained=False) then load checkpoint).
    model = getattr(torchvision.models, name)(weights=None)
    if name.startswith("resnet"):
        first_conv, classifier = model.conv1, model.fc
    elif name.startswith("convnext"):
        first_conv, classifier = model.features[0][0], model.classifier[2]
    else:
        raise ValueError(f"unsupported model {name}")
    # load weights from --ckpt (the only source). Handles {state_dict|model} wrappers and a
    # leading 'module.'; strict=False tolerates head differences.
    if getattr(args, "ckpt", None):
        if not os.path.isfile(args.ckpt):
            raise SystemExit(f"[ckpt] not found: {args.ckpt}")
        try:
            sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict):
            sd = sd.get("state_dict", sd.get("model", sd))
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        # drop shape-mismatched keys (e.g. a swapped classifier head) so load doesn't raise
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"ckpt loaded {args.ckpt} ({len(sd)} tensors; missing {len(missing)}, "
            f"unexpected {len(unexpected)})")
    else:
        log(f"WARNING: no --ckpt -> RANDOM weights for {name}; scores are meaningless. "
            f"Provide an imagenet checkpoint.")
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
        print(f"[{tag}] epoch {ep+1}/{epochs}  loss {loss:.3f}  train_acc {acc:5.2f}  "
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
def fold_bn(model, example_inputs):
    """Fold every Conv2d/Linear -> BatchNorm adjacency into the layer weights; replace the
    BN with Identity. Function-preserving (uses running stats; call in eval()).

    BN is a real per-channel operator (γ/σ_run scale) in the forward chain. If it's NOT
    folded, the score either ignores it (raw-W magnitude) or smuggles it in via the
    calibrated activation σ — entangling weight scale with activation scale. Folding bakes
    γ/σ_run into the weight so M is honest and the propagation chain is pure Conv/Linear.

    Adjacency is found by TRACING (data_ptr match), not by walking nn.Sequential — so it
    also catches ResNet bottleneck BNs (conv1->bn1 ... are block attributes, NOT in a
    Sequential, which the Sequential-only fold_all_conv_bn misses). ConvNeXt has no BN
    (LayerNorm) -> folds 0, logits unchanged."""
    model.eval()
    producers = {}      # data_ptr(conv/linear output) -> module
    pairs = []          # (conv_or_linear, bn) where bn consumes that output directly
    def prod_hook(mod, inp, out):
        producers[out.data_ptr()] = mod
    def bn_hook(mod, inp, out):
        p = producers.get(inp[0].data_ptr())
        if p is not None:
            pairs.append((p, mod))
    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(prod_hook))
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            handles.append(m.register_forward_hook(bn_hook))
    model(example_inputs)
    for h in handles:
        h.remove()

    name2mod = dict(model.named_modules())
    mod2name = {m: n for n, m in name2mod.items()}
    folded = 0
    for conv, bn in pairs:
        sigma = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight if bn.affine else torch.ones_like(bn.running_mean)
        beta  = bn.bias   if bn.affine else torch.zeros_like(bn.running_mean)
        scale = gamma / sigma                                   # [C_out]
        if isinstance(conv, nn.Conv2d):
            conv.weight.data.mul_(scale[:, None, None, None])
        else:
            conv.weight.data.mul_(scale[:, None])
        b = conv.bias.data if conv.bias is not None else torch.zeros(scale.numel(), device=conv.weight.device)
        b_eff = scale * (b - bn.running_mean) + beta
        if conv.bias is None:
            conv.bias = nn.Parameter(b_eff.detach().clone())
        else:
            conv.bias.data.copy_(b_eff)
        parent_name, _, child = mod2name[bn].rpartition(".")
        setattr(name2mod[parent_name] if parent_name else model, child, nn.Identity())
        folded += 1
    return folded


@torch.no_grad()
def calibrate(model, prunable, loader, device, n_batches, save_input_for):
    """One pass: per prunable layer capture INPUT mean/std (mu_in, sigma_in) per input
    channel, plus forward FIRING ORDER (a valid topo sort of the prunable DAG).
    `save_input_for` modules also stash one input batch (for the fold sanity check).

    The non-relative transfer Σ^{l+1} is the POST-activation std = the immediate on-path
    CONSUMER's input σ (built in compute_scores from sigma_in), NOT a layer's own output
    std — that own-output std is PRE-activation (= the D denominator) and would cancel D,
    collapsing nonrel to the raw M^p product. So we calibrate only the input σ here."""
    stats = {m: {"sum": None, "sqsum": None, "n": 0} for m in prunable.values()}
    fire_order = []
    seen = set()
    saved_inputs = {}

    def pre_hook(mod, inp):
        x = inp[0].detach()
        if mod not in seen:
            fire_order.append(mod); seen.add(mod)
        # conv-type -> NCHW reduce (dim 1 channels); else channels-LAST (dim -1)
        if isinstance(mod, nn.Conv2d):
            s = x.sum(dim=(0, 2, 3)); sq = (x * x).sum(dim=(0, 2, 3)); cnt = x.numel() // x.size(1)
        else:
            xf = x.reshape(-1, x.size(-1)); s = xf.sum(0); sq = (xf * xf).sum(0); cnt = xf.size(0)
        st = stats[mod]
        st["sum"]   = s  if st["sum"]   is None else st["sum"]   + s
        st["sqsum"] = sq if st["sqsum"] is None else st["sqsum"] + sq
        st["n"]    += cnt
        if mod in save_input_for and mod not in saved_inputs:
            saved_inputs[mod] = x[:4].clone()       # tiny slice for the assert

    handles = [m.register_forward_pre_hook(pre_hook) for m in prunable.values()]
    model.eval()
    log(f"calib: starting {n_batches} batches on {device} ...")
    bi = 0
    for x, _ in loader:
        model(x.to(device))
        bi += 1
        if bi == 1 or bi % 10 == 0 or bi >= n_batches:
            log(f"calib: batch {bi}/{n_batches}")
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
    """W' = sigma*W, m = b + W@mu  reproduces  W x + b  on normalized input. Function-preserving."""
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
def compute_scores(prunable, sigma, edges, fire_order, p=2):
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
    # Σ^{l+1} transfer = POST-activation output std (PDF steps 3-4: f = σ_post/σ_pre, the
    # activation std-gain). The layer's OWN output (sigma_out) is PRE-activation (= σ_pre =
    # the colsum/D denominator) → using it makes D and the transfer CANCEL, collapsing
    # nonrel to the raw M^p product (the 1e9 compound). The post-activation std equals the
    # IMMEDIATE on-path consumer's input std (a_L feeds straight into it). Pick the consumer
    # earliest in fire_order (the main path), NOT an average over off-path residual consumers.
    order_idx = {L: k for k, L in enumerate(fire_order)}
    sigma_post = {}
    for L in prunable.values():
        cands = [C for C in edges[L]
                 if sigma[C].numel() == n_out(L) and order_idx.get(C, 1 << 30) > order_idx.get(L, -1)]
        if cands:
            C = min(cands, key=lambda c: order_idx[c])
            sigma_post[L] = sigma[C]                       # consumer input std = σ_post[L]
        else:
            sigma_post[L] = torch.ones(n_out(L), device=M[L].device)   # terminal → no transfer

    # nci/mag are VARIANCE/L2 BY DEFINITION (the paper's NCI = σ_i²‖W_i‖²) — fixed exponent 2,
    # independent of `p`. `p` only flavors the PROPAGATION (rel/nonrel: p=2 variance / p=1 std).
    # So a `--p 1` dump pairs L1 propagation with the (unchanged) L2 nci/mag baselines on purpose.
    nci = {L: (M[L].pow(2)).sum(dim=1) for L in prunable.values()}   # one hop = sigma_i^2||W_i||^2
    # plain magnitude on the SAME input axis (no sigma): mag_i = ||W[:,i]||.  nci = sigma^2 mag^2
    mag = {L: (M[L] / sigma[L][:, None]).pow(2).sum(dim=1).sqrt() for L in prunable.values()}

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

    return {"magnitude": mag, "nci": nci, "rel": propagate(False), "nonrel": propagate(True)}


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


# ----------------------------------------------------------------------------- ExpHandler dump
def dump_channel_scores(path, model, scorer, layer_scores, stage=None, kept=None,
                        higher_is_better=True):
    """Write one channel_scores/v1 JSON for the ExpHandler 'Channels' viz.
    layer_scores: dict[name -> 1-D iterable of RAW per-channel scores] (no normalization;
    the app does per-layer / per-network normalization). kept: dict[name -> bool iterable]."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    layers = [{"name": n, "scores": [float(x) for x in s],
               **({"kept": [bool(x) for x in kept[n]]} if kept and n in kept else {})}
              for n, s in layer_scores.items()]
    with open(path, "w") as f:
        json.dump({"schema": "channel_scores/v1", "model": model, "scorer": scorer,
                   "stage": stage, "higher_is_better": higher_is_better, "layers": layers}, f)
    return path


# ----------------------------------------------------------------------------- prune one config
def prune_and_eval(base_model, example_inputs, prunable_names, first_conv_name, classifier_name,
                   mode_scores, args, train_loader, val_loader, device):
    model = copy.deepcopy(base_model).to(device)
    name2mod = dict(model.named_modules())
    mod2name = {m: n for n, m in name2mod.items()}
    ignored = [name2mod[first_conv_name], name2mod[classifier_name]]
    # Every criterion (incl. magnitude) goes through the SAME input-side path so the dumped
    # scores match the prune decision exactly. The per-layer normalizer ("mean"->each layer
    # mean-1, or "lamp") erases native cross-layer scale and equalizes layer ordering -- the
    # trick that makes GLOBAL pruning work in practice; "none" keeps raw scale.
    in_scores = {name2mod[n]: mode_scores[n] for n in prunable_names if n in mode_scores}
    orig_in = {name2mod[n]: len(mode_scores[n]) for n in prunable_names if n in mode_scores}
    imp = NormScoreImportance(in_scores, normalizer=args.normalizer)

    macs0, params0 = tp.utils.count_ops_and_params(model, example_inputs)
    pruner = tp.pruner.MetaPruner(model, example_inputs, importance=imp,
                                  global_pruning=args.global_prune,
                                  pruning_ratio=args.pruning_ratio,
                                  ignored_layers=ignored)
    # interactive step -> record which INPUT channels each layer lost (the kept mask)
    pruned_in = {}
    for group in pruner.step(interactive=True):
        for dep, idxs in group:
            lay, fn = dep.layer, dep.pruning_fn
            if fn in IN_FNS and lay in in_scores:
                pruned_in.setdefault(lay, set()).update(int(i) for i in idxs)
        group.prune()
    macs1, params1 = tp.utils.count_ops_and_params(model, example_inputs)

    if args.dump_scores:
        kept = {mod2name[lay]: [i not in pruned_in.get(lay, set()) for i in range(n_in)]
                for lay, n_in in orig_in.items()}
        layer_scores = {n: mode_scores[n] for n in prunable_names if n in mode_scores}
        norm = args.normalizer if args.normalizer else "raw"
        cfg = f"g{int(args.global_prune)}_{norm}"
        fp = dump_channel_scores(
            os.path.join(args.save_dir, f"{args.tag}_{args.mode}_{cfg}_channel_scores.json"),
            args.model, args.mode, layer_scores, stage="pre_prune", kept=kept)
        log(f"mode {args.mode}: dumped {os.path.basename(fp)} "
            f"(params -{100*(1-params1/params0):.1f}%)")

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
    ap.add_argument("--data_path", required=True, help="imagenet root with {train,val}_samples.pkl (or val/)")
    ap.add_argument("--ckpt", default=None, help="imagenet-1k state_dict (no hub download). "
                                                 "None -> random weights (scores meaningless)")
    ap.add_argument("--val_resize", type=int, default=256, help="imagenet val resize (v2 used 232)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs_ft", type=int, default=0, help="post-prune fine-tune epochs (0 = dumps only)")
    ap.add_argument("--lr_ft", type=float, default=0.005)
    ap.add_argument("--pruning_ratio", type=float, default=0.5)
    ap.add_argument("--normalizer", default="mean",
                    choices=["mean", "lamp", "max", "sum", "standarization", "gaussian", "none"],
                    help="per-layer score normalizer for GLOBAL pruning. 'mean' (tp default, "
                         "each layer->mean-1) or 'lamp' (SOTA) equalize layer ordering; 'none' "
                         "keeps raw cross-layer scale (true-global NCI).")
    ap.add_argument("--global_prune", action="store_true", default=True,
                    help="global ranking across layers (default, best practice)")
    ap.add_argument("--local", dest="global_prune", action="store_false",
                    help="per-layer fixed-ratio pruning instead of global")
    ap.add_argument("--calib_batches", type=int, default=10)
    ap.add_argument("--p", type=int, default=2, choices=[1, 2])
    ap.add_argument("--modes", default="magnitude,nci,rel,nonrel")
    ap.add_argument("--limit_batches", type=int, default=0, help="cap batches/epoch (smoke test)")
    ap.add_argument("--dump_scores", action="store_true", default=False,
                    help="write per-criterion channel_scores JSON (ExpHandler 'Channels' viz)")
    ap.add_argument("--save_dir", default="./channel_dumps")
    ap.add_argument("--tag", default=None, help="dump filename prefix (default = model name)")
    args = ap.parse_args()
    if args.normalizer == "none":
        args.normalizer = None      # raw scores -> true cross-layer (no per-layer equalize)
    if args.tag is None:
        args.tag = args.model

    global _LOG_FH
    os.makedirs(args.save_dir, exist_ok=True)
    _LOG_FH = open(os.path.join(args.save_dir, f"{args.tag}_progress.log"), "a")

    args.device = device = pick_device()
    log(f"== START tag={args.tag} model={args.model} imagenet "
        f"device={device} cuda_avail={torch.cuda.is_available()} ==")
    if device.type != "cuda":
        log("WARNING: not on CUDA — GPU will look idle and resnet50 calibration will be slow")

    log("loading data ...")
    train_loader, val_loader = build_data(args)
    log(f"building model {args.model} (ckpt={args.ckpt}) ...")
    model, first_conv, classifier = build_model(args)
    model = model.to(device)
    log("model ready")
    example_inputs = torch.randn(1, 3, 224, 224, device=device)

    # fold native BN into the preceding Conv/Linear BEFORE scoring: bakes γ/σ_run into the
    # weight so M is honest and the chain is pure Conv/Linear (boss's "fold BN back").
    # Function-preserving -> logits must be unchanged.
    model.eval()
    ref = model(example_inputs).detach().clone()
    n_bn = fold_bn(model, example_inputs)
    dlogit = (ref - model(example_inputs)).abs().max().item()
    log(f"fold_bn: {n_bn} BNs folded into Conv/Linear | max|Δlogit| = {dlogit:.2e} "
        f"{'OK' if dlogit < 1e-3 else 'FAIL (fold not function-preserving!)'}")

    # base = loaded checkpoint (no training; weights come from --ckpt)
    _, acc0 = run_epoch(model, val_loader, device, limit=args.limit_batches)
    macs0, params0 = tp.utils.count_ops_and_params(model, example_inputs)
    log(f"base acc {acc0:.2f} (limit_batches={args.limit_batches}) | "
        f"{params0/1e6:.2f}M params | {macs0/1e6:.1f}M MACs")

    # 2) normalize / calibrate
    prunable = select_prunable(model, first_conv, classifier)
    name2mod = dict(model.named_modules())
    mod2name = {m: n for n, m in name2mod.items()}
    fold_check = list(prunable.values())[:2]
    mu, sigma, fire_order, saved = calibrate(model, prunable, train_loader, device,
                                             args.calib_batches, save_input_for=fold_check)
    log(f"calib done: {len(prunable)} prunable layers | fire-order len {len(fire_order)}")

    # 3) fold function-preservation check
    for L in fold_check:
        if L in saved:
            d = assert_fold_preserving(L, mu[L], sigma[L], saved[L])
            print(f"[fold] {mod2name[L]:<40s} max|Wx+b - normalized| = {d:.2e}  "
                  f"{'OK' if d < 1e-3 else 'FAIL'}")

    # 4) scores
    log("discovering layer graph (dependency) ...")
    edges = discover_edges(model, example_inputs, prunable)
    log("computing scores (magnitude/nci/rel/nonrel) ...")
    scores_by_mode = compute_scores(prunable, sigma, edges, fire_order, p=args.p)
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
        log(f"mode {mode}: pruning (ratio={args.pruning_ratio}, "
            f"global={args.global_prune}, normalizer={args.normalizer}) ...")
        row = prune_and_eval(model, example_inputs, prunable_names, mod2name[first_conv],
                             mod2name[classifier], scores_named.get(mode), args,
                             train_loader, val_loader, device)
        rows.append(row)

    # 6) report
    print("\n" + "=" * 78)
    print(f"RESULTS  {args.model} / imagenet  | base acc {acc0:.2f} | "
          f"{params0/1e6:.2f}M params {macs0/1e6:.1f}M MACs")
    print("-" * 78)
    print(f"{'mode':<8}{'params(M)':>11}{'-%':>7}{'MACs(M)':>10}{'-%':>7}"
          f"{'acc(noFT)':>11}{'acc(FT)':>10}")
    for r in rows:
        print(f"{r['mode']:<8}{r['params']:>11.2f}{r['params_drop']:>7.1f}"
              f"{r['macs']:>10.1f}{r['macs_drop']:>7.1f}{r['acc_pre_ft']:>11.2f}{r['acc_ft']:>10.2f}")
    print("=" * 78)
    log(f"ALL DONE: {len(rows)} modes dumped -> {args.save_dir}")


if __name__ == "__main__":
    main()
