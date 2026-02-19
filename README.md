# SIRC Hardware-Aware Network Optimization (SHANO)

An easy-to-integrate framework for optimizing internal networks. It accelerates inference with minimal task degradation by removing redundant parameters. Currently supports:

- **Structured (Channel) Pruning** — removing entire channels (filters) for universal acceleration.
- **Semi-Structured (Slice) Pruning** — removing parts of filters for further reduction.

## Project Objectives

- **Reduce Model Size** — prune redundant parameters while preserving accuracy.
- **Accelerate Inference** — lower computational overhead for faster deployment.
- **Enhance Flexibility** — combine multiple pruning methods for varied architectures and hardware.

## Project Structure

- **`torch_pruning/`** — core library (original Torch-Pruning + our enhancements for channel and slice pruning).
- **`torch_pruning/utils/pruning_utils.py`** — orchestrates the pruning process (initialization, regularization, epoch-wise pruning).
- **`Documentation/`** — usage examples and method descriptions.
  - [Documentation/README.md](Documentation/README.md) — complete guidance.
  - [Documentation/pruning_methods_overview.md](Documentation/pruning_methods_overview.md) — detailed pruning methods overview.

## Installation

### Docker Image Integration

Add SSH config and install the library inside your Docker image:

```dockerfile
RUN mkdir -p ~/.ssh \
    && echo "Host *" > ~/.ssh/config \
    && echo "    StrictHostKeyChecking no" >> ~/.ssh/config \
    && chmod 600 ~/.ssh

RUN --mount=type=ssh $PIP_INSTALL git+ssh://git@gitlab-srv/avrahamra/sirc_torch_pruning@v2
```

## Usage

### Minimal Code Integration

1. **Initialization** — instantiate the pruner with your model:
   ```python
   from torch_pruning.utils import Pruning
   pruner = Pruning(model, output_dir, device=device)
   ```

2. **Regularization** — after the backward pass, add pruning regularization:
   ```python
   loss.backward()
   pruner.channel_regularize(model)
   ```

3. **Pruning Step** — call the prune method at each epoch:
   ```python
   pruner.prune(model, epoch)
   ```

### DOF Integration

1. Clone the DOF repo:
   ```bash
   git clone http://gitlab-srv/danr/nndof.git
   git checkout main_thetis
   ```

2. Add torch-pruning support (if needed), as explained in the [DOF integration guide](http://gitlab-srv/avrahamra/SHANO/-/blob/v2/Documentation/examples/DOF.txt?ref_type=heads).

3. Save `pruning_config.json` in your output directory (see [example config](http://gitlab-srv/avrahamra/SHANO/-/blob/v2/Documentation/examples/pruning_config.json?ref_type=heads)).

4. Run with Docker:
   ```bash
   /algo/ws/shared/remote-gpu/run_docker_gpu.sh \
     -q gpu_deep_train_high_q \
     -d gitlab-srv.transchip.com:4567/danr/nndof/ub22_py310_cu118_pt212_plus_qat:12021_tp2 \
     -C execute -W working_dir -M nndof/train.py \
     -s 32gb -n 8 \
     -A '-o <output_dir> -c config/train/rear_mfc_thetis_124.yaml' \
     -p VISION \
     -v /algo/NetOptimization:/algo/NetOptimization \
     -D 'dof + torch pruning'
   ```

### Benchmark (Standalone)

A standalone script at `benchmarks/vbp/vbp_imagenet_pat.py` supports multiple networks with full `pruning_utils` features. Configuration is handled via CLI arguments (which build `pruning_config.json` automatically).

To reproduce VBP paper results:

```bash
/algo/ws/shared/remote-gpu/run_docker_gpu.sh \
  -d gitlab-srv.transchip.com:4567/od-alg/od_next_gen:v1.7.6_tp2 \
  -q gpu_deep_train_high_q \
  -C execute -W working_dir -M benchmarks/vbp/vbp_imagenet_pat.py \
  -s 16gb -n 8 \
  -A '--disable_ddp --model_type vit --model_name /algo/NetOptimization/outputs/VBP/DeiT_tiny --save_dir <output_dir> --epochs 10 --global_pruning --pat_steps 0' \
  -p VISION \
  -v /algo/NetOptimization:/algo/NetOptimization \
  -R "select[gpu_hm48]" \
  -E force_python_3=yes
```

## Backlog

**Code:**
- IP-free layer handling
- User manual (integration cookbook + stage analysis guidelines)
- Remove redundant/overloaded arguments

**Algorithmic:**
- Split regularization and task loss
- HW-aware pruning

## Updates & Support

- **Release notes:** see [weekly updates](https://confluence.transchip.com/display/SOCC/HW+aware+Network+Optimization) for progress and roadmap.
- **Contact:** [Avraham](mailto:avraham.r@samsung.com), [Yonatan](mailto:yonatan.dina@samsung.com), [Ishay](mailto:ishay.goldin@samsung.com)
