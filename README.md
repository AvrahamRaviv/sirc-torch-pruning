# SIRC Hardware-Aware Network Optimization

SIRC Hardware-Aware Network Optimization (SHANO) is an easy-to-integrate framework for optimizing internal networks. It accelerates inference with minimal task degradation by removing redundant parameters. Currently, it supports:
- **Structured (Channel) Pruning:** Removing entire channels (filters) for universal acceleration.
- **Semi-Structured (Slice) Pruning:** Removing parts of filters for further reduction.

## Project Objectives

- **Reduce Model Size:** Prune redundant parameters while preserving accuracy.
- **Accelerate Inference:** Lower computational overhead for faster deployment.
- **Enhance Flexibility:** Combine multiple pruning methods for varied architectures and hardware.

## Project Structure

- **torch_pruning:**  
  Contains the original Torch-Pruning code along with our enhancements for channel and slice pruning.


- **pruning_utils.py (under torch_pruning/utils):**  
  Core module orchestrating the pruning process (initialization, regularization, and epoch-wise pruning).

- **Documentation:**  
  In-depth explanations, usage examples, and detailed descriptions of supported methods.  
  - See [Documentation/README.md](Documentation/README.md) for complete guidance.
  - Detailed pruning methods overview is available in [Documentation/pruning_methods_overview.md](Documentation/pruning_methods_overview.md).

## Installation & Usage

### Installation

1. **Docker Image Integration**

   Clone the repository and install the library inside your Docker image:
   ```bash
   RUN mkdir -p ~/.ssh \
       && echo "Host *" > ~/.ssh/config \
       && echo "    StrictHostKeyChecking no" >> ~/.ssh/config \
       && chmod 600 ~/.ssh
   RUN --mount=type=ssh $PIP_INSTALL git+ssh://git@gitlab-srv/avrahamra/torch-pruning@main
   ```

### Minimal Code Integration

1. **Initialization:**  
   Instantiate the Pruner with your model:
   ```python
   from torch_pruning.utils import Pruning
   pruner = Pruning(model, output_dir, device=device)
   ```

2. **Regularization:**  
   After the backward pass, add pruning regularization:
   ```python
   loss.backward()
   pruner.channel_regularize(model)
   ```

3. **Pruning Step:**  
   Call the prune method at each epoch (or step):
   ```python
   pruner.prune(model, epoch)
   ```

## Updates & Support

- **Backlog and future release:**  
  Code issues:
  - IP-free layer handling.
  - User manual, including an integration cookbook and guidelines on how to analyze each stage and interpret the results.
  - Remove redundant and overloading arguments and params.

  Algorithmic issues:
  - Split regularization and task loss.
  - HW-Aware pruning.

- **Release Notes & Updates:**  
  See our [weekly updates](https://confluence.transchip.com/display/SOCC/HW+aware+Network+Optimization) for progress and roadmap details.

- **Contact:**  
  For bugs, feature requests, or inquiries, contact:
  - [Avraham](mailto:avraham.r@samsung.com)
  - [Yonatan](mailto:yonatan.dina@samsung.com)
  - [Ishay](mailto:ishay.goldin@samsung.com)
