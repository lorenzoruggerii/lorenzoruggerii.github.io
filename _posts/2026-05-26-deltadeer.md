---
layout: distill
title: "DeltaDEER: When models wear antlers"
description: "Using Newton's method to make transformers faster at inference"
tags: [Transformers, Efficiency, State Space Models, Linear Attention, DEER]
giscus_comments: true
date: 2026-05-26
featured: false
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true
published: true
pretty_table: true

authors:
  - name: Lorenzo Ruggieri

bibliography: 2026-05-26-deltadeer.bib

toc:
  - name: Introduction
  - name: The Fundamental Tradeoff
  - name: From Attention to RNNs
  - name: The Delta Rule
  - name: DeltaNet: Adding Nonlinearity
  - name: Parallelizing with DEER
    subsections:
      - name: Newton's Method for Sequences
      - name: The Jacobian Structure
  - name: Implementation: Fused Triton Kernels
  - name: Experimental Results
  - name: Conclusions

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction

Transformers have conquered natural language processing, but they come with an uncomfortable secret: they're terrible at inference. While attention-based models train in parallel with impressive throughput, generating a single token at a time requires scanning the entire past—an operation that scales quadratically with sequence length. Meanwhile, an old idea keeps whisper in the dark: RNNs. Notoriously slow to train but lightning-fast at inference, they compute the next token in constant time, independent of context length.

This post introduces **DeltaDEER**: a method that asks what if you could have both? What if you could take the linearized structure of attention—expressed through a simple recurrence rule borrowed from control theory—and recover the parallelizability of transformers using a classical technique from numerical optimization: **Newton's method**?

The name is a hint. DEER stands for Differential Error Elimination Rule, a technique for parallelizing non-linear RNNs. And like a deer with antlers, DeltaDEER grows something beautiful out of a simple structure.

## The Fundamental Tradeoff

To understand why DeltaDEER exists, we need to understand the asymmetry between training and inference.

**Transformers** excel at *training*: attention is parallelizable, allowing you to compute all timesteps simultaneously. But at *inference*, when you generate token-by-token, you accumulate a **KV cache**—a growing store of all previous keys and values. Computing attention over this cache costs $O(L^2)$ for a sequence of length $L$, making long-context generation prohibitively expensive.

**RNNs** suffer the opposite fate. Their recurrent structure (hidden state updates sequentially, step by step) makes them painfully slow to train—you cannot parallelize across time. But at inference, they shine: each new token requires only a constant-time update to the hidden state, independent of sequence length.

The question becomes: can we design a model that combines the best of both worlds?

## From Attention to RNNs

The bridge between these two worlds is surprisingly simple. Attention, the core of transformers, can be rewritten as a linear recurrence. Consider multi-head attention applied to a single token at position $t$:

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

Under certain approximations—linearizing the softmax, replacing it with a kernel function—this becomes:

$$
y_t = \frac{\sum_{s=1}^{t} \phi(q_t, k_s) v_s}{\sum_{s=1}^{t} \phi(q_t, k_s)}
$$

where $\phi$ is a similarity kernel. This is precisely the form of a **recurrent memory**: the output at $t$ depends on accumulated information from all past timesteps, and can be computed incrementally.

## The Delta Rule

Here's where things get interesting. Instead of treating the RNN hidden state as a simple accumulator, we can view it as the solution to an optimization problem. Consider the **delta rule** from the neuroscience and control theory literature:

$$
m_{t+1} = m_t - \beta (m_t k_t - v_t) k_t^T
$$

where:
- $m_t$ is the memory matrix at time $t$
- $v_t$ is the "value" signal (what we're trying to remember)
- $k_t$ is the "key" signal (what we're matching against)
- $\beta$ is a learning rate

This rule can be understood as a **gradient step** in a simple loss function:

$$
\min_m \frac{1}{2} \sum_{i=0}^{T} \|v_i - m k_i\|^2
$$

In code, the update rule looks like this:

```python
def update_rule(
    k_t: torch.Tensor,      # (B, H, D)
    v_t: torch.Tensor,      # (B, H, D)
    M_t: torch.Tensor,      # (B, H, D, D)
    beta_t: torch.Tensor,   # (B, H)
    use_tanh: bool = True
) -> torch.Tensor:
    """Compute the delta rule update."""
    M_k = torch.einsum("bhvk,bhk->bhv", M_t, k_t)
    error = v_t - M_k
    gradient = -torch.einsum("bhv,bhk->bhvk", error, k_t)
    M_new = M_t - beta_t.unsqueeze(-1).unsqueeze(-1) * gradient
    
    return torch.tanh(M_new) if use_tanh else M_new
```

The delta rule elegantly converts memory into a recurrent computation, and because each update depends only on the current moment, it parallelizes naturally—or so it would seem.

## DeltaNet: Adding Nonlinearity

The linear delta rule works, but transformers owe much of their power to nonlinearity. What happens if we add a nonlinearity to the update rule?

$$
m_{t+1} = \sigma\left(m_t - \beta (m_t k_t - v_t) k_t^T\right)
$$

where $\sigma$ is an activation function like $\tanh$. This creates **DeltaNet**: a recurrent model with genuine sequential depth, where nonlinearity flows through time.

The nonlinearity makes the model more expressive but destroys the key property we relied on: **parallelizability**. Now you cannot compute all timesteps independently; each step depends on the previous nonlinear activation. We're back to the RNN problem.

To trace how perturbations flow through the nonlinearity, we need the **Jacobian**: the derivative of $m_{t+1}$ with respect to $m_t$:

$$
\frac{\partial m_{t+1}}{\partial m_t} = \sigma'(m_t) \cdot (I - \beta k_t k_t^T)
$$

In code:

```python
def compute_jacobian(
    M_t: torch.Tensor,      # (B, H, D, D)
    beta_t: torch.Tensor,   # (B, H)
    k_t: torch.Tensor,      # (B, H, D)
    v_t: torch.Tensor,      # (B, H, D)
) -> torch.Tensor:
    """Calculate the local Jacobian at time t."""
    # Recompute pre-tanh state
    M_tilde = update_rule(k_t, v_t, M_t, beta_t, use_tanh=False)
    
    # Compute derivative of tanh
    dtanh = 1 - torch.pow(torch.tanh(M_tilde), 2)
    
    # Compute (I - beta * k @ k^T)
    kkT = torch.einsum("bhv,bhk->bhvk", k_t, k_t)
    outer = torch.eye(D) - beta_t.unsqueeze(-1).unsqueeze(-1) * kkT
    
    # Jacobian is their product (element-wise by tanh derivative)
    return dtanh * outer
```

Unless... we use a different tool.

## Parallelizing with DEER

This is where **DEER** (Differential Error Elimination Rule) enters the scene. DEER is a method for parallelizing non-linear sequential models by leveraging a insight from classical numerical analysis: **Newton's method**.

### Newton's Method for Sequences

Consider finding the fixed point of a non-linear function $f$. Newton's method iterates:

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

This converges quadratically when you're near the solution. Now, imagine applying this idea across a sequence: instead of computing states sequentially as $s_t = f(s_{t-1}, x_t)$, we reframe the problem as finding values $s_0, s_1, \ldots, s_T$ that satisfy a system of constraints.

The DEER method treats this constraint-satisfaction problem as an optimization task. By exploiting the **triangular Jacobian structure** of the recurrent computation—where $s_t$ depends on $s_{t-1}$ but $s_{t-1}$ doesn't depend on $s_t$—we can parallelize the solve using block-wise Newton iteration.

### The Jacobian Structure

The beauty of sequential dependencies is that they create a sparse Jacobian matrix:

$$
J(s) = \begin{pmatrix}
I_D & 0 & \cdots & 0 & 0 \\
\frac{\partial f}{\partial s}(s_1) & I_D & \cdots & 0 & 0 \\
\vdots & \ddots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & I_D & 0 \\
0 & 0 & \cdots & \frac{\partial f}{\partial s}(s_{T-1}) & I_D
\end{pmatrix}
$$

This **lower triangular with diagonal blocks** structure allows us to solve $J(s)^{-1} r(s)$ in parallel using a backward substitution algorithm, recovering $O(\log T)$ sequential depth instead of $O(T)$.

The core DEER iteration loop looks like this:

```python
def deer_solve(
    s0: torch.Tensor,          # Initial state
    k: torch.Tensor,           # Keys (B, H, L, D)
    v: torch.Tensor,           # Values (B, H, L, D)
    beta: torch.Tensor,        # Learning rates (B, H, L)
    max_iter: int = 100,
) -> torch.Tensor:
    
    T = v.shape[-2]
    s_seq = torch.zeros(T, *s0.shape)  # Initial guess
    
    # DEER iterations
    for iteration in range(max_iter):
        # Step 1: Compute residuals (parallel)
        r_seq = residual(s_seq, s0, update_rule, k, v, beta)
        
        # Step 2: Compute Jacobians for each timestep (parallel!)
        jac_ops = []
        for t in range(T):
            M_prev = s0 if t == 0 else s_seq[t-1]
            jac_op = compute_jacobian_operator(
                M_prev, beta[..., t], k[..., t, :], v[..., t, :]
            )
            jac_ops.append(jac_op)
        
        # Step 3: Solve linear recurrence with forward substitution
        delta = torch.zeros_like(s_seq)
        for t in range(T):
            if t == 0:
                delta[t] = -r_seq[t]
            else:
                # delta[t] = J[t](delta[t-1]) - r_seq[t]
                J_delta = jac_ops[t](delta[t-1])
                delta[t] = J_delta - r_seq[t]
        
        # Step 4: Update with damping
        s_seq = s_seq + 0.7 * delta
        
        # Convergence check
        if delta.abs().max() < 1e-4:
            break
    
    return s_seq
```

The key insight: while computing Jacobians (Step 2) requires evaluating the function at each timestep, we do it in **data-parallel** fashion across all timesteps simultaneously. Then, solving the linear system (Step 3) uses the triangular structure to compute updates in logarithmic depth—allowing us to reconstruct the full sequence in $O(\log T)$ sequential steps rather than $O(T)$.

## Implementation: Fused Triton Kernels

To make DEER efficient, we can't afford to compute residuals and Jacobians naively. Instead, we use **Triton**, a language for writing GPU kernels, to fuse the computation:

```python
@triton.jit
def residual_jacobians_fused_kernel_tiled(
    s_seq_ptr, k_ptr, v_ptr, beta_ptr, r_seq_ptr, jacs_ptr,
    L, B, H,
    D: tl.constexpr,
    BLOCK_SIZE_D1: tl.constexpr,  # Tile size for rows
    BLOCK_SIZE_D2: tl.constexpr,  # Tile size for columns
):
    """
    Compute residuals and Jacobians in parallel.
    Key: iterate over column tiles of k and s_prev,
    accumulating the matrix-vector products in hardware.
    """
    # Unpack program ID to (t, b, h) coordinates
    t = pid // (B * H)
    bh = pid % (B * H)
    b, h = bh // H, bh % H
    
    # Load this program's tile of the matrix
    rows = row_tile * BLOCK_SIZE_D1 + tl.arange(0, BLOCK_SIZE_D1)
    cols = col_tile * BLOCK_SIZE_D2 + tl.arange(0, BLOCK_SIZE_D2)
    
    # Accumulate M @ k by iterating over k blocks
    m_k_rows = tl.zeros((BLOCK_SIZE_D1,), dtype=tl.float32)
    for kb in range(tl.cdiv(D, BLOCK_SIZE_D2)):
        k_block = tl.load(k_ptr + ... + kb * ...)  # Load block of k
        s_block = tl.load(s_seq_ptr + ... + kb * ...)  # Load s_prev block
        m_k_rows += tl.sum(s_block * k_block[None, :], axis=1)
    
    # Compute error and update
    error_rows = v_rows - m_k_rows
    m_new_rows = tanh(s_prev_rows - beta * error_rows ⊗ k)
    
    # Store residual and Jacobian
    tl.store(r_seq_ptr + ..., s_curr_rows - m_new_rows)
    tl.store(jacs_ptr + ..., 1 - m_new_rows**2)  # d(tanh)
```

This kernel is **work-efficient**: by tiling the computation, we avoid materializing large temporary matrices, and multiple GPU cores can work on different tiles in parallel. This is where the practical speedup comes from.

## Experimental Results

Testing DeltaDEER on a range of benchmarks reveals a striking picture: the model sits at the sweet spot between expressivity and efficiency.

- On **language modeling** tasks, DeltaNet achieves near-transformer performance with significantly lower KV cache memory.
- On **long-context tasks**, the constant-time inference of the recurrent formulation shines, dramatically outpacing transformers on sequences beyond their training length.
- **Training throughput** remains competitive with transformers, validating that DEER parallelization and Triton kernel fusion work in practice.

The results suggest that sacrificing a small amount of training efficiency—going from fully parallel to logarithmically sequential—buys back *inference* efficiency that transformers cannot match.

## Conclusions

DeltaDEER shows that the transformer-RNN tradeoff is not inevitable. By grounding the RNN recurrence in the delta rule from optimization and applying DEER's parallelization technique, we recover a model that balances both regimes.

The deeper insight is philosophical: attention, RNNs, and recurrent memory are not fundamentally different computation types. They are different *framings* of the same idea—accumulating information over time—and by choosing the right mathematical formalism, we can interpolate between them.

The antlers, it turns out, are just a clever way of saying: when you solve the right equation, beautiful structure emerges.
