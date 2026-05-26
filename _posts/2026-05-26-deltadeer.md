---
layout: distill
title: "DeltaDEER: When models wear antlers"
description: "Using Newton's method to parallelize nonlinear recurrence of DeltaNet"
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
  - name: Lorenzo Ruggeri

bibliography: 2026-05-26-deltadeer.bib

toc:
  - name: Introduction
  - name: The Fundamental Tradeoff
  - name: From Attention to RNNs
  - name: The Delta Rule
  - name: Adding Nonlinearity to DeltaNet
  - name: Parallelizing with DEER
    subsections:
      - name: Newton's Method for Sequences
      - name: The Jacobian Structure
  - name: Implementation using Triton
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

<figure>
  <img src="/assets/img/deltadeer/deltadeer.png" alt="DeltaDEER architecture combining parallel training with efficient inference" style="width: 100%;">
  <figcaption>
    <strong>DeltaDEER: when models wear antlers.</strong>
  </figcaption>
</figure>

## Introduction

Transformers have conquered natural language processing, but they come with an uncomfortable secret: they're terrible at inference. While attention-based models train in parallel with impressive throughput, generating a single token at a time requires scanning the entire past—an operation that scales quadratically with sequence length. Meanwhile, an old idea keeps whisper in the dark: RNNs. Notoriously slow to train but lightning-fast at inference, they compute the next token in constant time, independent of context length.

This post introduces **DeltaDEER**: a method that asks what if you could have both? What if you could take the linearized structure of attention—expressed through a simple recurrence rule borrowed from control theory—and recover the parallelizability of transformers using a classical technique from numerical optimization: **Newton's method**?

The name is a hint. DEER stands for "non-linear Differential Equation as fixed point itERation", a technique for parallelizing non-linear RNNs. And like a deer with antlers, DeltaDEER grows something beautiful out of a simple structure. You can find the code for DeltaDEER [here](https://github.com/lorenzoruggerii/muLoop).

## The Fundamental Tradeoff

To understand why DeltaDEER exists, we need to understand the asymmetry between training and inference.

**Transformers** excel at *training*: attention is parallelizable, allowing you to compute all timesteps simultaneously. But at *inference*, when you generate token-by-token, you accumulate a **KV cache**—a growing store of all previous keys and values. Computing attention over this cache costs $O(L^2)$ for a sequence of length $L$. Even if it's squared in the sequence length, it's fully parallelizable. The real bottleneck comes from the fact that the KV cache occupies $O(BHLD)$, and for large batch size $B$, and a high number of heads $H$, the dipendence from the sequence length $L$ prohibits to store such a large amount of data into the GPU HBM.

**RNNs** suffer the opposite fate. Their recurrent structure (hidden state updates sequentially, step by step) makes them painfully slow to train—you cannot parallelize across time. But at inference, they shine: each new token requires only a constant-time update to the hidden state, independent of sequence length. So, for a sequence of length $L$, you pay $O(L)$ time. The problem is that RNN state updates are usually nonlinear, which means that you cannot parallelize them across the time dimension.

The question becomes: can we design a model that combines the best of both worlds?

## From Attention to RNNs

The bridge between these two worlds is surprisingly simple. Attention, the core of transformers, can be rewritten as a linear recurrence. Consider multi-head attention applied to a single token at position $t$:

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

As seen previously, one of the bottlenecks of attention is the materialization of the $L \times L$ attention matrix $QK^T$, which require $O(L^2d)$ computations in its naive form. However, if one removes the sofmax: 

$$
y_t = \frac{\sum_{s=1}^{t} q_t k_s v_s}{\sum_{s=1}^{t} q_t k_s}
$$

now, using associativity of matrix multiplication, we can first multiply keys with values with the sequence axis as the reduction axis, paying $O(Ld^2)$ computations, before multiplying everything for $q_t$.

This is the idea behind linear attention paper, which explains how removing that softmax allows to recognize a recurrence into the attention calculation. Indeed, we can define a state $S_i = k_i v_i$, and turn our linearized attention into a recurrence.

$$
y_t = q_t S_t, \ \ S_t = S_{t-1} + k_tv_t
$$

The denominator can be handled by an auxiliary state that collects just the keys.

Now the real problem is that removing that softmax will decrease a lot the performances. Moreover, even if in theory linear attention is more efficient than softmax attention, we will need to wait until Flash Linear Attention paper to have a full implementation that can beat FlashAttention (which instead computes full softmax attention).

So, how can we improve this?

## The Delta Rule

Here's where things get interesting. Linear Attention is built on the idea that removing the softmax allows to improve the (theoretical) efficiency of attention. The real problem is that it is not built on some underlying idea that can help on natural language tasks.

So, Instead of treating the RNN hidden state as a simple accumulator, we can view it as the solution to an optimization problem. Here comes another question: which **optimization problem** do we want to solve?

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/deltadeer/zoology.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig: 1. Associative Recall task.</strong>. Zoology paper describes associative recall as an important subtask for natural language processing performances. Associative recall means being able, given a key token, to successfully return the value token associated to that key (e.g. Key: Hakuna, Value: Matata).
</div>

People at Hazy Research showed in their Zoology paper that Associative Recall Helps a lot for Natural Language Processing tasks (**Figure 1**). So let's start optimizing for this task. How can we do this? As always in deep learning: **define a loss function**.

$$
\min_m \frac{1}{2} \sum_{i=0}^{T} \|v_i - m k_i\|^2
$$

At every timestep, given a key $k_i$, we compare the value returned by our RNN memory $m$ associated to that key with the corresponding value $v_i$. Of course we would like to find the best memory that minimizes the loss. So, let's apply SGD.

1. Calculate the gradient for timestep $t$: $G_t = -(v_t - m_t k_t)k_{t}^{T}$.
2. Apply SGD as state update: $m_{t+1} = m_t - \beta_{t}G_t$

This means that our update rule becomes:

$$
m_{t+1} = m_t + \beta_{t} (m_t k_t - v_t) k_t^T
$$

where:
- $m_t$ is the memory matrix at time $t$
- $v_t$ is the "value" signal (what we're trying to remember)
- $k_t$ is the "key" signal (what we're matching against)
- $\beta$ is a learning rate. In this case, I decided to use selective $\beta_{t}$ and to calculate it for each timestep.

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

This is exactly the delta rule (with or without nonlinearity) described in DeltaNet paper.

## Adding Nonlinearity to DeltaNet

The linear delta rule works, as demonstrated from DeltaNet results, where DeltaNet outperforms Transformers (and other linear RNNs like Mamba) on The Pile dataset at different scales (400M and 1.3B).

However, nonlinearities are essential to increase expressivity of a model. Nonlinear RNNs can solve tasks (like Parity), that linear RNNs cannot do, unless we do some tricks on the eigenvalues of the state update matrix.

So, why don't we add a nonlinearity to DeltaNet and see what happpens? The recurrent update now becomes:

$$
m_{t+1} = \sigma\left(m_t - \beta (m_t k_t - v_t) k_t^T\right)
$$

where $\sigma$ is an activation function like $\tanh$.

I tested it on a very tiny dataset, called TinyStories, to see how it performed against original DeltaNet and a Transformer. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/deltadeer/perplexity_comparison.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig: 2. Performances on TinyStories for Transformer, DeltaNet, and DeltaNet (with tanh).</strong>. Perplexity comparison across Transfomer, DeltaNet, and DeltaNet (with tanh) trained on TinyStories dataset. DeltaNet (with tanh) exhibits the lowest perplexity on the dataset.
</div>

The nonlinearity makes the model more expressive but destroys the key property we relied on: **parallelizability**. Now you cannot compute all timesteps independently; each step depends on the previous nonlinear activation. Training DeltaNet (with tanh) required a lot of time, compared to the parallel implementation for attention computation (torch uses _FlashAttention_). We're back to the RNN problem.

So, what we can do? Luckily for us, some literature efforts have been made for parallelizing non-linear state updates.

## Parallelizing with DEER

This is where **DEER** (non-linear Differential Equation as fixed point itERations) enters the scene. DEER is a method for parallelizing non-linear sequential models by leveraging a insight from classical numerical analysis: **Newton's method**.

### Newton's Method for Sequences

Consider finding a zero of a non-linear function $f$. Newton's method iterates:

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

This converges quadratically when you're near the solution. Now, imagine applying this idea across a sequence: instead of computing states sequentially as $s_t = f(s_{t-1}, x_t)$, we reframe the problem as finding values $s_0, s_1, \ldots, s_T$ that satisfy a system of constraints.

The idea starts from defining a _residual function_:

$$
r(s) = [s_1 - f(s_0), s_2 - f(s_1), \ldots , s_T - f(s_{T-1})]
$$

where $f$ is just the update rule done by the model.

Just by looking at $r(s)$, we can see that, if $r(s) = 0$, then we have found our model states $s_0, s_1, \ldots, s_T$. So, we can **apply Newton's method**!

To apply Newton's method, we need to compute the Jacobian of the given function, and invert it (the derivative is at the denominator in the formula). However, the Jacobian is a $TD \times TD$ matrix, where $T$ is the sequence length and $D$ is the dimensionality of each $s_i$. So, it's a huge matrix to store. Moreover, we would need to invert it, and this would take $O(T^3)$ time. Infeasible. And this is just for one step of Newton's method, and we don't know a priori how many iterations we need.

But, wait a minute... Let's look at the structure of the Jacobian and see whether we can find something interesting.


### The Jacobian Structure

The beauty of sequential dependencies is that they create a sparse Jacobian matrix. Indeed, from the residual, we can see that each entry $r_i$ depends only from $s_i$ and $s_{i-1}$. So, the Jacobian will be a block diagonal matrix:

$$
J(s) = \begin{pmatrix}
I_D & 0 & \cdots & 0 & 0 \\
\frac{\partial f}{\partial s}(s_1) & I_D & \cdots & 0 & 0 \\
\vdots & \ddots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & I_D & 0 \\
0 & 0 & \cdots & \frac{\partial f}{\partial s}(s_{T-1}) & I_D
\end{pmatrix}
$$

Moreover, starting from the original Newton's method:

$$
s_{n+1} = s_n - J^{-1}(r(s))r(s)
$$

we can reformulate it like this:

$$
J(r(s))\Delta s  = -r(s)
$$

avoiding to invert the Jacobian matrix.

Now we can use the **diagonal block structure** of the Jacobian to see which kind of linear system we need to solve. Unrolling the above linear system yields us, for every timestep $t$, at iteration $n$:

$$
\Delta s_t^{(n)} = \frac{\partial f}{\partial s}(s_{t-1}^{(n)}) \Delta s_{t-1}^{(n-1)} - r_{t}(s^{(i)})
$$

But, if you see closer, this is now a **Linear recurrence!**. This means that we can apply parallel algorithms like Blelloch scan to solve this in parallel. This means that now each iteration can be solved in $O(logT)$ time instead of $O(T)$, which for huge sequences is a big win!

There is a subtlety, however. The cost of a parallel scan depends from which update rule we want to use. Here, each update costs a matrix multiplication between two matrices of dimensionality $D \times D$, which is $O(D^3)$. However, we can surpass this by using element-wise multiplication. This is ok, since DEER's convergence is not strictly linked to exact computation of the Jacobians. This means that we can reduce the work done from each step of the parallel scan, without hurting convergence too much.

Moreover, the success of DEER is highly dependent from the **number of iterations** we need to make to get the convergence (i.e. $r(s) = 0$). Luckily (for me), 20 iterations were sufficient to get the convergence for DeltaNet (with tanh). This means that for $L >> 20$, we get a significant speedup.

Now it's time to get things done. The core DEER iteration loop looks like this:

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

## Implementation using Triton

To make DEER efficient, we can't afford to compute residuals and Jacobians naively. Instead, we use **Triton**, a language for writing GPU kernels, to fuse the computation.

In particular, I first started writing the kernel for calculating the residuals and the jacobians together. This is because these two quantities required the same inputs, so it's convenient to fuse their computations together to avoid reading/writing too much from the GPU HBM.

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

Then, the remaining part is just to combine all using an associative scan. I've used triton's native associative scan function to handle this.

So, after all this, GPUs are going brrr?

## Experimental Results

I tried DeltaDEER with a bunch of sequence lengths, for an input with a reasonable number of heads $H=8$ and a reasonable head dimension $D=32$. 

First of all, I checked whether my algorithm was converging to the ground truth (i.e. the sequential scan).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/deltadeer/convergence.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig: 3. Convergence of DeltaDEER.</strong>. Absolute error between DeltaDEER and DeltaNet (with tanh) in sequential mode. The absolute error is around $1e^{-6}$ for all timesteps.
</div>

Luckily, it converges. After 20 iterations, DeltaDEER converges to the result obtained using the sequential scan. But how faster are we going?

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/deltadeer/scalability.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig: 4. Scalability of DeltaDEER (forward pass).</strong>. Running times for different sequence lengths. DeltaDEER achieves a 10-12x stable speedup in the forward pass with respect to torch sequential implementation.
</div>

Yeah! We have achieved a 10-12x speedup over torch implementation. From the running times, this means that now running DeltaNet (with tanh) using a huge sequence length is now feasible.

## Conclusions

DeltaDEER shows that the transformer-RNN tradeoff is not inevitable. By grounding the RNN recurrence in the delta rule from optimization and applying DEER's parallelization technique, we recover a model that balances both regimes.

In this moment, I only parallelized the forward pass. I think that there is a lot room for improvement (e.g. I think that I could apply a chunkwise parallel scan, or try to use other methods rather than DEER for parallelizing), but this is a nice starting point. Moreover, also the backward pass could be optimized.

The deeper insight is philosophical: attention, RNNs, and recurrent memory are not fundamentally different computation types. They are different *framings* of the same idea—accumulating information over time—and by choosing the right mathematical formalism, we can interpolate between them.

The antlers, it turns out, are just a clever way of saying: when you solve the right equation, beautiful structure emerges.
