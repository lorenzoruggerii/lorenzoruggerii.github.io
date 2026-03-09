---
layout: distill
title: "Mambacoders find interpretable Mamba feature circuits"
description: "Transcoders adapted to Mamba models"
tags: [Mechanistic Interpretability, AI, Mamba, SSMs]
giscus_comments: true
date: 2025-08-26
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

bibliography: 2026-03-04-mambacoders.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: What is a Transcoder?
  - name: Mamba SSM
  - name: The Mamba Block
  - name: How to apply Transcoders to Mamba?
  - name: Computing Attributions for Mamba layers
    subsections:
      - name: Attribution Between MambaCoder Features
      - name: Attribution from a MambaCoder Feature to a Mamba Mixer Head
      - name: Deriving the Attribution Formula
      - name: Handling the Convolution and Gate
  - name: An "hungry" example
    subsections:
      - name: Discovering the Feature
      - name: What Does the Feature Represent?
      - name: Steering the Model with the Feature
      - name: Tracing the Circuit
  - name: Conclusions

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
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

Mechanistic interpretability has made impressive strides on transformers — we now have sparse autoencoders, transcoders, and circuit-tracing algorithms that can identify human-interpretable features and trace how they interact. But the landscape of sequence modeling is shifting. Mamba has emerged as a compelling alternative to transformers, offering linear-time inference and competitive language modeling performance. More importantly, a growing number of frontier models are hybrid architectures <d-cite key="Nvidia2025"></d-cite>, interleaving Mamba layers with attention blocks, and if we want to understand these models mechanistically, we need interpretability tools that work for SSM layers too.
This post introduces MambaCoders: transcoders adapted for Mamba mixer blocks. I'll show how to train them, how to derive attribution formulas that account for Mamba's selective scan dynamics, and how to trace circuits across SSM heads. As a proof of concept, I'll walk through a fully interpretable, causally verified "hunger" circuit discovered in a small Mamba model trained on TinyStories: a concrete example of mechanistic interpretability working end-to-end on a pure SSM.

## What is a Transcoder?

If you want to deep dive, check out my [previous post on transcoders](https://lorenzoruggerii.github.io/blog/2025/transcoders-new/). In short: a transcoder <d-cite key="dunefsky2024"></d-cite> is a sparse, wide one-hidden-layer MLP trained to imitate a model sublayer's input-output behavior. Given an input $x \in \mathbb{R}^{d_{\text{model}}}$, it comprises a sparse feature vector $z_{\text{TC}}(x) = \text{ReLU}({W_{\text{enc}}}x) + b_{\text{enc}}$ and reconstruct the sublayer's output as $\text{TC}(x) = W_{\text{dec}}z_{\text{TC}}(x) + b_{\text{dec}}$. Compared to sparse autoencoders <d-cite key="cunningham2023"></d-cite> (SAEs), which decompose hidden states but don't explain how features **flow** through a layer, transcoders give us a feature-level view of the actual computation, making them a powerful tool for constructing interpretability circuits.

## Mamba SSM

To understand MambaCoders, we first need to understand the architecture they are designed to interpret. Transformers have dominated sequence modeling for years, but they come with a well-known cost: attention scales quadratically with sequence length, making long-context modeling expensive both in compute and memory. State Space Models (SSMs) emerged as a promising alternative, drawing inspiration from classical control theory. An SSM maps an input sequence $x(t)$ through a latent state $h(t)$ via a linear recurrence:

$$
h_{t} = \bar{A}h_{t-1} + \bar{B}x_{t}, \ y_{t} = Ch_{t}
$$

where $\bar{A}$ and $\bar{B}$ are the discretized versions of continuous-time parameters. Crucially, this recurrence can also be computed as a convolution during training, giving SSMs the best of both worlds: parallelizable training and efficient $\mathcal{O}(1)$-per-step autoregressive inference (with constant memory overhead). Early SSMs like S4 <d-cite key="gu2021"></d-cite> demonstrated strong results on long-range benchmarks, but struggled on discrete, information-dense data like text. The issue lied in their weights: their parameters are fixed across time steps (a property called *linear time invariance*, or LTI) preventing them from performing content-aware reasoning.

Mamba <d-cite key="gu2023"></d-cite> addressed this limitation with a **selection mechanism**: rather than keeping $\bar{A}$, $\bar{B}$, $C$ fixed across time, they are made functions of the current input $x_{t}$, turning the time-invariant recurrence into a time-varying one:

$$
h_{t} = \bar{A}_{t} h_{t-1} + \bar{B}_{t} x_{t} \\ y_{t} = C_{t} h_{t}
$$

where $ \bar{A}\_{t} , \bar{B}\_{t} , C_{t} $ are now all linear functions of the input $f \( x_{t} \)$. This input dependence is what gives the model its name<d-footnote>S6, standing for S4 with a <strong>S</strong>election mechanism computed via a <strong>S</strong>can</d-footnote> and is what allows it to selectively propagate or forget information depending on the content of the current token, rather than treating all inputs uniformly. This breaks LTI, and therefore Mamba cannot be view from a convolutional point of view, but since its state update is linear, outputs can be computed in parallel via a parallel scan <d-cite key="blelloch1989"></d-cite>. This overcomes the efficiency challenges avoiding materializing the large intermediate state in GPU HBM.

## The Mamba Block

The S6 layer sits inside a larger **Mamba block** (Figure 1), which wraps the selective SSM with projections and a gating mechanism. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/MambaCoder/MambaBlock.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig: 1. The Mamba Block</strong>. The mamba block is composed by a time-mixing and a channel-mixing path, combined through an elementwise product. The time-mixing path contains a short convolution 
</div>

Given a residual stream input $x \in \mathbb{R}^{d_{\text{model}}}$, the block proceeds as follows. First, the input is projected into a higher-dimensional space via two parallel linear projections:

$$
u = W_{\text{in}}x, \ g = W_{\text{gate}}x
$$

where both $u, v \in \mathbb{R}^{\text{ED}}$ with $E$ being the expansion factor. The $u$ branch is then passed through a short convolution (which seems to introduce asymmetries <d-cite key="Chen2026"></d-cite> and substituted in Mamba-3 <d-cite key="mamba3"></d-cite>) followed by a SiLU activation before entering the SSM:

$$
u' = \text{SiLU}(\text{Conv1d}(u))
$$

The S6 parameters $B_{t}$, $C_{t}$ and $\Delta_{t}$ are then computed as linear projections of $u'$, implementing the selection mechanism. The SSM is applied to produce:

$$
y = \text{S6}(u')
$$

The output is then gated by the second branch through an elementwise product with a SiLU-activated gate, before a final output projection brings it back to the residual stream dimension:

$$
z = y \odot \text{SiLU}(g) 
\\
\text{out} = W_{\text{in}}z
$$

The full block is stacked homogeneoysly, interleaved with RMSNorm and residual connections, forming the Mamba architecture. Compared to a Transformer layer, which separates attention and MLP into two distinct sublayers, the Mamba block fuses sequence mixing and nonlinear transformation into a single unit, which is both more parameter-efficient and harder to decompose with standard interpretability tools.

The catch, from an interpretability standpoint, is that Mamba's computational structure is fundamentally different from transformers. There are no attention heads to probe, and information flows through a compressed hidden state governed by learned, input projections. That's exactly what we want to accomplish here: reformulate transcoder's theory to apply it to Mamba models.

## How to Apply Transcoders to Mamba?

Applying transcoders to transformers is relatively straightforward: since the MLP sublayer has a clean input-output interface sitting alongside the attention sublayer, one can simply train a transcoder to mimic the MLP's input-output behavior in isolation. The two sublayers are additive and largely independent, which makes the decomposition natural.

Mamba breaks this assumption. As described above, the Mamba block fuses sequence mixing (the SSM) and channel mixing (the gated projections) into a single unit. There is no clean MLP sublayer to target, since the two blocks interact in a nonlinear way.

The solution is to treat the **entire Mamba block** as the unit of transcoding: I trained the MambaCoder to take the block's input from the residual stream and predict its output, bypassing the internal complexity. Formally, if $x_{l}$ is the residual stream at layer $l$ before the Mamba block, the MambaCoder learns:

$$
\text{MC}_{l}(x_{l}) \approx \text{MambaBlock}_{l}(x_{l})
$$

This is the same transcoder objective as in the transformer case, just applied at a coarser granularity.

There is one sublety worth flagging: because the Mamba block contains both temporal mixing (SSM) and channel mixing (SSM), any feature interaction in the last mixing layer is not explicitly captured by the transcoder's feature decomposition, since we won't be able to tell whether that feature comes from the temporal or channel axis. In practice, I found this to be a minor issue, since most of the information between tokens is passed before the last layer, expecially in deeper models.

## Computing Attribution for Mamba Layers

Once we have trained MambaCoders for each layer, we can use them to trace how features propagate through the model. Attribution works differently depending on whether we are connecting two MambaCoder features across layers, or connecting a MambaCoder feature back to a specific Mamba mixer head and token.

### Attribution Between MambaCoder Features

Recall that the MambaCoder at layer $l$ approximates the Mamba block's output as a sparse sum over features:

$$
\text{MC}^{(l)}(x^{(l, t)}) \approx \sum_{j} z_{\text{TC}}^{(l, j)}(x^{(l, t)})f_{dec}^{(l, j)}
$$

where $z_{\text{TC}}^{(l, j)}$ is the activation of feature $j$ at layer $l$ and token $t$, and $f_{dec}^{(l, j)}$ is its decoder direction in the residual stream.

Now suppose we want to measure how much feature $j$ in layer $l$ contributed to the activation of feature $i'$ in a later layer $l' \gt l$. The encoder of the later MambaCoder reads from the residual stream, so the contribution of the decoder direction $f_{dec}^{(l, j)}$ to the activation of feature $i'$ is simply their dot product, weighted by the activation strength:

$$
\text{attr}(l, j) \to (l', i') = z_{\text{TC}}^{(l, j)}(x^{(l, t)}) \cdot (f_{enc}^{(l', i')} \cdot f_{dec}^{(l, j)})
$$

This is the standard transcoder attribution formula, and it works the same way here as in the transformer case.

### Attribution from a MambaCoder Feature to a Mamba Mixer Head

The more interesting case is attributing a MambaCoder feature in layer $l'$ back to the **Mamba mixer** in an earliear layer $l$. This requires us to unroll the SSM recurrence and reason about which token positions contributed to the current hidden state.

Recall that Mamba selective scan computes, for each output token $t$:

$$
y_{t} = C_{t}h_{t} = \sum_{s=1}^{t} C_{t} \prod_{j=s}^{t}\bar{A}^{j} \bar{B}_{s} x^{(l, s)}
$$

This has the form of a weighted sum over all past tokens, and we can define an attention-like matrix <d-cite key="Ali2024"></d-cite> $\alpha$ whose $(t, s)$ entry captures the correlation between output token $t$ and input token $s$.

$$
\alpha_{t, s} = C_{t} \prod_{j=s}^{t}\bar{A}^{j} \bar{B}_{s}
$$

However, this scalar picture only holds for a single SSM dimension. In Mamba, there is one independent SSM per input dimension $d$, so the full model has $D$ such "heads", one per channel. We can treat each input dimension $h \in \{1, \dots, D\}$ as a separate Mamba attention head, in direct analogy with multi-head attention in transformers.

### Deriving the Attribution Formula

We want to compute the contribution of Mamba head $h$ at token $s$ in layer $l$ to the activation of encoder feature $i'$ in layer $l'$. We need to evaluate:

$$
f_{enc}^{(l', i')} \cdot \text{stack}_{h} (\sum_{s=1}^{t} C_{t}^{(h)} \prod_{j=s}^{t}\bar{A}^{j} \bar{B}_{s} x^{(l, s)}[h])
$$

where $ \text{stack}\_{h} $ assembles the scalar SSM outputs across all heads into a single $D$-dimensional vector. Defining $ s(h) = sum_{s=1}^{t} C_{t}^{(h)} \prod_{j=s}^{t}\bar{A}^{j} \bar{B}\_{s} x^{(l, s)}[h]$ and writing the stack as $ \text{stack}\_{h} s(h) = \sum_{h} s(h)e(h) $, where $e(h)$ is the $h$-th standard basis vector, we get:

$$
f_{enc}^{(l', i')} \cdot \text{stack}_{h}(s(h)) = \sum_{h=1}^{D} s(h) \cdot f_{enc}^{(l', i')}[h]
$$

Substituting back and exchanging the order of summation:

$$
= \sum_{s=1}^{t} \sum_{h=1}^{D} C_{t}^{(h)} \prod_{j=s}^{t}\bar{A}^{j} \bar{B}_{s} x^{(l, s)}[h] \cdot f_{enc}^{(l', i')}[h]
$$

Fixing a specific head $h$ and source token $s$, the attribution from Mamba layer $l$, head $h$, token $s$ to feature $i'$ at token $t$ in layer $l'$ is therefore:

$$
\text{attr}((l, h, s) \to (l', i', t)) = C_{t}^{(h)} \prod_{j=s}^{t}\bar{A}^{j} \bar{B}_{s} x^{(l, s)}[h] \cdot f_{enc}^{(l', i')}[h]
$$

With the new feature vector that will be:

$$
f' = C_{t}^{(h)} \prod_{j=s}^{t}\bar{A}^{j} \bar{B}_{s}  \cdot f_{enc}^{(l', i')}
$$

### Handling the Convolution and Gate

The attribution formula above atttributes a MambaCoder feature back to the residual stream input $x^{(l, s)}$ of the Mamba block. However, as described in the Mamba block section, the residual stream input does not flow directly into the S6 layer: it first passes through a short **convolution**, and after the SSM the output is multiplied elementwise by a **SiLU-activated gate**. When tracing features back through the full circuit, we need to account for both of these operations.

Handling them precisely would require unrolling the convolution kernel and the gate into the attribution computation, significantly complicating the circuit tracing. Instead, I treat both the convolution and the gate as **pointwise scaling operations** on the feature vectors. I know that this is a brute approximation, but it holds when the dominant source of token mixing in Mamba is the S6 selective scan, not the short local convolution.

Concretely, for a given layer $l$ and token $s$, we compute the scaling factor of each operation as the ration of activations before and after it:

$$
\gamma_{\text{conv}^{(l, s)}} = \frac{\text{act_after_conv}^{(l,s)}}{\text{act_before_conv}^{(l,s)}}, \quad \gamma_{\text{gate}^{(l, s)}} = \frac{\text{act_after_gate}^{(l,s)}}{\text{act_before_gate}^{(l,s)}}
$$

and then rescale the feature vector accordingly as we trace it back toward the embeddings:

$$
f' = f' \odot \gamma_{\text{gate}}^{(l, s)} \odot \gamma_{\text{conv}}^{(l, s)}
$$

While the conv could give us some attributions in the time domain, the gate is applied channel wise, so it's okay to apply scaling to each dimension.

## An "Hungry" Example

To put everything together, I applied MambaCoders to a concrete case to examine whether they could extract some intresting features. I trained a simple 4-layer pure Mamba model on the TinyStories dataset, then trained MambaCoders on each of its layers. What follows is the story of one particuarly interpretable feature: **feature 3498 in layer 1**.

### Discovering the Feature

The first hint came from a simple experiment. Given the prompt *"I'm so"*, the model confidently predicts emotional words, like *excited, glad, happy*. But given *"I like to eat toasts for breakfast. I'm so*, something changes: the top prediction shifts to **hungry**, and feature 3498 in layer 1 fires strongly on the last token.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/MambaCoder/discovering_the_feature.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig. 2.</strong> For the prompt "I'm so", only a generic layer 1 feature fires (left). Adding food context activates feature 3498 in layer 1, shifting the prediction to "hungry" (right).
</div>

To verify that this feature is causally responsible, and not just correlated, I artificially activated feature 3498 on the *"I'm so"* prompt and observed the effect on the output distribution. The result is intresting:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/MambaCoder/hungry_next_tok.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig. 3.</strong> Artificially activating feature 3498 on the prompt "I'm so" shifts the model's top prediction from "excited" to "hungry".
</div>

### What Does the Feature Represent?

To understand what feature 3498 is actually encoding, I projected its decoder direction together with the token embedding vectors of the tokenizer into a 3D space using UMAP:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/MambaCoder/umap_tokens.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig. 4.</strong> 3D UMAP projection of normalized token embeddings. The feature vector (red arrow) points directly into a cluster of food-related tokens: "fork", "bowl", "eating", "ate", "eat", "food", "hungry", "delicious".
</div>

The feature vector lands squarely in a cluster of food-related tokens: *fork, bowl, eating, ate, eat, food, hungry, delicious*. This is not a coincidence — the feature has learned to represent the concept of food and hunger as a direction in the residual stream.

### Steering the Model with the Feature

To further validate the feature's semantics, I ran a set of generation experiments: I gave *"Once Upon a Time"* as a prompt to my model and I generated completions both with and without feature 3498 artificially activated. The results are telling:

| Without feature | With feature |
| :------------ | :------------ |
|*there was a little girl named Lily. She loved to play outside...*    |*there was a little girl named Lily. She had a yummy dinner with her food. Her...*|
|*there were two friends called Jack and Stop...*|*a mummy was cooking. She was baking a cake inside...*|
|*there was a little girl who liked to skip...*|*there was a little girl who loved to eat her dinner. Every night she would eat dinner and*|
|*a girl named Jane was in her garden with her dad. Jane was feeling serious about the sunshine...*|*a little girl named Anna was in her kitchen. She wanted to help her mom. A...*|

Every time the feature is active, the model steers the story toward food, cooking, or eating, regardless of the original context. Feature 3498 is a **food/hunger** feature.

### Tracing the Circuit

Having identified the feature, I used the circuit tracing algorithm to ask: where does this feature come from in the prompt *"I like to eat toasts for breakfast. I'm so"*?

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/MambaCoder/the_hungry_circuit.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig. 5.</strong> The traced circuit for feature 3498 at the last token. The feature is activated by mixer head 920 in layer 0, which routes information from token 3 ("eat") all the way to the final token position.
</div>

The circuit tracer finds the following path:

`block1@-1 -> mixer0[920]@3: 0.22 -> embed0@3: 1.5`

In plain English: feature 3498 in layer 1 at the last token is primarily driven by **head 920 of the layer 0 mixer**, which in turn is reading from **token 3** — the token *"eat"* — in the input. And the feature vector being passed through that head is precisely the **embedding of the word "eat"** itself. The model has learned to route the semantic content of the word *"eat"* across the sequence via a specific SSM head, accumulate it into a food/hunger feature in layer 1, and use that feature to condition the next-token prediction.

So this demonstrated that MambaCoders can recover human-interpretable features in Mamba-based architectures.

## Conclusions

MambaCoders show that the core goals of mechanistic interpretability (finding human-interpretable features and understanding how they connect) are achievable in Mamba-based architectures. The "hunger" circuit is a small but concrete proof of concept: a single food/hunger feature, causally verified, traced all the way back to the embedding of the word "eat" being routed across the sequence by a specific SSM head.
But the broader motivation is what excites me most. Hybrid models mixing Mamba and attention layers are becoming increasingly common at the frontier, and understanding them mechanistically requires tools that can handle both layer types and trace circuits across them. MambaCoders provide the missing piece for the SSM side of that picture. Scaling this approach to larger hybrid models, and combining it with existing transformer interpretability tools, feels like a genuinely promising direction and I hope this post is a small step toward getting there.