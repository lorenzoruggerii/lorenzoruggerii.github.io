---
layout: distill
title: "Circuit Tracing: An in-depth explanation about creating interpretability circuits"
description: "How do we extract interpretability circuits from transcoders?"
tags: [Mechanistic Interpretability, AI, Transformers]
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

authors:
  - name: Lorenzo Ruggeri

bibliography: 2025-08-26-transcoders.bib

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
  - name: Transcoder's Architecture
  - name: Circuit Tracing
    subsections:
      - name: Computing the Attributions between Transcoder Feature Pairs
      - name: Computing the Attributions through Attention Heads
      - name: Tracing Features Back
      - name: Computing the Attribution Graph
      - name: An example using fairytales
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

## What is a Transcoder?

Transcoders <d-cite key="dunefsky2024"></d-cite> are a new way of peeking inside the black box of transformers. Traditionally, researches have used sparse autoencoders (SAEs) <d-cite key="cunningham2023"></d-cite> to make sense of model activations, since SAEs break down the hidden states (i.e. the residual stream) of large language models into interpretable "features". But there's a catch: while SAEs often uncover meaningful features, they don't tell us much about how those features flow through the MLP and attention blocks of a transformers. That's where transcoders come in. A transcoder is trained to imitate an MLP layer's input-output behavior using a wide, sparsely-activating approximation. 
The transcoder architecture allows us to separate features' attribution into input-dependent and input-invariant components. Crucially, the input-invariance is key - it lets us ask general questions about how features connect and transform, rather than chasing explanations that only apply to specific examples. This gives us a powerful tool for understaning how features interact inside transformers, allowing us to construct interpretability circuits.

## Transcoder's Architecture

At its core, a transcoder is a simple but powerful idea: we replace a transformer's original MLP sublayer with a wider, sparse approximation that is (hopefully) easier to interpret. Concretely, a transcoder is just a one-hidden-layer ReLU MLP that learns to mimic the input-output behavior of the original MLP. Given an input activation vector $x \in \mathbb{R}^{d_{\text{model}}}$ the transcoder computes an intermediate sparse feature vector:

$$
z_{\text{TC}}(x) = \text{ReLU}({W_{\text{enc}}}x) + b_{\text{enc}}
$$

where $W_{\text{enc}} \in \mathbb{R}^{d_\text{features} \times d_{\text{model}}}$ maps the input into a higher-dimensional "feature space" ($d_{\text{features}} \gg d_{\text{model}}$) and $b_{\text{enc}}$ is a bias term. Each component of $z\_{\text{TC}}(x)$ represents the activation of a sparse feature.

The decoder stage then maps these features back into the model's hidden space:

$$
\text{TC}(x) = W_{\text{dec}}z_{\text{TC}}(x) + b_{\text{dec}},
$$

where $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times {d_{\text{features}}}}$ and $b_{\text{dec}}$ reconstruct the MLP's output. Intuitively, the encoder decides _which features should fire_, while the decoder decides _how each feature contributes_ to the final output.

To train transcoders, we minimize a loss that balances **faithfulness** to the original MLP with **sparsity** of activations:

$$
\mathcal{L}_{\text{TC}}(x) = \| \text{MLP}(x) - \text{TC}(x) \|_2^2 + \lambda_1 \| z_{\text{TC}}(x) \|_1
$$

where the first term ensures the transcoder closedly matches the true MLP outputs, and the second term (with hyperparameter $ \lambda_1 $) encourages sparse activations, making the features easier to interpret.

In short, transcoders trade a little extra width for a lot of interpretability: they give us a sparse, feature-level view of how information flows through MLP sublayers, without losing track of the original model's computations.

## Circuit tracing

With transcoders in hand, we can go a step further and actually trace the circuits that transformers use to solve tasks. The idea is simple: instead of asking how individual neurons connect, we ask how features discovered by transcoders in one layer connect to features in later layers. Suppose we found a feature in layer $\text{l'}$ that activates on certain tokens with a clear pattern (for example, semicolons after a year token <d-footnote>like in (Einstein 1946<strong>;</strong> Feynman 1947<strong>;</strong>)</d-footnote> ). Now we want to know which features fire in all layers $\text{l}, \text{l} \lt \text{l'}$. In other words: what makes that feature fire in layer $\text{l'}$? Is it just the semicolon or there is some information coming from previous tokens? To do this, we treat the model as a graph of computations and try to extract the subgraph that’s truly responsible for that given behavior. That requires a way of scoring each connection: how much do the attention heads and the MLP layers contribute to one feature’s activation?

<!-- Put the image of the circuit finding algorithm -->

### Computing the Attributions between Transcoder Feature Pairs

To understand how circuits are build, we start by giving a way of measuring the connection between two transcoder features. With transcoders, this turn out to be extremely simple. Formally, let $z^{(l,i)}\_{\text{TC}}(x^{(l,t)}\_{\text{mid}})$ denote the scalar activation of the $i$-th feature in the layer $l$ transcoder on token $t$, given the MLP input $x^{(l, t)}\_{\text{mid}}$. Then for $l \lt l'$ the contribution of feature $ i $ in transcoder $ l $ to the activation of feature $ i' $ in transcoder $ l' $ is:

$$
z^{(l,i)}_{\text{TC}}\left(x^{(l,t)}_{\text{mid}}\right)
\cdot
\left( f^{(l,i)}_{\text{dec}} \cdot f^{(l',i')}_{\text{enc}} \right)
\tag{1}
$$

Here:

- $z^{(l,i)}\_{\text{TC}}(x^{(l,t)}\_{\text{mid}})$ is the activation of the earlier feature — this depends on the current input.
- $f^{(l,i)}\_{\text{dec}}\cdot f^{(l',i')}\_{\text{enc}}$ is the dot product between the earlier feature’s decoder vector and the later feature’s encoder vector — this is input-invariant once the transcoder is trained.

This clean factorization gives us the best of both worlds:

- An **input-invariant term** that tells us how features are generally connected across the model.
- An **input-dependent term** that tells us how much a feature mattered for the specific input at hand.

Now we'll see how to derive equation 1. We are interested in the activation of feature $ i' $ in transcoder layer $ l' $ that activates on token $ t $. The activation is given by:

$$
z_{\text{TC}}^{(l', i')}(x_{\text{mid}}^{(l', t)}) = \text{ReLU}(f_{\text{enc}}^{(l', i')} \cdot x_{\text{mid}}^{(l', t)} + b_{\text{enc}}^{(l', i')})
$$

where $f_{\text{enc}}^{(l', i')}$ is the $ i-th $ row of $W_{\text{enc}}$ for the layer $ l' $ transcoder and $b_{\text{enc}}^{(l', i')}$ is the learned encoder bias for feature $ i' $ in the layer $ l' $ transcoder. Now, we know that this feature is active (i.e. $z_{\text{TC}}^{(l', i')}(x_{\text{mid}}^{(l', t)}) \gt 0$) and it's reasonable to assume that this firing is not given by the bias term. We can therefore ignore $b_{\text{enc}}^{(l', i')}$. Then, since $z_{\text{TC}}^{(l', i')}(x_{\text{mid}}^{(l', t)}) \gt 0$ we can further ignore the $ \text{ReLU} $, leaving us only with:

$$
z_{\text{TC}}^{(l', i')}(x_{\text{mid}}^{(l', t)}) \approx f_{\text{enc}}^{(l', i')} \cdot x_{\text{mid}}^{(l', t)}
$$

Thanks to the _residual connections_ in transformers $x_{\text{mid}}^{(l', t)}$ can be decomposed as the sum of the outputs of all previous components in the model (MLPs and Attention blocks). For instance, in a two-layer model, if $x_{\text{mid}}^{(2, t)}$ is the input for the second MLP sublayer:

$$
x_{\text{mid}}^{(2, t)} = \sum_{h}\text{attn}^{(2, h)}(x_{\text{pre}}^{(2, t)};x_{\text{pre}}^{(2, 1:t)}) + \text{MLP}(x_{\text{mid}}^{(1, t)}) + \sum_{h}\text{attn}^{(1, h)}(x_{\text{pre}}^{(1, t)};x_{\text{pre}}^{(1, 1:t)})
$$

Now, assume that we want to consider how much a feature from the first MLP layer contributes to a feature in the second layer. Because of linearity, the amount of $\text{MLP}^{(1)}(x_{\text{mid}}^{(1, t)})$ that contributes to $f_{\text{enc}}^{(2, i')} \cdot x_{\text{mid}}^{(2, t)}$ is given by:

$$
f_{\text{enc}}^{(2, i')} \cdot \text{MLP}^{(1)}(x_{\text{mid}}^{(1, t)})
\tag{2}
$$

This is generalizable for understanding the contribution of $ \text{MLP} \ l $ to the activation of feature $ i' $ in transcoder $ l' $, with $ l \lt l' $. Now, we can start from here to find the attribution of feature $ i $ in transcoder $ l $ with respect to feature $ i' $ in transcoder $ l' $.

In particular, we know that transcoders are trained to reconstruct MLP outputs. So, if layer $ l $ transcoder is a good approximation to the layer $ l $ MLP, then $\text{MLP}^{(1)}(x_{\text{mid}}^{(l, t)}) \approx \text{TC}^{(l)}(x_{\text{mid}}^{(l, t)})$. Hence, we can use this approximation in Equation 2 to obtain:

$$
f_{\text{enc}}^{(l', i')} \cdot \text{MLP}^{(l)}(x_{\text{mid}}^{(l, t)}) \approx f_{\text{enc}}^{(l', i')} \cdot \text{TC}^{(l)}(x_{\text{mid}}^{(l, t)})
\tag{3}
$$

Equation 3 tells us the **whole** contribution of $\text{TC}^{(l)}$ on feature $ i' $ in transcoder $ l' $. To extract individual feature attributions, we can leverage transcoder's linearity $\text{TC}^{(l)}(x_{\text{mid}}^{(l, t)}) = \sum_{\text{feature} \ j} z_{\text{TC}}^{(l, j)}(x_{\text{mid}}^{(l, t)})f_{\text{dec}}^{(l, j)}$. This is how we can decompose the transcoder attribution into its features contributions to feature $ i' $ in transcoder $ l' $.

Thus, we have:

$$
f_{\text{enc}}^{(l', i')} \cdot \text{MLP}^{(l)}(x_{\text{mid}}^{(l, t)}) \approx f_{\text{enc}}^{(l', i')} \cdot \sum_{\text{feature} \ j} z_{\text{TC}}^{(l, j)}(x_{\text{mid}}^{(l, t)})f_{\text{dec}}^{(l, j)}
\tag{4}
$$

Therefore, the attribution of feature $ i $ in layer $ l $ with respect to feature $ i' $ in transcoder $ l' $ is given by:

$$
z_{\text{TC}}^{(l, j)}(x_{\text{mid}}^{(l, t)})(f_{\text{enc}}^{(l', i')} \cdot f_{\text{dec}}^{(l, j)})
\tag{5}
$$

where we have just put $f_{\text{enc}}^{(l', i')}$ inside the parenthesis since $z_{\text{TC}}^{(l, j)}(x_{\text{mid}}^{(l, t)})$ is a scalar.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/transcoders/MLP_attrib.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig: 1. Attribution between Transcoder Feature Pairs</strong>. Here we are calculating the attribution of feature j in layer l with respect to feature i' in layer l'. The attribution is given by Equation 5.
</div>

### Computing the Attributions through Attention Heads

So far, we have focused on tracing how a lower-layer transcoder feature directly contributes to a higher-layer transcoder feature at the same token position. However, transcoder features can also be mediated by **attention heads**, which requires us to extend the analysis. Attention can be decomposed into two circuits, a QK (query-key) circuit, which decides _where_ to move information from and to, and the OV (output-value) circuit, which decides _what_ information to move <d-footnote>you can read more <a href=https://transformer-circuits.pub/2021/framework/index.html>here</a></d-footnote>.
Specifically, we now want to compute the attribution of transcoder features through the **OV (output-value) circuit** of an attention head. To do this, we treat the QK circuit scores as fixed, and only look at the contributions through the OV circuit.
As before, our goal is to explain what causes feature $i'$ in the transcoder at layer $l'$ to activate on token $t$. Given an attention head $h$ at layer $l$ (with $l \lt l'$), the same reasoning as in the direct case shows that the head's contribution to feature $i'$ is given by:

$$
f^{(l',i')}_{\text{enc}} \cdot \text{attn}^{(l,h)}\big(x^{(l,t)}_{\text{pre}}; x^{(l,1:t)}_{\text{pre}}\big) \tag{6}
$$

This can be decomposed into a sum over source tokens $s$:

$$
f^{(l',i')}_{\text{enc}} \cdot \sum_{\text{source token}\,s}\text{score}^{(l,h)}\big(x^{(l,t)}_{\text{pre}}, x^{(l,s)}_{\text{pre}}\big) \, W^{(l,h)}_{\text{OV}}\,x^{(l,s)}_{\text{pre}} \tag{7}
$$

$$
=\sum_{\text{source token}\,s}\text{score}^{(l,h)}\big(x^{(l,t)}_{\text{pre}}, x^{(l,s)}_{\text{pre}}\big) \, \big(f^{(l',i')}_{\text{enc}}\big)^{T}W^{(l,h)}_{\text{OV}}\,x^{(l,s)}_{\text{pre}} \tag{8}
$$

$$
=\sum_{\text{source token}\,s}\text{score}^{(l,h)}\big(x^{(l,t)}_{\text{pre}}, x^{(l,s)}_{\text{pre}}\big) \, \big(\big(W^{(l,h)}_{\text{OV}}\big)^{T}f^{(l',i')}_{\text{enc}}\big) \cdot x^{(l,s)}_{\text{pre}} \tag{9}
$$

where from (7) to (8) we put $f^{(l',i')}\_{\text{enc}}$ inside the parenthesis since $\text{score}^{(l,h)}\big(x^{(l,t)}\_{\text{pre}}, x^{(l,s)}\_{\text{pre}}\big)$ are scalars for all $s$ and from (8) to (9) we used $A\cdot B = (B^{T}\cdot A^{T})^{T}$.

Thus, the contribution of source token $s$ at layer $l$ through head $h$ can be written as:

$$
\text{score}^{(l,h)}\big(x^{(l,t)}_{\text{pre}}, x^{(l,s)}_{\text{pre}}\big) \, \big(\big(W^{(l,h)}_{\text{OV}}\big)^\top f^{(l',i')}_{\text{enc}}\big) \cdot x^{(l,s)}_{\text{pre}} \tag{10}
$$

So now, for a given feature $i'$ that fires in layer $l'$ on token $t$, we can now trace back **all its influences**. We can calculate the contribution from every previous-layer feature (Equation 5) and from any source token $s$, with $s \lt t$, processed from attention head $h$ (Equation 10).

The final step is to note that $x^{(l,s)}_{\text{pre}}$ itself can be decomposed again into the output of MLP sublayers and attention heads, allowing us to recurse back until the token embeddings.
To visualize this better, assume that you know that on layer 8 feature 2020 is firing on token 5. You apply Equation (5) and you find that most of the contribution is coming from feature 1546 in layer 4, activating on token 5. Your next question is: why is feature 1546 activating? So, given equations 5 and 10, you can recurse back, constructing a computational graph, until you get to the first layer of your model, in which you can see the tokens that make your feature fire. In the next subsection, I'll explain how to get the attribution graph.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/transcoders/attn_attrib.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig: 2. Attribution through attention heads</strong>. Here we are calculating the attribution of token s through attention head h in layer l with respect to feature i' in layer l'. The attribution is given by Equation 10.
</div>

### Tracing Features Back

So far, we've seen how to calculate the attribution from an earlier-layer transcoder feature or attention head to a later-layer feature. But that's only half of the story. To truly understand the model's behavior, we need to flip the perspective: **what contributes to these earlier-layer features or heads in the first place?**

This bring us to the idea of **recursive attribution tracing**, that can be expressed as follows:
>Starting from the final feature vector we care about (e.g. $f\_{enc}^{(l', i')}$ for $i$-th feature in layer $l'$) we move backwards step by step. At each node - earlier transcoder feature or attention head - we compute how much that node contributes to the current feature vector using Equations 5 and 10. We than update our feature vector accordingly and continue tracing backwards until we reach the inputs.

Now, why do we need to update the feature vector step? Recall that we are calculating attributions for a particular feature in a given layer computing the inner product beween its **feature vector** and its inputs $f\_{\text{enc}}^{(l', i')} \cdot x\_{\text{mid}}^{(l, t)}$. So the feature vector represents the lens thruogh with we measure attribution, and that lens changes as activations are transformed along the computational path <d-footnote>for example, traversing a linear layer</d-footnote>. If we held the same vector fixed everywhere, we'd be ignoring how each node remaps information. Consider a simple example: suppose a node $c$ takes an input $x$ and produces an output $y=Ax$ where $A$ is a weight matrix. If our current feature is $f'$, then the attribution of $c$ is $a' = f' \cdot y$ and assume that this attribution is significant. We therefore want to trace back again to understand what causes $c$ to activate. However, here it's not enough to reuse $f'$ directly, because $f'$ "lives" in the output space of $y$ not in the input space of $x$. Instead, we update the feature vector so that the original $a'$ attribution gets untouched:

$$
a' = f' \cdot y = f' \cdot (Ax) = (f' \cdot A)x = f \cdot x
\tag{11}
$$

So our new feature feature vector will be $f=f'\cdot A$. This way, $f$ is aligned with the input space and the attribution remains valid one step earlier in the graph. The same principles applies for MLP and Attention nodes that we have in the graph: the feature vector **must be updated to remain consistent** with the representation space of the preceding layer.

So, in order to compute the new feature vector $f$ through node $c$ starting from $f'$ we'll use the following contraint given from Equation 11:

$$
a' = f' \cdot y = f \cdot x
$$

When node $c$ is an attention head in layer $l$, we have $y = x\_{\text{mid}^{(l, t)}}$, so:

$$
a' = f' \cdot y = f' \cdot x\_{\text{mid}^{(l, t)}} = f' \cdot \sum_{h}\sum_{\text{source token}\,s}\text{score}^{(l,h)}\big(x^{(l,t)}_{\text{pre}}, x^{(l,s)}_{\text{pre}}\big) \, W^{(l,h)}_{\text{OV}}\,x^{(l,s)}_{\text{pre}}\
$$

However, assume that we are tracing the attribution through head $h$ of a source token at position $s$. Therefore, we have $x = x\_{\text{pre}}^{(l, s)}$ and the updated feature vector will be:

$$
a' = f' \cdot \text{score}^{(l,h)}\big(x^{(l,t)}_{\text{pre}}, x^{(l,s)}_{\text{pre}}\big) \, W^{(l,h)}_{\text{OV}}\,x^{(l,s)}_{\text{pre}}\
$$

$$
= f' \cdot \text{score}^{(l,h)}\big(x^{(l,t)}_{\text{pre}}, x^{(l,s)}_{\text{pre}}\big) \, W^{(l,h)}_{\text{OV}}\,x\
$$

$$
= \big(f' \cdot \text{score}^{(l,h)}\big(x^{(l,t)}_{\text{pre}}, x^{(l,s)}_{\text{pre}}\big) \, W^{(l,h)}_{\text{OV}}\big)\,x\
$$
$$
= f\cdot\,x\
$$

Hence:

$$
f = f' \cdot \text{score}^{(l,h)}\big(x^{(l,t)}_{\text{pre}}, x^{(l,s)}_{\text{pre}}\big) \, W^{(l,h)}_{\text{OV}}
\tag{12}
$$

In contrast, if $c$ is a transcoder feature then $y = x\_{\text{post}}^{(l, t)}$, so we have:

$$
a' = f' \cdot x\_{\text{post}}^{(l, t)} = f' \cdot \text{MLP}^{(l)}(x\_{\text{mid}}^{(l, t)}) \approx f' \cdot \text{TC}^{(l)}(x\_{\text{mid}}^{(l, t)}) = f' \cdot \sum\_{j}z\_{\text{TC}}^{(l, j)}(x\_{\text{mid}}^{(l, t)}) \cdot f\_{\text{dec}}^{(l, j)}
$$

Moreover, from the definition of transcoder, we have that $z\_{\text{TC}}^{(l, t)}(x\_{\text{mid}}^{(l, t)}) = f\_{\text{enc}}^{(l, j)} \cdot (x\_{\text{mid}}^{(l, t)})$. Therefore, we obtain:

$$
a' = f' \cdot x\_{\text{post}}^{(l, t)} = f' \cdot \sum\_{j} f\_{\text{enc}}^{(l, j)} \cdot (x\_{\text{mid}}^{(l, t)}) \cdot f\_{\text{dec}}^{(l, j)}
$$

From this, we can just insert inside the sum $f'$ thanks to the linearity of the inner product:

$$
a' = f' \cdot x\_{\text{post}}^{(l, t)} = f' \cdot \sum\_{j} f\_{\text{enc}}^{(l, j)} \cdot (x\_{\text{mid}}^{(l, t)}) \cdot f\_{\text{dec}}^{(l, j)}
$$

$$
=\sum\_{j} \big(f\_{\text{enc}}^{(l, j)} \cdot (x\_{\text{mid}}^{(l, t)})\big) \cdot \big( f' \cdot f\_{\text{dec}}^{(l, j)}\big)
$$

And from this, knowing that our $x = (x\_{\text{mid}}^{(l, t)})$ and assuming we are interested in the contribution of feature $j$ from layer $l$ it's easy to see that:

$$
f = \big( f' \cdot f\_{\text{dec}}^{(l, j)}\big) \cdot f\_{\text{enc}}^{(l, j)}
\tag{13}
$$

There is, however, an important caveat. Transformer architectures insert a LayerNorm operation before every MLP and attention sublayer. This nonlinearity complicates attribution because it rescales and recenters activations. Fortunately, prior work provides intuition and justification for approximating LayerNorm as a linear scaling transformation. Neel Nanda [here](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) argues that it behaves roughly like multiplying the input by a constant, and Dunefsky & Cohan <d-cite key="dunefsky2024"></d-cite> provide both theoretical motivation and empirical evidence supporting this view. In practice, this means that when we update the feature vector $f$ at each sublayer, we multiply it by an empirically estimated _LayerNorm scaling constant_ defined as the ratio between the norm of the pre-LayerNorm activation vector and the post-LayerNorm activation vector.


### Computing the Attribution Graph

### An example using fairytales

## Conclusions
