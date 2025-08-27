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
      - name: Compute the Attribution Graph
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

## Transcoders

Transcoders <d-cite key="dunefsky2024"></d-cite> are a new way of peeking inside the black box of transformers. Traditionally, researches have used sparse autoencoders (SAEs) <d-cite key="cunningham2023"></d-cite> to make sense of model activations, since SAEs break down the hidden states (i.e. the residual stream) of large language models into interpretable "features". But there's a catch: while SAEs often uncover meaningful features, they don't tell us much about how those features flow through the MLP and attention blocks of a transformers. That's where transcoders come in. A transcoder is trained to imitate an MLP layer's input-output behavior using a wide, sparsely-activating approximation. 
The transcoder architecture allows us to separate features' attribution into input-dependent and input-invariant components. Crucially, the input-invariance is key - it lets us ask general questions about how features connect and transform, rather than chasing explanations that only apply to specific examples. This gives us a powerful tool for understaning how features interact inside transformers, allowing us to construct interpretability circuits.

## Transcoder's Architecture

At its core, a transcoder is a simple but powerful idea: we replace a transformer's original MLP sublayer with a wider, sparse approximation that is (hopefully) easier to interpret. Concretely, a transcoder is just a one-hidden-layer ReLU MLP that learns to mimic the input-output behavior of the original MLP. Given an input activation vector $ x \in \mathbb{R}^{d\_{\text{model}}} $ the transcoder computes an intermediate sparse feature vector:

$$
z_{\text{TC}}(x) = \text{ReLU}({W_{\text{enc}}}x) + b_{\text{enc}}
$$

where $ W*{\text{enc}} \in \mathbb{R}^{d*\text{features} \times d*{\text{model}}} $ maps the input into a higher-dimensional "feature space" ($ d*{\text{features}} \gg d*{\text{model}} $) and $ b*{\text{enc}} $ is a bias term. Each component of $ z\_{\text{TC}}(x) $ represents the activation of a sparse feature.

The decoder stage then maps these features back into the model's hidden space:

$$
\text{TC}(x) = W_{\text{dec}}z_{\text{TC}}(x) + b_{\text{dec}},
$$

where $ W*{\text{dec}} \in \mathbb{R}^{d*{\text{model}} \times {d*{\text{features}}}} $ and $ b*{\text{dec}} $ reconstruct the MLP's output. Intuitively, the encoder decides _which features should fire_, while the decoder decides _how each feature contributes_ to the final output.

To train transcoders, we minimize a loss that balances **faithfulness** to the original MLP with **sparsity** of activations:

$$
\mathcal{L}_{\text{TC}}(x) = \| \text{MLP}(x) - \text{TC}(x) \|_2^2 + \lambda_1 \| z_{\text{TC}}(x) \|_1
$$

where the first term ensures the transcoder closedly matches the true MLP outputs, and the second term (with hyperparameter $ \lambda_1 $) encourages sparse activations, making the features easier to interpret.

In short, transcoders trade a little extra width for a lot of interpretability: they give us a sparse, feature-level view of how information flows through MLP sublayers, without losing track of the original model's computations.

## Circuit tracing

With transcoders in hand, we can go a step further and actually trace the circuits that transformers use to solve tasks. The idea is simple: instead of asking how individual neurons connect, we ask how features discovered by transcoders in one layer connect to features in later layers. Suppose we found a feature in layer $ \text{l'} $ that activates on certain tokens with a clear pattern (for example, semicolons after a year token <d-footnote>like in (Einstein 1946<strong>;</strong> Feynman 1947<strong>;</strong>)</d-footnote> ). Now we want to know which features fire in all layers $ \text{l}, \text{l} \lt \text{l'} $. In other words: what makes that feature fire in layer $ \text{l'} $? Is it just the semicolon or there is some information coming from previous tokens? To do this, we treat the model as a graph of computations and try to extract the subgraph that’s truly responsible for that given behavior. That requires a way of scoring each connection: how much do the attention heads and the MLP layers contribute to one feature’s activation?

<!-- Put the image of the circuit finding algorithm -->

### Computing the Attributions between Transcoder Feature Pairs

To understand how circuits are build, we start by giving a way of measuring the connection between two transcoder features. With transcoders, this turn out to be extremely simple. Formally, let $z^{(l,i)}_{\text{TC}} (x^{(l,t)}_{\text{mid}})$ denote the scalar activation of the $ \text{i-th} $ feature in the layer $ l $ transcoder on token $ t $, given the MLP input $ x^{(l, t)}\_{\text{mid}} $. Then for $ l \lt l' $ the contribution of feature $ i $ in transcoder $ l $ to the activation of feature $ i' $ in transcoder $ l' $ is:

$$
z^{(l,i)}_{\text{TC}}\!\left(x^{(l,t)}_{\text{mid}}\right)
\cdot
\left( f^{(l,i)}_{\text{dec}} \cdot f^{(l',i')}_{\text{enc}} \right)
\tag{1}
$$

Here:

- $z^{(l,i)}_{\text{TC}} (x^{(l,t)}_{\text{mid}})$ is the activation of the earlier feature — this depends on the current input.
- $f^{(l,i)}_{\text{dec}} \cdot f^{(l',i')}_{\text{enc}}$ is the dot product between the earlier feature’s decoder vector and the later feature’s encoder vector — this is input-invariant once the transcoder is trained.

This clean factorization gives us the best of both worlds:

- An **input-invariant term** that tells us how features are generally connected across the model.
- An **input-dependent term** that tells us how much a feature mattered for the specific input at hand.

Now we'll see how to derive equation 1. We are interested in the activation of feature $ i' $ in transcoder layer $ l' $ that activates on token $ t $. The activation is given by:

$$
z_{\text{TC}}^{(l', i')}(x_{\text{mid}}^{(l', t)}) = \text{ReLU}(f_{\text{enc}}^{(l', i')} \cdot x_{\text{mid}}^{(l', t)} + b_{\text{enc}}^{(l', i')})
$$

where $f_{\text{enc}}^{(l', i')}$ is the $ i-th $ row of $W_{\text{enc}}$ for the layer $ l' $ transcoder and $b_{\text{enc}}^{(l', i')}$ is the learned encoder bias for feature $ i' $ in the layer $ l' $ transcoder. Now, we know that this feature is active (i.e. $z_{\text{TC}}^{(l', i')}(x_{\text{mid}}^{(l', t)}) \gt 0$) and it's reasonable to assume that this firing is not given by the bias term. We can therefore ignore $b_{\text{enc}}^{(l', i')}$. Then, since $z_{\text{TC}}^{(l', i')}(x_{\text{mid}}^{(l', t)}) \gt 0$ we can further ignore the $ \text{ReLU} $, leaving us only with:

$$
z_{\text{TC}}^{(l', i')}(x_{\text{mid}}^{l', t}) \approx f_{\text{enc}}^{(l', i')} \cdot x_{\text{mid}}^{(l', t)}
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

### Tracing Features Back

### Computing the Attribution Graph

### An example using fairytales

## Conclusions
