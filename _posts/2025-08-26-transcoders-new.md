---
layout: distill
title: Circuit Tracing:\ An in-depth explanation about creating interpretability circuits
description: How do we extract interpretability circuits from transcoders?
tags: Mechanistic Interpretability, AI, Transformers
giscus_comments: true
date: 2025-08-26
featured: true
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
      - name: Computing the Attributions
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
At its core, a transcoder is a simple but powerful idea: we replace a transformer's original MLP sublayer with a wider, sparse approximation that is (hopefully) easier to interpret. Concretely, a transcoder is just a one-hidden-layer ReLU MLP that learns to mimic the input-output behavior of the original MLP. Given an input activation vector $ x \in \mathbb{R}^{d_{\text{model}}} $ the transcoder computes an intermediate sparse feature vector:

$$
z_{\text{TC}}(x) = \text{ReLU}{W_{\text{enc}}}x + b_{\text{enc}}
$$

where $ W_{\text{enc}} \in \mathbb{R}^{d_\text{features} \times d_{\text{model}}} $ maps the input into a higher-dimensional "feature space" ($ d_{\text{features} \gg d_{\text{model}}} $) and $ b_{\text{enc}} $ is a bias term. Each component of $ z_{\text{TC}}(x) $ represents the activation of a sparse feature.

The decoder stage then maps these features back into the model's hidden space:

$$
\text{TC}(x) = W_{\text{dec}}z_{\text{TC}}(x) + b_{\text{dec}},  
$$

where $ W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times {d_{\text{features}}}} $ and $ b_{\text{dec}} $ reconstruct the MLP's output. Intuitively, the encoder decides *which features should fire*, while the decoder decides *how each feature contributes* to the final output.

To train transcoders, we minimize a loss that balances **faithfulnes** to the original MLP with **sparsity** of activations:

$$
\mathcal{L}_{\text{TC}}(x) = \| \text{MLP}(x) - \text{TC}(x) \|_2^2 + \lambda_1 \| z_{\text{TC}}(x) \|_1
$$

where the first term ensures the transcoder closedly matches the true MLP outputs, and the second term (with hyperparameter $ \lambda_1 $) encourages sparse activations, making the features easier to interpret.

In short, transcoders trade a little extra width for a lot of interpretability: they give us a sparse, feature-level view of how information flows through MLP sublayers, without losing track of the original model's computations.

## Circuit tracing

### Computing the Attributions

### Tracing Features Back

### Computing the Attribution Graph

### An example using fairytales

## Conclusions