---
layout: post
title: Your Favourite Genomic Model knows more than you think
date: 2025-12-18 11:12:00-0400
description: How to generate DNA sequences starting from a BERT-like model? 
tags: formatting math
categories: sample-posts
related_posts: true
---

## Introduction

## Masked Language Modeling (MLM)

Masked Language Modeling (MLM) is essentially a sophisticated, high-speed game of "fill-in-the-blanks" designed to teach AI models deep contextual understanding. Unlike traditional models that read text sequentially from left to right—predicting the next word based only on what came before—BERT uses MLM to become truly **bidirectional**. It takes a complete sentence and hides 15% of the words, but it does so using a clever **80/10/10 strategy** to prevent the model from becoming lazy.

Specifically, for that 15% of chosen words:
* **80%** are replaced with a `[MASK]` token (forcing the model to rely on context).
* **10%** are replaced with a **random word** (forcing the model to act as a "spell checker" that detects logic errors).
* **10%** are left **unchanged** (ensuring the model still values the actual word when it is present).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/BERT/BERT_model.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Fig: 1. How BERT model behaves. Random tokens are masked from the input sequence, and the model is asked to identify them using the surrounding context.
</div>

By forcing the model to analyze these surrounding "clues" from both directions, MLM enables BERT to develop a much richer grasp of language nuances, syntax, and semantics. This is why BERT can easily distinguish between "bank" (a river) and "bank" (a financial institution) based entirely on the words sitting next to it.

## Discrete Diffusion

## BERT is a one-step diffusion model

## DNABERT generates enhancers










This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine. You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`. If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph. Here is an example:

$$
\sum_{k=1}^\infty |\langle x, e_k \rangle|^2 \leq \|x\|^2
$$

You can also use `\begin{equation}...\end{equation}` instead of `$$` for display mode math.
MathJax will automatically number equations:

\begin{equation}
\label{eq:cauchy-schwarz}
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
\end{equation}

and by adding `\label{...}` inside the equation environment, we can now refer to the equation using `\eqref`.

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) that brought a significant improvement to the loading and rendering speed, which is now [on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php).
