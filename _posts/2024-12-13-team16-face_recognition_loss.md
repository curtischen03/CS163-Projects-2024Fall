---
layout: post
comments: true
title: Loss Functions in CNNs - A Comparison of A-Softmax, CosFace, and ArcFace
author: Curtis Chen
date: 2024-12-13
---

> In this report, we focus on analyzing loss functions used in Convolutional Neural Networks (CNNs), specifically comparing A-Softmax, CosFace, and ArcFace, and examining their performances.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
We will be focusing on loss functions used in CNNs, and analyzing their performances against each other.

---

## A-Softmax
A-Softmax is a loss function based on Softmax. The key difference is that it projects the Euclidean space into an angular space to incorporate an angular margin. The angular margin is preferred because the cosine of the angle aligns better with the concept of softmax.

### Loss function:
\[
\mathcal{L}_{A\text{-Softmax}} = -\frac{1}{N} \sum_{i=1}^{N} \log \left( \frac{\exp(\mathbf{w}_y^T \mathbf{z}_i)}{\sum_{c=1}^{C} \exp(\mathbf{w}_c^T \mathbf{z}_i)} \right)
\]
Where \( N \) is the number of training samples.

From the paper:
“The decision boundary of the A-Softmax is defined over the angular space by \(\cos(m_1) = \cos(2)\), which has difficulty in optimization due to the non-monotonicity of the cosine function. To overcome this, an ad-hoc piecewise function for A-Softmax is often employed.”

---

## CosFace
CosFace, proposed in 2018, introduces the large margin cosine loss (LMCL) as a novel approach to improve discriminative power. The decision boundary is placed in the angular space to ensure consistent margins for all classes.

### Loss function:
\[
\mathcal{L}_{CosFace} = -\frac{1}{N} \sum_{i=1}^{N} \log \left( \frac{\exp(\mathbf{w}_y^T \mathbf{z}_i)}{\sum_{c=1}^{C} \exp(\mathbf{w}_c^T \mathbf{z}_i) + m} \right)
\]
Where \( m \) is the cosine margin.

The CosFace decision boundary is clearly more robust than A-Softmax in maintaining consistent inter-class margins.

---

## ArcFace
ArcFace employs an Additive Angular Margin Loss that further enhances the discriminative ability of feature representations.

### Loss function:
\[
\mathcal{L}_{ArcFace} = -\frac{1}{N} \sum_{i=1}^{N} \log \left( \frac{\exp(\mathbf{w}_y^T \mathbf{z}_i + m)}{\sum_{c=1}^{C} \exp(\mathbf{w}_c^T \mathbf{z}_i)} \right)
\]
Where \( m \) is the additive angular margin.

The ArcFace model tends to have stricter class separation and fewer false positives compared to both A-Softmax and CosFace.

---

## Results & Experiments
### Toy Experiments
Several toy experiments were conducted to visualize the impact of these loss functions on 8 identities. The decision boundaries in Figure 5 illustrate the improved class separation using ArcFace compared to Softmax.

#### Figure 5: A toy experiment comparing Softmax, CosFace, and ArcFace.

---

## Code Implementation
Here is a simple example of how you might implement the ArcFace loss in a neural network:

