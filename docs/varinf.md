---
layout: page
title: "Variational Inference"
description: "with PyTorch"
inNav: on
---

<!-- sorting: https://gist.github.com/Phlow/1f27dfafdf2bbcc5c48e -->

# Variational Inference with PyTorch

{% for notes in site.varinf %}
  {% if notes.lang == "python" %}
  - [{{ notes.date | date: "%-d %b, %Y"}}&raquo; {{notes.title}}]({{notes.url | prepend: site.baseurl}})
  {% endif %}
{% endfor %}

***

The flavor of variational inference I will discuss in these posts are Automatic
Differentiation Variational Inference ([ADVI][1]).


[1]: https://arxiv.org/pdf/1603.00788.pdf
