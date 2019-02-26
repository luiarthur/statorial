---
layout: default
title: "Variational Inference"
inNav: on
---

### Variational Inference with PyTorch
{% for notes in site.varinf %}
  {% if notes.lang == "python" %}
  - [{{ notes.date | date: "%-d %b, %Y"}}&raquo; {{notes.title}}]({{notes.url}})
  {% endif %}
{% endfor %}

