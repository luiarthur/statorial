---
layout: page
---

# Statorial

Hi! This site contains example-implementations of common Bayesian methods,
including:

{% for page in site.pages %}
{% if page.title %}
{% if page.inNav %}
- [{{page.title}}]({{ page.url | prepend: site.baseurl }})
{% endif %}
{% endif %}
{% endfor %}

Eventually, I would like to include:
- Adaptive MCMC
- Hamiltonian Monte Carlo

