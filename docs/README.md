This site contains example-implementations of common Bayesian methods, including:

{% for page in site.pages %}
{% if page.title %}
{% if page.inNav %}
<li>
    <a href="{{ page.url | prepend: site.baseurl }}">{{ page.title }}</a>
</li>
{% endif %}
{% endif %}
{% endfor %}


<!-- TODO:
- Adaptive MCMC
- Hamiltonian Monte Carlo
-->
