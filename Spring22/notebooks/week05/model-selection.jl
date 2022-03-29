### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ aa7cba50-068f-4ffb-8100-3e060fe70868
begin
    using PlutoUI
    #TableOfContents()
end

# ╔═╡ 47d463f4-85db-11ec-2f89-7fdf817bf150
md"""
# Model selection, combination, and regularization

**Note**: this content borrows heavily from a [literature review](https://github.com/vsrikrish/model-selection/blob/master/doc/2020-07-16-presentation-keller-lab.ipynb) that I developed with [Vivek Srikrishnan](viveks.bee.cornell.edu/).

**Reading:** For a more accessible discussion, see Chapter 7 of McElreath's Statistical Rethinking.
For a technical paper, see Piironen & Vehtari (2017) or other references listed below.
"""

# ╔═╡ ecb0b7cc-0077-455e-86c7-77cff0897bc0
md"""
## The challenge

We want to make probabilistic predictions about **unobserved** data $\tilde{y}$.
This is hard because Earth systems are:

1. high-dimensional
1. multi-scale
1. nonlinear / complex

To approximate the true system, we come up with a **model space** $\mathcal{M}$ defining a family of candidate models, then use them to make predictions.
"""

# ╔═╡ 77fab9f0-3314-4c13-93b1-49ee6afdb957
md"""
## Some background theory

Recall:

```math
D_\text{KL} (P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \left[ \frac{P(x)}{Q(x)} \right]
```

One interpretation of $D_\text{KL} (P \parallel Q)$ is the measure of information gained by revising one's beliefes from the prior distribution $Q$ to the posterior distribution $P$.
Another interpretation is the amount of information lost when $Q$ is used to approximate $P$.
Note that for continuous RVs the above sum can be written as an integral.
"""

# ╔═╡ 4526cce5-bbb8-4803-8f0d-9f1a1bef76bc
md"""
### Measures of predictive accuracy

Predictive performance of a model defined in terms of a utility function $u(M, \tilde{y})$.
Commonly used: log predictive density: 
```math
\log p(\tilde{y} | D, M).
```
Future observations $\tilde{y}$ are unknown, so we must approach it in expectation:
```math
\overline{u}(M) = \mathbb{E}\left[ \log p(\tilde{y} | D, M) \right] = \int p_t(\tilde{y}) \log [(\tilde{y} | D, M) d\tilde{y}
```
where $p_t(\tilde{y})$ is the true data generating distribution (unknown!)

This has nice properties: maximizing $\overline{u}(M)$ is equivalent to minimizing KL divergence from candidate model $p(\tilde{y} | D, M)$ to true data distribution $p_t(\tilde{y})$
"""

# ╔═╡ 1ff39b97-aaf9-403b-aa6d-2c1be3a89359
md"""
### In practice we work with posterior estimates

We don't know the true distribution $\theta$ so we have to approximate it.
The log pointwise predictive density is
```math
\begin{align}
\text{lppd} &= \log \prod_{i=1}^N p_\text{post}(y_i) = \sum_{i=1}^N \log \int p(y_i | \theta) p_\text{post} (\theta) d \theta \\
&\approx \sum_{i=1}^N \log \left[ \frac{1}{S} \sum_{i=1}^S p(y_i | \theta^s) \right]
\end{align}
```
where we have approximated the posterior with $S$ simulations from the posterior (eg, using MCMC).

Key point:

> the LPPD of observed data $y$ is an overestimate of the expected LPPD for future data. Thus tools will start with our approximate form and then derive some correction.
"""

# ╔═╡ 49600b32-606e-44ec-bc37-8b468737a740
md"""
## Model combination

We could try to sample from the model space.
If we have an exhaustive list of candidate models $\{ M_\ell \}_{\ell=1}^L$, then the distribution over the *model space* is given by
```math
p(M | D) \propto p(D | M) p(M)
```
and we can average over them
```math
p(\tilde{y} | D) = \sum_{\ell=1}^L p(\tilde{y}|D, M_\ell) p(M_\ell | D)
```
strictly speaking this is an $\mathcal{M}$-closed assumption but in practice this is often not a super critical assumption
"""

# ╔═╡ 76a4f044-12ab-4d01-aa4b-a00f30bb64e2
md"""
### MAP

Alternatively, choose the model with the highest posterior probability (i.e., the "best" model).
For $L=2$ we can see the close analogy to the Bayes Factor:
```math
K = \frac{\Pr(D|M_1)}{\Pr(D|M_2)}
= \frac{\int \Pr(\theta_1|M_1)\Pr(D|\theta_1,M_1)\,d\theta_1}
{\int \Pr(\theta_2|M_2)\Pr(D|\theta_2,M_2)\,d\theta_2}
= \frac{\Pr(M_1|D)}{\Pr(M_2|D)}\frac{\Pr(M_2)}{\Pr(M_1)}.
```
Remember, though, that the candidate model set can be arbitrary -- hence the many problems with significance testing!
"""

# ╔═╡ ce995701-f35b-4a20-ada9-1b395c50c771
md"""
## Generalizing LPPD

### Cross-Validation

* We can use the sample data $D$ as a proxy for $p_t(\tilde{y}$
* Estimating $\mathbb{E}[u(M, \tilde{y})]$  using training data $D$ biases generalization performance (overfitting)
* Divide the data into $K$ subsets; for $i=1, \ldots, K$ hold out the $i$th subset, use it for validation, and use the rest for traning
* Small $K$ also induces bias -- $K=N$ is ideal but expensive so $K=10$ often used
* Approximations to LOO ($K=N$) will be discussed later
"""

# ╔═╡ 49e81579-e7b9-45cb-948b-3e55b6d508ae
md"""
### AIC Criterion

If our inference on the parameters is summarized by a point estimate $\hat{\theta}$ (e.g., the maximum likelihood estimate) then out of sample predictive accuracy is defined by
```math
\text{elpd}_\hat{\theta} = \mathbb{E}_f \left[ \log p(\tilde{y} | \hat{\theta}(y)) \right]
```
If the model estimates $k$ parameters, and if they are assumed asymptotically normal (ie a normal linear model with known variance and uniofrm prior) then fitting $k$ parameters will increase the predictive accuracy by chance alone:
```math
\hat{\text{elpd}}_\text{AIC} = \log p(y | \hat{\theta}_\text{mle}) - k
```
Thus we can define
```math
\text{AIC} = 2 k - 2 \ln \hat{\mathcal{L}}
```
and select the model that minimizes it.

_For complicated models, what is $k$?_
There are formula to approximate effective number of parameters.
Note that AIC asssumes residuals are independent given $\hat{\theta}$
"""

# ╔═╡ 1732c591-d838-4636-b8ac-35f6728080c2
md"""
### DIC Criterion

1. Start with AIC
1. Replace $\hat{\theta}_\text{mle}$ by posterior mean $\hat{\theta}_\text{Bayes} = \mathbb{E}[\theta | y]$
1. Replace $k$ by a data-based bias correction; there are different forms

```math
\hat{\text{elpd}}_\text{DIC} = \log p(y | \hat{\theta}_\text{Bayes}) - p_\text{DIC}
```
where $p_\text{DIC}$ is derived from assumptions about the effective number of parameters.
The quantity
```math
\text{DIC} = -2 \log p(y | \hat{\theta}_\text{Bayes}) + 2 p_\text{DIC}
```
can be assigned to each model, and the model with lowest DIC chosen.
Note that DIC asssumes residuals are independent given $\hat{\theta}$
"""

# ╔═╡ f056b6e2-73c5-4702-8540-90c6ed6e34b3
md"""
### WAIC

More fully Bayesian information criterion and _can be viewed as approximation to cross-validation_.

Define the bias correction penalty:
```math
p_\text{WAIC2} = \sum_{i=1}^N \mathbb{V}_\text{post} \left[ \log p(y_i | \theta) \right]
```
and use
```math
\hat{\text{elppd}}_\text{WAIC} = \text{lppd} - p_\text{WAIC}
```

WAIC is an approximation to the number of 'unconstrained' parameters in the model:

1. a parameter counts as 1 if it is estimated with no constraints or prior information
	* param counts as 0 if it is fully constrained by the prior
    * param gives intermediate value if both the data and prior distributions are informative.

WAIC averages over posterior, which is good.
_BDA3 recommends WAIC over AIC and DIC but it requires partioning data into $n$ pieces_.
"""

# ╔═╡ e2fda6e2-c1fd-4857-9104-862706fb13f7
md"""
### Schwarz criterion / "Bayesian" information criterion (BIC, SBC, SIC, SBIC)

Goal: approximate marginal probability of the data $p(y)$ (this is different)

Assuming the existence of a true model ($\mathcal{M}-closed$), the model that minimizes BIC converges to the "true" model.
```math
\text{BIC} = k \ln (n) - 2 \ln \hat{\mathcal{L}}
```
where
```math
\hat{\mathcal{L}}= \max_\theta p(x | \theta, M)
```
and where $k$ is the number of model parameters.
The BIC can be viewed as a rough approximation to the Bayes factor (Kass and Raftery 1995).
"""

# ╔═╡ 9f3c8339-1281-49b1-a275-b7df3a75722c
md"""
### Significance criteria

Use Null Hypothesis Significance Testing (NHST) to decide whether to include a variable.
For example, should we add a trend term in our regression?

1. Form a null hypothesis: $\beta = 0$
1. Test statistics $\Rightarrow$ $p$-value
1. If $p < \alpha$ then use $M_2$ else use $M_1$

Note that

* This is equivalent to Bayes factor.
* Still assumes existence of a true model (hence the many problems with NHST)

**This is widely used in practice, often without justification**
"""

# ╔═╡ 3908a957-686e-4fa2-8c68-503474516f9e
md"""
## Reference model approach

* Instead of trying to find a MAP model from $\mathcal{M}$, approximate a "reference" model $M_*$
* IE: what is the best emulator, conditional on believing the reference model?
"""

# ╔═╡ 990b5a72-3086-448e-89c6-a36570497bdf
md"""
### Reference predictive method

We can estimate the utilities of the candidate models by replacing $p_t(\tilde{y})$ by $p(\tilde{y} | D, M_*)$:
```math
\overline{u}_\text{ref}(M) = \frac{1}{N} \sum_{i=1}^N \int \underbrace{p(\tilde{y} | x_i, D, M_*)}_{\approx p_t} \log p(\tilde{y} | x_i, D, M) d \tilde{y}
```

Maximizing the reference utility is equivalent to minimizing $D_\text{KL} (M_* \parallel M)$.
"""

# ╔═╡ 519c3929-b1a4-429f-a62e-1510d19ec699
md"""
### Projection predictive method

Goal: project the information in the posterior of the reference model $M_*$ onto the candidate models.

* Parameters of candidate models are determined by the reference model, not by data
* Only reference model needs to be fit / calibrated

Given parameters of the reference model $\theta^*$, the parameters of model $M$ is
```math
\theta^\perp = \arg \min_\theta \frac{1}{N} \sum_{i=1}^N \text{KL} \left[ p(\tilde{y} | x_i, \theta^*, M_*) \parallel p(\tilde{y} | x_i, \theta^\perp, M) \right]
```
which can be approximated by samples and used to set a rule for choosing a model.
"""

# ╔═╡ 82e4c2fc-a13a-4494-ad1d-65908140c1be
md"""
## Conclusions: No Magic Here

* Regularization: fit models that are more robust in the first place
* "In a sparse-data setting, a poor choice of prior distribution can lead to weak inferences and poor predictions." BDA3. Analagous: equifinality.
* "Informative prior distributions and hierarchical structures tend to reduce the amount of overfitting, compared to what would happen under simple least squares or maximum likelihood estimation."

A complement to automatic variable selection / combination is to start with a simple model and expand iteratively in a [principled](https://github.com/betanalpha/jupyter_case_studies/blob/master/principled_bayesian_workflow/principled_bayesian_workflow.ipynb) fashion using posterior predictive checks and domain expertise, expanding the model only where clear deficiencies are identified
"""

# ╔═╡ 0847ac80-c4f1-40ac-81ea-ed294312905f
md"""
### Topics deferred

* Stacking -- a clever way to do model combination in the $\mathcal{M}$-open case
* Regularization (LASSO, Ridge, sparse priors, hierarchical structure) can push some coefficients towards group mean or zero
"""

# ╔═╡ f1c583f2-fe0c-4d08-9a22-1ff3106d3019
md"""
## References / read more

* Gelman, A., & Loken, E. (2013, November 14). The garden of forking paths: Why multiple comparisons can be a problem, even when there is no “fishing expedition” or “p-hacking” and the research hypothesis …. Retrieved from [http://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf](http://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf)
* Heinze, G., Wallisch, C., & Dunkler, D. (2018). Variable selection – A review and recommendations for the practicing statistician. Biometrical Journal, 60(3), 431–449. [https://doi.org/10.1002/bimj.201700067](https://doi.org/10.1002/bimj.201700067)
* Kass, R. E., & Raftery, A. E. (1995). Bayes Factors. Journal of the American Statistical Association, 90(430), 773–795. [https://doi.org/10.1080/01621459.1995.10476572](https://doi.org/10.1080/01621459.1995.10476572)
* Navarro, D. J. (2018). Between the Devil and the Deep Blue Sea: Tensions Between Scientific Judgement and Statistical Model Selection. Computational Brain & Behavior. [https://doi.org/10.1007/s42113-018-0019-z](https://doi.org/10.1007/s42113-018-0019-z)
* Piironen, J., & Vehtari, A. (2017). Comparison of Bayesian predictive methods for model selection. Statistics and Computing, 27(3), 711–735. [https://doi.org/10.1007/s11222-016-9649-y](https://doi.org/10.1007/s11222-016-9649-y)
* Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian Model Evaluation Using Leave-One-out Cross-Validation and WAIC. Statistics and Computing, 27(5), 1413–1432. [https://doi.org/10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4)
* Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using Stacking to Average Bayesian Predictive Distributions. Bayesian Analysis. [https://doi.org/10.1214/17-BA1091](https://doi.org/10.1214/17-BA1091)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.34"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0b5cfbb704034b5b4c1869e36634438a047df065"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8979e9802b4ac3d58c503a20f2824ad67f9074dd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.34"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═aa7cba50-068f-4ffb-8100-3e060fe70868
# ╟─47d463f4-85db-11ec-2f89-7fdf817bf150
# ╟─ecb0b7cc-0077-455e-86c7-77cff0897bc0
# ╟─77fab9f0-3314-4c13-93b1-49ee6afdb957
# ╟─4526cce5-bbb8-4803-8f0d-9f1a1bef76bc
# ╟─1ff39b97-aaf9-403b-aa6d-2c1be3a89359
# ╟─49600b32-606e-44ec-bc37-8b468737a740
# ╟─76a4f044-12ab-4d01-aa4b-a00f30bb64e2
# ╟─ce995701-f35b-4a20-ada9-1b395c50c771
# ╟─49e81579-e7b9-45cb-948b-3e55b6d508ae
# ╟─1732c591-d838-4636-b8ac-35f6728080c2
# ╟─f056b6e2-73c5-4702-8540-90c6ed6e34b3
# ╟─e2fda6e2-c1fd-4857-9104-862706fb13f7
# ╟─9f3c8339-1281-49b1-a275-b7df3a75722c
# ╟─3908a957-686e-4fa2-8c68-503474516f9e
# ╟─990b5a72-3086-448e-89c6-a36570497bdf
# ╟─519c3929-b1a4-429f-a62e-1510d19ec699
# ╟─82e4c2fc-a13a-4494-ad1d-65908140c1be
# ╟─0847ac80-c4f1-40ac-81ea-ed294312905f
# ╟─f1c583f2-fe0c-4d08-9a22-1ff3106d3019
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
