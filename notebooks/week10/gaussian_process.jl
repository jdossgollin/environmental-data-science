### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ c2135d1a-3b5b-43bb-a433-4148ba69bec9
begin
	using Distances # we will use to compute pairwise Euclidean distance
	using Distributions
	using GaussianProcesses # this package provides useful syntax
	using LaTeXStrings
	using LinearAlgebra 
	using Optim
	using Plots
	using Random
end

# ╔═╡ c76a7aab-b613-40a8-8aa0-9a46cc0c560a
using Interpolations

# ╔═╡ e4ccd0c6-5404-4f8f-a6cf-5ad5f67fe47b
using PlutoUI

# ╔═╡ 4e0a32e2-a941-11ec-27ed-958ec70306f3
md"""
# 1D Gaussian Process for Nonlinear Regression

Thus far the regression approaches we've seen (least squares, Bayesian linear regression, sparse regression, generalized linear models, etc) all assume that there is a linear relationship between our $X$ and $y$.
However, it is common in environmental data science to deal with data where the relationship between our $X$ and $y$ is nonlinear!
In this notebook we'll explore some possible approaches, then learn about 1D Gaussian Processes.

Gaussian Processes form the underlying theory for much of spatial statistics (ie, 2D) but it's helpful to first wrap our heads around the 1D case.

I'd like to thank my colleague [Ben Seiyon Lee](https://statistics.gmu.edu/node/346) from George Mason for sharing some ideas and codes that informed this notebook.

## Reading List

Lots of great materials have already been written about Gaussian Processes and there's no point being redundant.
In this notebook we will breeze through theoretical ideas covered in other resources in order to focus on implementation.
If you don't understand where results are coming from, however, you'll be confused.
Please read the following articles:

1. Start with this article about [Kriging temperature](https://towardsdatascience.com/kriging-the-french-temperatures-f0389ca908dd). You don't need to read everything in detail; it provides a helpful framing.
1. Next read ["Gaussian Processes, Not Quite for Dummies"](https://yugeten.github.io/posts/2019/09/GP/). This explains the theory (I think) quite clearly, so please read through it in some detail.
1. Finally, have a look through [Gaussian Processes from Scratch](https://peterroelants.github.io/posts/gaussian-process-tutorial/). You don't need to worry about the python implementation, but you may enjoy seeing someone else's implementation.

For a more complete reference, see [this textbook](www.GaussianProcess.org/gpml) by Rasmussen and Williams.
It's a classic!


## Motivation

When analyzing environmental data, we are frequently interested in using nearby observations to make predictions about points we haven't yet observed.
For example:

1. Given air temperatures at some places, can we predict the air temperature somewhere else?
1. We drill some holes and measure minerals. Where should we mine to maximize our chances of finding the minerals we are looking for?
1. We measure some process at a few time steps. What happened in between?

Let's consider an example.
Imagine that we have some true process $f(t)$ (let's pretend it's unknown), and that we observe it, with noise, at a few points $\mathbf{t}$.
"""

# ╔═╡ 2d62c571-53f3-4866-8901-0a8fb6f142b5
f(t) = @. sin(2 * π * t / 7.2 + 0.9) + sin(2 * π * t / 2.5 + 1.3);

# ╔═╡ 1032164a-894b-45c0-a6f6-96c280fddf00
begin
	N = 25 # number of observations
	σy = 0.25; # noise parameter
	x0 = 0 # lower bound
	x1 = 10 # upper bound
	Random.seed!(70005) # set seed
	x = sort(rand(Uniform(x0, x1), N)) # points we observe at
	xprime = collect(range(x0, x1, length=1_000)); # locations to estimate
	y = f(x) .+ rand(Normal(0, σy), N) # values we observe
	baseplot = plot(xlabel=L"$x$", ylabel=L"$y(t)", legend=:bottomright) # for plots
end;

# ╔═╡ 595e7e4f-60a5-4666-903b-c292a57e82af
let
	p = deepcopy(baseplot)
	plot!(p, f, x0, x1, label="True Values")
	scatter!(p, x, y, label="Observed Points")
end

# ╔═╡ 1f636e3c-172e-4aa3-a50b-370aacf0a24e
md"""
Suppose we didn't have the blue line and only had measured the orange dots.
How could we make predictions at a new location?
There are lots of options available to us!

## Not Gaussian Processes

Before we dive into Gaussian Processes, let's take a quick look at some other approaches.
We will not be comprehensive!
"""

# ╔═╡ 17b605dc-fd6a-4bb5-90f7-d827cc1e407c
md"""
### Interpolation

A simple thing we could try is a linear interpolation.
This is essentially connect the dots.
It works fine if we have a very high density of unbiased estimates (pretend, for example, say that $f(x)$ was computationally expensive to run and we needed to run it a lot of times -- we might compute it once for $x = 0, 0.001, 0.002, \ldots, 10$, train an interpolation, and then use the computationally cheap interpolated model).
If we're trying to learn about $f$ from a small number of points sampled with noise, it struggles!
"""

# ╔═╡ 77164223-fc16-4af9-ba61-2832404d2ee2
let
	itp_linear = LinearInterpolation(x, y; extrapolation_bc = Line())
	f_linear(x) = itp_linear(x)
	p = deepcopy(baseplot)
	scatter!(p, x, y, label="Observed")
	plot!(p, f, x0, x1, label=L"$f(x)$")
	plot!(f_linear, xprime, label="Linear interpolation")
end

# ╔═╡ fe323332-5c06-4bb2-8d71-953fbe8a18fb
md"""
### Inverse Distance Weighting Interpolation

A simple (and commonly used) method to make predictions about new points is to come up with some sort of weighted average of other data points.
Our prediction at new point $x'$ would then be
```math
\mathbb{E}[y'] = \sum_{n=1}^N w_n y_n
```
where $\sum_{w_n} = 1$.
We're free to do what we like with the weights, but a common approach is to set the weights to 1 divided by the distance between $x$ and $x'$ (for now we will think of Euclidean distance $w_n \propto \frac{1}{(x - x')^2}$ (the $\propto$ refers to the fact that this will not be normalized -- we need to normalize so the weights sum to 1) but keep in the back of your mind that there are other ways to think about distance).

We can most efficiently compute this using linear algebra.
"""

# ╔═╡ 4457e0af-2f64-4aa9-b38a-7d417942becc
calc_dist(x1, x2) = Distances.pairwise(Distances.Euclidean(), x1, x2);

# ╔═╡ 99c4590c-38b8-45a2-ae4d-ccb6e15f94af
function idw(x, y, xprime) where T <: Real
	dist = Distances.pairwise(Distances.Euclidean(), x, xprime)
	weights = dist .^ (-2)
	weights_norm = weights ./ sum(weights, dims=1)
	return weights_norm' * y
end;

# ╔═╡ 3645b651-b65a-4847-93a8-e81707e044af
let
	yprime = idw(x, y, xprime)
	p = deepcopy(baseplot)
	plot!(p, f, x0, x1, label="True Values")
	scatter!(p, x, y, label="Observed Points")
	plot!(p, xprime, yprime, label="Predicted (IDW)")
end

# ╔═╡ 9f1aef00-4105-4e99-b433-ee1fcb00ad5c
md"""
Where we have good data, this works pretty well.
Where we lack data, predictions are flat and not very accurate.
We also see that there are some weird behaviors around data points -- the lack of "smoothness" (to be vague) raises some questions.

### K Nearest Neighbors (KNN)

A common addition to the inverse distance weighting considered here is to consider only the $K$ nearest neighbors.
That is, we set all but the $K$ smallest weights to zero before normalizing.
In practice this often improves interpolation algorithms by setting the influence of distant points to be truly zero, rather than just small.
"""

# ╔═╡ b24862e9-8f73-40a3-862a-af2d5a3e0325
function knn(x, y, xprime, K) where T <: Real
	dist = Distances.pairwise(Distances.Euclidean(), x, xprime)
	weights = dist .^ (-2)
	
	# the only step that's different here is we truncate some weights to zero
	ranks = hcat([sortperm(col; rev=true) for col in eachcol(weights)]...)
	weights[ranks .> K] .= 0

	# now normalize and done
	weights_norm = weights ./ sum(weights, dims=1)
	return weights_norm' * y
end;

# ╔═╡ f51d860e-5424-4438-bf36-c86daaa1a420
md"Notice how when $K=N$ we get our IDW estimate"

# ╔═╡ 02117329-6535-4ee6-9ce6-04565f1fad04
let
	K = N
	yprime = knn(x, y, xprime, K)
	p = deepcopy(baseplot)
	plot!(p, f, x0, x1, label="True Values", title="K=N")
	scatter!(p, x, y, label="Observed Points")
	plot!(p, xprime, yprime, label="Predicted (KNN)")
end

# ╔═╡ a2966076-90d6-41e6-b523-9039424408a8
md"But when we reduce K, we change our estimates"

# ╔═╡ 1c5c44de-b61d-401e-be0d-00a6a88a2d63
let
	K = 5
	yprime = knn(x, y, xprime, K)
	p = deepcopy(baseplot)
	plot!(p, f, x0, x1, label="True Values", title="K=$K")
	scatter!(p, x, y, label="Observed Points")
	plot!(p, xprime, yprime, label="Predicted (KNN)")
end

# ╔═╡ 6c67be2c-f23c-4da2-b8e0-5dd1b2951d10
md"""
This algorithm can be a bit choppy when points are far away from each other and $K$ is small because a very small change in $x$ wcan change the set of neighbors that is selected. This makes the weights discontinuous and can lead to 'jumpy' predictions

* Try playing around with different values of $K$. What do you notice?
* This is similar, but less smooth because the set of nearest neighbors can jump around
* Finding a good value of $K$ usually takes some playing around
"""

# ╔═╡ 6ed69ac4-f57b-4db4-95cd-04dfb527212c
md"""
### Other approaches

We're not going to cover every possible tool!

* We could fit a high-order polynomial, using least squares or similar (we've seen how to do this)
* We could use methods like Loess, Local Polynomial Regression, or cubic splines -- these have theoretical links to Gaussian Processes but deserve their own presentation.

Instead we'll look at 1D Gaussian Processes as a specific example of models for nonlinear regression.
"""

# ╔═╡ 7a44bcdb-7604-4cd5-9bfd-fd041dd10760
md"""
## 1D Gaussian Processes

If you have not yet done so, go read the suggested articles!

In this section we build a Gaussian Process for our data from scratch.
Then we'll see how to use the `GaussianProcesses.jl` package for easy and flexible inference.

### Kernel

We mentioned earlier that Euclidean distance is not the only way to think about the similarity between two points.
There are lots of other distance metrics we could use (see `Distances.jl` for some examples).
However, we can generalize further.

We will use a **kernel** $K(x, x')$ to answer the question "given $x$ and $x'$, how similar do we expect $y(x)$ and $y(x')$ to be?
A kernel maps two vectors to the real  space ($K: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}$) and one way to think kernels is as a generalization of the dot product.

We will start with the squared exponential kernel, because it's common and used in some of the tutorials we have seen.
It has the form
```math
K(x, x' | \sigma, \ell) = \sigma^2 \exp \left[ -\frac{1}{2\ell^2} (x-x')^2 \right].
```
For computational reasons we typically add $\epsilon I$ to this, where $I$ is a diagonal matrix of ones.
"""

# ╔═╡ ae04d017-7295-4483-921f-6b1d0dbadbcc
md"""
Letting $D$ be be the matrix of pairwise distances $(x-x')^2$, we can write our kernel in terms of these distances.
Let's parameterize in terms of $\log \ell$ and $\log \sigma$ because $\ell > 0$ and $\sigma > 0$, and take advantage of the fact that $\exp(x)^2 = \exp(2x)$.
"""

# ╔═╡ ce8fe84f-e70b-4173-897d-e85389f5b6a6
function sqexp_cov_fn(D, log_ℓ, log_σ) 
	exp(2 * log_σ) * exp.(- D.^2 ./ (2 * exp(2 * log_ℓ)))
end;

# ╔═╡ c3d41de5-38ff-466b-ace3-14a8c1973286
md"""
Given some known values of $\sigma$ and $\ell$, we can use this function to calculate the kernel for each pair of points in our data ($x$ and $x$ -- this will give us an $N \times N$ matrix).
For reasons that will become apparent shortly, we'll call it $\Sigma{x,x}$.
"""

# ╔═╡ 0794f0db-d6bc-4457-b652-66654a2a5368
D_x_x = Distances.pairwise(Distances.Euclidean(), x, x); # pairwise x to x

# ╔═╡ fbd84b17-683c-48fc-9cc3-3afa5ddfddca
let
	ℓ_guess = 1.0 # guess
	σ_guess = 1.0 # guess

	ϵ = 1.0e-3
	Σ_x_x = sqexp_cov_fn(D_x_x, ℓ_guess, σ_guess)

	p1 = heatmap(D_x_x, xlabel=L"$x$", ylabel=L"Also $x$", title="Distance")
	p2 = heatmap(
		Σ_x_x,
		xlabel=L"Index of $x$", ylabel=L"Index of $x$",
		title=L"$\Sigma_{x,x}$",
	)
	plot(p1, p2, size=(800, 400))
end

# ╔═╡ ba48ed3b-3131-404e-a3da-34fd975fd6f0
md"""
How does this help us?
Our  model for the $(length(x)) data points that we have thus far is
```math
p(\mathbf{y} | \sigma, \ell) \sim \mathrm{N} \left(\mu, K(x, x' | \ell, \sigma) + \sigma_y^2 I \right)
```
where $\mu = \frac{1}{N} \sum_i y_i$, $K$ is the kernel function described above, $\sigma_y$ is our noise parameter, and $I$ is the identity matrix (a square matrix with ones on the diagonal).

Using $\sigma_y$ is important -- what this is really saying is that the $y$ we observed had some normally distributed noise (which is how we generated the data!)

### Kernel Parameters

Of course, we don't know what $\sigma$ or $\ell$ is -- we guessed!
We can use this formulation to optimize $\sigma$ and $\ell$ by finding values of $\sigma$ and $\ell$ that increase the (log) probability, just like we've been doing all along.

*Note that $\sigma > 0$ and $\ell > 0$ by definition. To make life easy, we'll optimize over $\log \sigma$ and $\log \ell$.*
"""

# ╔═╡ 87f6f009-d1ab-48f4-acc4-c38089fdf5fb
function gp_logpdf(D, logℓ, logσ, logσy)
	
	N = size(D, 1)
	
	μ = ones(N) * mean(y)
	K = sqexp_cov_fn(D, logℓ, logσ)
	Σ = K + LinearAlgebra.I * exp(logσy)^2
	
	joint_dist = MvNormal(μ, Σ)
	logpdf(joint_dist, y)
end;

# ╔═╡ 101161af-d850-49da-a8f1-95b66416e552
md"Some parameter guesses are more likely than others"

# ╔═╡ eedd6b4d-9894-4932-b16d-cc219b38a0f7
gp_logpdf(D_x_x, -2.5, 2.0, 1.0)

# ╔═╡ 71abeada-179c-411c-999e-9514f12f671a
gp_logpdf(D_x_x, 0.5, 1.0, -0.1)

# ╔═╡ 7c846ff2-84cf-4f80-aaf5-5648128de166
md"""
We can even optimize this to find the "best parameters"!
Let's fix $\sigma_y$ to be constant, because otherwise the optimizer will always set $\sigma_y \rightarrow 0$.
We defined $\sigma_y=$ $(σy) above, so let's use the known value for now
"""

# ╔═╡ e7d611ac-3b9e-41e9-a466-c24fcd0ca9bf
θ_best = let
	θ0 = zeros(2)
	P = length(θ0)
	f(θ) = -gp_logpdf(D_x_x, θ[1], θ[2], log(σy))
	res = optimize(f, θ0)
	Optim.minimizer(res)
end

# ╔═╡ 95b649ad-42ee-437f-a95f-745fed8752df
md"""
Quick recap: we used a multivariate Normal distribution constructed over the data we already had to optimize the parameters of the Gaussian Process ($\sigma$ and $\ell$).
However, we're only halfway there!
Our ultimate goal is to make prediction at all the `xprime`.

### Predictions

To do that we again write down a joint probability distribution using a multivariate Normal.
Let's write this out using some slightly funky notation:
```math
\left[ \begin{matrix} y' \\ y \end{matrix} \right] \sim \mathcal{N} \left( 
\left[ \begin{matrix} \mu' \\ \mu \end{matrix} \right],
\left[ \begin{matrix} K(x',x') & K(x', x) \\ K(x, x') & K(x, x) \end{matrix} \right]
+ \sigma_y^2 I
\right) 
```
This is not a $2 \times 2$ matrix!
If $x$ has $N$ points and $x'$ has $N'$ points, then 
* Block $K(x', x')$ will give us an $N' \times N'$ matrix
* Block $K(x', x)$ will give us an $N' \times N$ matrix
* Block $K(x, x')$ will give us an $N \times N'$ matrix
* Block $K(x, x)$ will give us an $N \times N$ matrix

If you put all those pieces together (four blocks) you get an $(N + N') \times (N + N')$ covariance matrix

The good news is that, since we know the parameters of our kernel (for the squared exponential kernel we have chosen, $\ell$ and $\sigma$ but we would have different parameters if we chose a different kernel), we can fill in all of these pieces.
"""

# ╔═╡ 1f1056a4-e75d-4c27-b302-96d62e674302
md"""
If you're reading the [GP not quite for dummies](https://yugeten.github.io/posts/2019/09/GP/), the author names the four blocks above:
```math
\left[ \begin{matrix} y' \\ y \end{matrix} \right] \sim \mathcal{N} \left( 
\left[ \begin{matrix} \mathbf{a} \\ \mathbf{b} \end{matrix} \right],
\left[ \begin{matrix} A & B \\ B^T & C \end{matrix} \right]
+ \sigma_y^2 I
\right)
```
($B^T$ is the transpose of $B$).

As discussed in the not quite for dummies post, we have
```math
\begin{align}
p(y' | y) &= \mathcal{N} \left( \mathbf{m}, S \right) \\
\mathbf{m} &= \mathbf{a} + B C^{-1} \left(y - \mathbf{b} \right) \\
S &= A - B C^{-1} B^{T} + \sigma_y^2 I
\end{align}
```
where $B^{T}$ is the transpose of $B$.
Note that we are assuming stationary means: $\mathbf{a} = \mathbf{b} = \mu = \frac{1}{N} \sum y_j$.
"""

# ╔═╡ f623d720-74b3-4995-9cf4-e1af0988b575
"""
Joint distribution of yprime as a function of y (given parameters, x, and xnew)
"""
function predict_gp(logℓ, logσ, logσy, x, y, xnew)

    N = length(x)
    M = length(xnew)
    Q = N + M
    Z = vcat(xnew, x) # all obs
    D = pairwise(Euclidean(), Z, Z)
	μ = mean(y)

	# the entire kernel in one step [A B; B' C]
	K = sqexp_cov_fn(D, logℓ, logσ) + exp(2 * logσy) * I

	# indices
	new = 1:M
	old = (M+1):(M+N)

	# A, B, C, a, b
	A = K[new, new]
	B = K[new, old]
	C = K[old, old]
	a = μ * ones(length(xnew)) # vector
	b = μ * ones(length(x)) # vector

	# compute m, S
	m = a + B / C * (y - b)
	S = A - B / C * B'
	Σ = S + exp(2 * logσy) * I # add observational noise
	Σ = Matrix(Hermitian(S)) # linear algebra trick
	return MvNormal(m, Σ) #  we return the joint distribution
end;

# ╔═╡ 80ab8bfb-7ab1-4425-804c-794c90a8a4de
let
	mvn = predict_gp(θ_best[1], θ_best[2], log(σy), x, y, xprime)
	p = deepcopy(baseplot)
	σ = sqrt.(diag(mvn.Σ))
	plot!(
		p, xprime, mvn.μ, ribbon=2σ,
		label=L"Gaussian Process $\pm 2\sigma$", linewidth=3,
	)
	plot!(p, f, 0, 10, label="True Value", linewidth=2)
	scatter!(p, x, y, label="Observed Points")
	plot!(p, title="Linear Algebra Fit (No Package)")
end

# ╔═╡ a450382a-4c03-4c69-b03f-b55233458a99
md"""
### `GaussianProcesses.jl`

It's good to work through the linear algebra from scratch.
However, we can use the `GaussianProcesses.jl` packages to use better, more stable implementations (they are more stable because they take advantage of numerical tricks, factorizations, etc) and that have good plotting capabilities.
"""

# ╔═╡ ebc6554d-e753-4c8a-9e83-1e942c3c90fa
md"""
In order to build a GP, we need to specify

* A kernel
* A mean function
* A noise parameter $\sigma_y$

In `GaussianProcesses.jl`, we specify the parameters $\ell, \sigma, \sigma_y$ by their log values (to ensure no negative values).

Let's start by making up some  values of the parameters.
This shouldn't fit the data well!
We can plot really easily.
"""

# ╔═╡ 12b5c75a-31a4-4452-8dea-2a62167c2db6
let
	ℓ = 2.5
	σ = 1.5
	kern = GaussianProcesses.SE(log(ℓ), log(σ))
	μ = GaussianProcesses.MeanZero()
	gp = GP(x, y, μ, kern, log(σy))
	p = deepcopy(baseplot)
	plot!(p, f, x0, x1, label="True Values", linewidth=3)
	plot!(gp, title=L"Random Guess Parameters: $\ell=%$(ℓ)$, $\sigma=%$σ$", linewidth=3, label="Fit")
end

# ╔═╡ 6d1787a3-4b14-406d-86b4-11d010bea570
md"We can repeat the exercise by plugging in the parameters that we developed before. We're still using the true, known, value of $\sigma_y$.
Since the model we estimated didn't account for noise, we can probably do better than this."

# ╔═╡ 05672787-edd7-4cd3-8cb8-bf4b1d79dd89
let
	kern = GaussianProcesses.SE(θ_best...) # note: log ℓ, log σ
	μ = GaussianProcesses.MeanConst(mean(y)) # constant mean
	gp2 = GP(x, y, μ, kern, log(σy))
	p = deepcopy(baseplot)
	plot!(p, f, x0, x1, label="True Values", linewidth=3)
	plot!(gp2, label=false, title="Best Parameters (Manual Optimization)", linewidth=3)
end

# ╔═╡ 10a73e7d-cff1-4255-bf37-fbbf70de8dfb
md"""
Last but not least, we can use stable built-in optimization methods.
(There are some well-known numerical issues that Gaussian Process models can run into, and avoiding them takes luck or skill or both).
"""

# ╔═╡ a6a21dca-0ae3-4eb4-8360-f2ac9050e802
gp3 = let
	kern = GaussianProcesses.SE(θ_best...)
	μ = GaussianProcesses.MeanZero()
	gp3 = GP(x, y, μ, kern, log(σy))
	optimize!(gp3) # piece of cake!
	gp3
end;

# ╔═╡ 69e54f00-9e8e-4f04-be87-52e6485bb96c
let
	p = deepcopy(baseplot)
	plot!(
		p, f, x0, x1, label="True Values", linewidth=3,
		title="Best Parameters (Built-In Optimization)",
	)
	plot!(p, gp3, label="GP Fit", legend=:bottomright, linewidth=3)
end

# ╔═╡ 3621ecae-4856-4dcb-90bd-3a2b41ebfb2b
md"This looks much better! We can see that the uncertainty grows where we don't have any data, which seems appropriate!"

# ╔═╡ d58d5828-1f84-4801-9686-7c0d8ea81edf
md"""
## Recap

As we discussed in class (and have been discussing all semester), often we want to know not only what is $\mathbb{E}[y' | x']$, but also what the uncertainty in that esitmate is.
Gaussian Processes provide a flexible and principled way to address this question for 1D regression models.

Next week we'll look at applying these to spatial data (which is 2D).
This won't require a lot of theoretical innovations (everything is the same except that $x$ and $x'$ are now vectors) but will require a lot of practical considerations about choosing a kernel, comparing to data, and, as always, implementation.
"""

# ╔═╡ 2847aaee-3f38-4c39-b8a6-b81d87d6f370
md"""
## Appendix
"""

# ╔═╡ 83e9c00c-c8e8-4577-9628-59f26b4831d9
TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GaussianProcesses = "891a1506-143c-57d2-908e-e1f8e92e6de9"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Distances = "~0.10.7"
Distributions = "~0.24.18"
GaussianProcesses = "~0.12.4"
Interpolations = "~0.13.5"
LaTeXStrings = "~1.3.0"
Optim = "~1.6.2"
Plots = "~1.27.1"
PlutoUI = "~0.7.37"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "6e8fada11bb015ecf9263f64b156f98b546918c7"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.5"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c9a6160317d1abe9c44b3beb367fd448117679ca"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.13.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a837fdf80f333415b69684ba8e8ae6ba76de6aaa"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.24.18"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "90b158083179a6ccbce2c7eb1446d5bf9d7ae571"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.7"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[ElasticArrays]]
deps = ["Adapt"]
git-tree-sha1 = "a0fcc1bb3c9ceaf07e1d0529c9806ce94be6adf9"
uuid = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
version = "1.2.9"

[[ElasticPDMats]]
deps = ["LinearAlgebra", "MacroTools", "PDMats"]
git-tree-sha1 = "5157c93fe9431a041e4cd84265dfce3d53a52323"
uuid = "2904ab23-551e-5aed-883f-487f97af5226"
version = "0.2.2"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "58d83dd5a78a36205bdfddb82b1bb67682e64487"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.4.9"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "56956d1e4c1221000b7781104c58c34019792951"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.11.0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "9f836fb62492f4b0f0d3b06f55983f2704ed0883"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a6c850d77ad5118ad3be4bd188919ce97fffac47"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.0+0"

[[GaussianProcesses]]
deps = ["Distances", "Distributions", "ElasticArrays", "ElasticPDMats", "FastGaussQuadrature", "ForwardDiff", "LinearAlgebra", "Optim", "PDMats", "Printf", "ProgressMeter", "Random", "RecipesBase", "ScikitLearnBase", "SpecialFunctions", "StaticArrays", "Statistics", "StatsFuns"]
git-tree-sha1 = "9cf8ba8037e332b1be14c71e549143e68c42a22d"
uuid = "891a1506-143c-57d2-908e-e1f8e92e6de9"
version = "0.12.4"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "65e4589030ef3c44d3b90bdc5aac462b4bb05567"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.8"

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

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b15fc0a95c564ca2e0a7ae12c1f095ca848ceb31"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.5"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "4f00cc36fede3c04b8acf9b2e2763decfdcecfa6"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.13"

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

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "bc0a748740e8bc5eeb9ea6031e6f050de1fc0ba2"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.6.2"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "95a4038d1011dfdbde7cecd2ad0ac411e53ab1bc"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.10.1"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "1690b713c3b460c955a2957cd7487b1b725878a7"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.1"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "bf0a1121af131d9974241ba53f601211e9303a9e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.37"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "995a812c6f7edea7527bb570f0ac39d0fb15663c"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "cbf21db885f478e4bd73b286af6e67d1beeebe4c"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.4"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "87e9954dfa33fd145694e42337bdd3d5b07021a6"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6976fab022fea2ffea3d945159317556e5dad87c"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.2"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "25405d7016a47cf2bd6cd91e66f4de437fd54a07"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.16"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═c2135d1a-3b5b-43bb-a433-4148ba69bec9
# ╟─4e0a32e2-a941-11ec-27ed-958ec70306f3
# ╠═2d62c571-53f3-4866-8901-0a8fb6f142b5
# ╠═1032164a-894b-45c0-a6f6-96c280fddf00
# ╠═595e7e4f-60a5-4666-903b-c292a57e82af
# ╟─1f636e3c-172e-4aa3-a50b-370aacf0a24e
# ╟─17b605dc-fd6a-4bb5-90f7-d827cc1e407c
# ╠═c76a7aab-b613-40a8-8aa0-9a46cc0c560a
# ╠═77164223-fc16-4af9-ba61-2832404d2ee2
# ╟─fe323332-5c06-4bb2-8d71-953fbe8a18fb
# ╠═4457e0af-2f64-4aa9-b38a-7d417942becc
# ╠═99c4590c-38b8-45a2-ae4d-ccb6e15f94af
# ╠═3645b651-b65a-4847-93a8-e81707e044af
# ╟─9f1aef00-4105-4e99-b433-ee1fcb00ad5c
# ╠═b24862e9-8f73-40a3-862a-af2d5a3e0325
# ╟─f51d860e-5424-4438-bf36-c86daaa1a420
# ╠═02117329-6535-4ee6-9ce6-04565f1fad04
# ╟─a2966076-90d6-41e6-b523-9039424408a8
# ╠═1c5c44de-b61d-401e-be0d-00a6a88a2d63
# ╟─6c67be2c-f23c-4da2-b8e0-5dd1b2951d10
# ╟─6ed69ac4-f57b-4db4-95cd-04dfb527212c
# ╟─7a44bcdb-7604-4cd5-9bfd-fd041dd10760
# ╟─ae04d017-7295-4483-921f-6b1d0dbadbcc
# ╠═ce8fe84f-e70b-4173-897d-e85389f5b6a6
# ╟─c3d41de5-38ff-466b-ace3-14a8c1973286
# ╠═0794f0db-d6bc-4457-b652-66654a2a5368
# ╠═fbd84b17-683c-48fc-9cc3-3afa5ddfddca
# ╟─ba48ed3b-3131-404e-a3da-34fd975fd6f0
# ╠═87f6f009-d1ab-48f4-acc4-c38089fdf5fb
# ╟─101161af-d850-49da-a8f1-95b66416e552
# ╠═eedd6b4d-9894-4932-b16d-cc219b38a0f7
# ╠═71abeada-179c-411c-999e-9514f12f671a
# ╟─7c846ff2-84cf-4f80-aaf5-5648128de166
# ╠═e7d611ac-3b9e-41e9-a466-c24fcd0ca9bf
# ╟─95b649ad-42ee-437f-a95f-745fed8752df
# ╟─1f1056a4-e75d-4c27-b302-96d62e674302
# ╠═f623d720-74b3-4995-9cf4-e1af0988b575
# ╠═80ab8bfb-7ab1-4425-804c-794c90a8a4de
# ╟─a450382a-4c03-4c69-b03f-b55233458a99
# ╟─ebc6554d-e753-4c8a-9e83-1e942c3c90fa
# ╠═12b5c75a-31a4-4452-8dea-2a62167c2db6
# ╟─6d1787a3-4b14-406d-86b4-11d010bea570
# ╠═05672787-edd7-4cd3-8cb8-bf4b1d79dd89
# ╟─10a73e7d-cff1-4255-bf37-fbbf70de8dfb
# ╠═a6a21dca-0ae3-4eb4-8360-f2ac9050e802
# ╠═69e54f00-9e8e-4f04-be87-52e6485bb96c
# ╟─3621ecae-4856-4dcb-90bd-3a2b41ebfb2b
# ╟─d58d5828-1f84-4801-9686-7c0d8ea81edf
# ╟─2847aaee-3f38-4c39-b8a6-b81d87d6f370
# ╠═e4ccd0c6-5404-4f8f-a6cf-5ad5f67fe47b
# ╠═83e9c00c-c8e8-4577-9628-59f26b4831d9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
