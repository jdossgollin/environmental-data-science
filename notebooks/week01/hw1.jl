### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 05616b15-728b-47ee-8d8d-d030dbe0bdb7
begin
	using Pkg
	#Pkg.upgrade_manifest()
	Pkg.update()
end

# ╔═╡ 48c4bef4-c5ba-4ef3-bc9c-50039eaeb60f
begin
    using PlutoUI
    TableOfContents()
end

# ╔═╡ 9b1923fd-6247-4a1d-bd93-d1183bdcc565
using StatsBase: quantile

# ╔═╡ 17f535ff-45e8-43ea-b1bf-cedceefecbc9
using Distributions: Normal

# ╔═╡ fafae38e-e852-11ea-1208-732b4744e4c2
md"_Homework 1, version 1 -- Spring 2022_"

# ╔═╡ 7308bc54-e6cd-11ea-0eab-83f7535edf25
# edit the code below to set your name and net ID (i.e. email without rice.edu)

student = (name="James Doss-Gollin", netid="jd82")

# press the ▶ button in the bottom right of this cell to run your edits
# or use Shift+Enter

# you might need to wait until all other cells in this notebook have completed running. 
# scroll down the page to see what's up

# ╔═╡ cdff6730-e785-11ea-2546-4969521b33a7
md"""
Submission by: **_$(student.name)_** ($(student.netid)@rice.edu)
"""

# ╔═╡ a2181260-e6cd-11ea-2a69-8d9d31d1ef0e
md"""
# Homework 1: Getting up and running with Julia

**HW1 due date: Thursday 1/13**.

First of all, **_welcome to the course!_**
I am excited to teach you about the exciting world of modeling and understanding environmental data.

This is a short assignment to get you set up and familiar with the toolkit we'll use this semester.
If you run into issues, please post on the `HW1 & installing Julia / Pluto` thread (go to Canvas and then Discussions).

Make sure you've seen this week's lectures and found the notebooks on the course website!
"""

# ╔═╡ 31a8fbf8-e6ce-11ea-2c66-4b4d02b41995
md"""## Homework Logistics
Homeworks are in the form of [Pluto notebooks](https://github.com/fonsp/Pluto.jl). Your must complete them and submit them on [Canvas](https://canvas.rice.edu/courses/48366) (if you are an enrolled Rice student.).

HW1 is for you to get your system set up correctly and to test our grading software.

You can also click on the eyeball icon at the upper left of any code block to reveal the code that generated a particular output.
This is a great way to build some intuition about how Julia and Pluto work.
"""

# ╔═╡ f9d7250a-706f-11eb-104d-3f07c59f7174
md"""
## Requirements for HW1

- Install Julia and set up Pluto    
- Do the required Exercises
"""

# ╔═╡ 430a260e-6cbb-11eb-34af-31366543c9dc
md"""## Installation
Before being able to run this notebook succesfully locally, you will need to **set up Julia and Pluto** (click "Software installation" on the left).

One you have Julia and Pluto installed, you can click the button at the top right of this page and follow the instructions to edit this notebook locally and submit.
"""

# ╔═╡ a05d2bc8-7024-11eb-08cb-196543bbb8fd
md"## Exercise 1 - _Making a basic function_

Computing the square of a number is easy -- you just multiply it with itself.

##### Algorithm:

Given: $x$

Output: $x^2$

1. Multiply `x` by `x`"

# ╔═╡ e02f7ea6-7024-11eb-3672-fd59a6cff79b
function basic_square(x)
    return 1 # this is wrong, write your code here!
end

# ╔═╡ 6acef56c-7025-11eb-2524-819c30a75d39
let
    result = basic_square(5)
    if !(result isa Number)
        md"""
      !!! warning "Not a number"
          `basic_square` did not return a number. Did you forget to write `return`?
      		"""
    elseif abs(result - 5 * 5) < 0.01
        md"""
      !!! correct
          Well done!
      		"""
    else
        md"""
      !!! warning "Incorrect"
          Keep working on it!
      		"""
    end
end

# ╔═╡ 8178e21c-4116-423c-8a40-03e868c6e943
md"""
## Exercise 2 - _Simulating Random Variables_

Now that you know how to write a function, let's use a function _to do stats_!
Specifically, in this class we'll use simulation to answer questions that are hard to solve analytically.

As an example problem, let's imagine that we wanted to measure the difference betweem the 95th percentile of $x$ and $x^2 - x$, where $x$ is a Normal random variable with known mean μ (to write this Greek character in Pluto, write `\mu` and press `tab`; pretty cool huh?) and standard deviation σ (`\sigma` + `tab`).
This is the sort of problem that, if it's tractable at all, is a pain!

Fortunately, it's not so hard if we use the right tools.
First, we'll use the `StatsBase.jl` package, which provides the function `quantile`, which does exactly what you'd expect:
"""

# ╔═╡ b1da36db-64e8-4611-884f-d2190547f6a0
quantile(0:100, 0.95)

# ╔═╡ 888e0b39-d416-4e13-99f6-49e85be85b6f
md"""
Next let's call the [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/) package, which we will use all semester, and which provides (among many other distributions) `Normal`
"""

# ╔═╡ cc2816b9-8d63-41e3-bdbf-a6c8b512bdaf
md"""
This package also works pretty intuitively.
"""

# ╔═╡ 73dfa406-b4e9-49c5-8411-25d5db9c7b41
my_normal_dist = Normal(3.5, 1.15)

# ╔═╡ 100148f2-27dd-4e2a-849d-63f1acf8e630
md"You can even combine it with `quantile` for distributions that have analytic solutions!"

# ╔═╡ 906cb1f4-4a14-41f9-8939-910ad898e7b1
quantile(my_normal_dist, 0.95)

# ╔═╡ 3a072803-7b63-421c-85be-bd416d78b706
md"Importantly, we can draw random samples from a distribution"

# ╔═╡ 3f32648c-98ca-4e4c-b84a-9375b921e938
rand(my_normal_dist)

# ╔═╡ 2aac1b6d-5db9-41a3-918a-6eddf747c45a
rand(my_normal_dist, 100)

# ╔═╡ b6032a17-68db-49e4-8a2d-5f1ae3e18c3e
md"""
OK, so how are we going to use these tools to estimate the difference between the 95th percentile of $x^2 - x$ and the 95th percentile of $x$?

Our recipe looks like this

1. Draw N (should be large) simulations of $x$
1. Estimate the 95th percentile of $x$
1. Calculate $y = x^2 - x$ for each of the N simulations
1. Estimate the 95th percentile of $y$
1. Compute the difference (95th perecntile of $y$ - 95th percentile of $x$)

This is something we can put in a function!
This function will take, as inputs, μ, σ, and N.
"""

# ╔═╡ 5f3aa9a5-3201-4e6e-b925-ea103abecf3b
function exercise_2(μ, σ, N)
    x_dist = Normal(μ, σ)
    x_sims = 1 # this is wrong; draw N samples of x
    x_q95 = 0.5 # wrong; estimate the 95th percentile of these
    y_sims = x_sims .^ 2 .- x_sims # this is right. But **why do we need the `.`**?
    y_q95 = 0.5 # this is wrong; estimate the 95th perecntile of y
    # return the difference
    return 0
end

# ╔═╡ 22d08726-3953-4c9e-b08b-6839e91d0f5e
let
    N = 100_000
    μ, σ = (3.5, 1.5)
    result2 = exercise_2(μ, σ, N)
    target2 = 23.6674
    if result2 == 0
        md"""
      !!! warning "Returns Zero"
          `exercise_2` returned the default value of zero. Did you forget to update the `return` statement?
      		"""
    elseif abs(result2 - target2) < 1 # large buffer!
        md"""
      !!! correct
          Well done!
      		"""
    else
        md"""
      !!! warning "Incorrect"
          Keep working on it!
      		"""
    end
end

# ╔═╡ b3c7a050-e855-11ea-3a22-3f514da746a4
if student.netid === "jd82"
    md"""
   !!! danger "Oops!"
       **Before you submit**, remember to fill in your name and netID at the top of this notebook!
   	"""
end

# ╔═╡ d5c2b83c-ecb3-4387-a528-f4e921a4264e
md"""
## Rubric

See rubric on Canvas
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Distributions = "~0.25.37"
PlutoUI = "~0.7.29"
StatsBase = "~0.33.14"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.6.5"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "54fc4400de6e5c3e27be6047da2ef6ba355511f8"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "5863b0b10512ed4add2b5ec07e335dc6121065a5"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.41"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "22df5b96feef82434b07327e2d3c770a9b21e023"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "92f91ba9e5941fc781fecf5494ac1da87bdac775"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "5c0eb9099596090bb3215260ceca687b888a1575"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.30"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e08890d19787ec25029113e88c34ec20cac1c91e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.0.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═48c4bef4-c5ba-4ef3-bc9c-50039eaeb60f
# ╟─fafae38e-e852-11ea-1208-732b4744e4c2
# ╠═cdff6730-e785-11ea-2546-4969521b33a7
# ╠═7308bc54-e6cd-11ea-0eab-83f7535edf25
# ╟─05616b15-728b-47ee-8d8d-d030dbe0bdb7
# ╟─a2181260-e6cd-11ea-2a69-8d9d31d1ef0e
# ╟─31a8fbf8-e6ce-11ea-2c66-4b4d02b41995
# ╟─f9d7250a-706f-11eb-104d-3f07c59f7174
# ╟─430a260e-6cbb-11eb-34af-31366543c9dc
# ╟─a05d2bc8-7024-11eb-08cb-196543bbb8fd
# ╠═e02f7ea6-7024-11eb-3672-fd59a6cff79b
# ╟─6acef56c-7025-11eb-2524-819c30a75d39
# ╟─8178e21c-4116-423c-8a40-03e868c6e943
# ╠═9b1923fd-6247-4a1d-bd93-d1183bdcc565
# ╠═b1da36db-64e8-4611-884f-d2190547f6a0
# ╟─888e0b39-d416-4e13-99f6-49e85be85b6f
# ╠═17f535ff-45e8-43ea-b1bf-cedceefecbc9
# ╟─cc2816b9-8d63-41e3-bdbf-a6c8b512bdaf
# ╠═73dfa406-b4e9-49c5-8411-25d5db9c7b41
# ╟─100148f2-27dd-4e2a-849d-63f1acf8e630
# ╠═906cb1f4-4a14-41f9-8939-910ad898e7b1
# ╟─3a072803-7b63-421c-85be-bd416d78b706
# ╠═3f32648c-98ca-4e4c-b84a-9375b921e938
# ╠═2aac1b6d-5db9-41a3-918a-6eddf747c45a
# ╟─b6032a17-68db-49e4-8a2d-5f1ae3e18c3e
# ╠═5f3aa9a5-3201-4e6e-b925-ea103abecf3b
# ╟─22d08726-3953-4c9e-b08b-6839e91d0f5e
# ╟─b3c7a050-e855-11ea-3a22-3f514da746a4
# ╟─d5c2b83c-ecb3-4387-a528-f4e921a4264e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
