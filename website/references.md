# References

There are lots of great resources beyond this website.
Here are some particularly good ones.

## Julia cheat sheets

1. [Plots and visualizations](/plots_tutorial/)
1. [Plotting cheatsheet](https://github.com/sswatson/cheatsheets/blob/master/plotsjl-cheatsheet.pdf)
1. JuliaDocs: [Fastrack to Julia](https://juliadocs.github.io/Julia-Cheat-Sheet/) cheatsheet
1. [QuantEcon group](https://quantecon.org): [MATLAB-Julia-Python comparative cheatsheet](https://cheatsheets.quantecon.org/)

## Helpful stats codes

1. Code examples from the Statistical Rethinking book have been coded in `Turing.jl` [here](https://statisticalrethinkingjulia.github.io/TuringModels.jl/) (the original is R)
1. the [Stan users guide](https://mc-stan.org/docs/2_28/reference-manual/index.html) offers detailed and practical advice on probabilistic modeling using the Stan language, which is similar to `Turing.jl` in many ways.
1. the [`Turing.ml` tutorials](https://turing.ml/dev/tutorials/) offer an introduction to modeling with Turing
1. the [Earth and Environmental Data Science](earth-env-data-science.github.io/) textbook by Ryan Abernathey and colleagues is freely available online. This is a very different _and complementary_ take on data science from this course (it's a big field). If you're looking for best practices in the analysis of gridded climate data, you should check out this course.
1. Austin Rochford has a series of [blog posts](https://austinrochford.com/posts/intro-prob-prog-pymc.html) that use the PyMC language (in Python) to teach some introductory probabilistic computing concepts. It's worth a look if you want to dig into the (very good!) probabilistic computing ecosystem in Python.
1. There is a cool website called [Earth Lab](https://www.earthdatascience.org/about/) with some helpful tutorials of various things, mostly in R and Python

## Julia tips and tricks

- the MIT course [Intro to Computational Thinking](https://computationalthinking.mit.edu/) was an inspiration for the digital organization of this course and offers great lectures on a number of computational and mathematical tools
- [Tim Holy](https://neuroscience.wustl.edu/people/timothy-holy-phd/) has a course called [Advanced Scientific Computing: Producing Better Code](https://www.youtube.com/watch?v=x4oi0IKf52w&list=PL-G47MxHVTewUm5ywggLvmbUCNOD2RbKA) that provides intermediate to advanced tips and tricks

## Textbooks

There is no assigned textbook for this course, but the following texts may be helpful as you explore specific concepts.

For a fairly detailed exploration of hierarchical space-time models, see Cressie and Wikle (2011).
There have been some computational advances since then that are worth keeping in mind before you apply these models directly, but it's a clearly written overview.

> Cressie, Noel A. C., and Christopher K. Wikle. 2011. Statistics for Spatio-Temporal Data. Hoboken, N.J.: Wiley.

The USGS uses the following manual.
As you can see, there is a tremendous gap between modern statistical theory and methodology, and what federal agencies *require* analysis to look like.

> England Jr., John F., Timothy A Cohn, Beth A. Faber, Jery R. Stedinger, Wilbert O. Thomas Jr., Andrea G. Veilleux, Julie E. Kiang, and Robert R. Mason Jr. 2019. “Guidelines for Determining Flood Flow Frequency Bulletin 17c.” In Techniques and Methods, Version 1.1, 4-B5:148. U.S.Geological Survey. [https://doi.org/10.3133/tm4B5](https://doi.org/10.3133/tm4B5).

In parallel is the USGS textbook Statistical Methods in Water Resources (there has been a recent update but it's mostly a cosmetic one).

> Helsel, Dennis R., and Robert M. Hirsch. 2002. Statistical Methods in Water Resources. Techniques of Water-Resources Investigations. Reston, VA: U.S. Geological Survey. [https://doi.org/10.313 3/twri04A3](https://doi.org/10.313 3/twri04A3).

A classic textbook for Bayesian analysis is Gelman et al (2014)

> Gelman, Andrew, John B Carlin, Hal S Stern, and Donald B Rubin. 2014. Bayesian Data Analysis. 3rd ed. Chapman & Hall/CRC Boca Raton, FL, USA.

A simliar (slightly broader) set of authors has written a manual on 'Bayesian workflow', which everyone should look at

> Gelman, Andrew, Aki Vehtari, Daniel Simpson, Charles C. Margossian, Bob Carpenter, Yuling Yao, Lauren Kennedy, Jonah Gabry, Paul-Christian 1. Bürkner, and Martin Modrák. 2020. “Bayesian Workflow.” November 3, 2020. [http://arxiv.org/abs/2011.01808](http://arxiv.org/abs/2011.01808).

An alternative to the Gelman et al textbook is McElreath's Statistical Rethinking textbook.
An advantage of this text is that there are lectures online that directly follow the book.
If you've learned statistical approaches that look like long "flow charts" devoid of any meaningful statistical theory (see: USGS references above) this book spends a lot of time explaining what's wrong with that approach.

> McElreath, Richard. 2020. Statistical Rethinking: A Bayesian Course with Examples in R and Stan. Second edition. Texts in Statistical Science Series. Boca Raton ; CRC Press, Taylor & Francis Group.

A slightly less technical textbook, but with really clear and well-worked examples, is

> Gelman, A. (2021). Regression and other stories. Cambridge, United Kingdom; Cambridge University Press. (free PDF available on the book's [website](https://avehtari.github.io/ROS-Examples/index.html))

To take another tack, if you want to think about links between nonlinear dynamics and statistics, this long review article by Ghil et al is worth a read

> Ghil, M, P Yiou, S Hallegatte, B D Malamud, P Naveau, A Soloviev, P Friederichs, et al. 2011. “Extreme Events: Dynamics, Statistics and Prediction.” Nonlinear Processes in Geophysics 18 (3): 295–350. [https://doi.org/10.5194/npg-18-295-2011](https://doi.org/10.5194/npg-18-295-2011).

For a more classic take on machine learning, Introduction to Statistical Learning is a classic

> James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani. 2013. An Introduction to Statistical Learning. Vol. 103. Springer Texts in Statistics. New York, NY: Springer New York.

## Other sources

1. Michael Betancourt has a number of highly detailed [essays](https://betanalpha.github.io/writing/) on probabilistic computing (often using R) that I highly recommend. They are quite technical, so be warned.
1. The wiki for the stan probabilistic programming language (a gold standard although sometimes tricky to work with) has a set of [prior choice recommendations](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations) that is a go-to reference.
1. More generally, the [documentation](https://mc-stan.org/users/documentation/) for the stan language has a number of detailed and well-worked examples. It's a fantastic reference for building any kind of probabilistic model, even if you're going to use a different language (with a few exceptions, these models should be fairly straightforward to translate from one language to another)
1. Similarly, the Turing language has some good [tutorials](https://turing.ml/dev/tutorials/) that are worth looking at, and the [PyMC docs](https://docs.pymc.io/en/v3/nb_examples/index.html) are also excellent.
1. There are lots of ethical and social issues pertaining to the use and application of environmental data science. We will barely scratch the surface of these issues in this class; for a reading list, have a look at [this course](https://beth-tellman.github.io/hum-envseminar.docx.pdf) by Beth Tellman at the University of Arizona
