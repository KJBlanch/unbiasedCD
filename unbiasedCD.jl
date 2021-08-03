### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 03fcb360-421d-441a-bd8c-5372d6bb2be5
using PlutoUI # To enable table of contents

# ╔═╡ 61ce3068-a319-4f67-b4fd-25745727f0a1
using Flux # Julias main Deep Learning library. We don't need autodiff for this project, but Flux is still useful for activation functions and stateful optimizers.

# ╔═╡ 8b11badc-2a03-4918-841a-a6459d1aac28
PlutoUI.TableOfContents(depth = 6)

# ╔═╡ ef35700e-8df6-4446-b9f4-2e82bf8801c0
html"<button onclick='present()'>present</button>"

# ╔═╡ 9e830b5e-f37f-11eb-083f-277a24c3cd6c
md"# Unbiased Contrastive Divergence
The paper *Unbiased Contrastive Divergence Algorithm for Training Energy-Based Latent Variable Models* by Yiuxuan Qiu, Lingsong Zhang and Xiao Wang highlights a known an issue with the Contrastive Divergence algorithm. Namely the fact that it produces biased gradient estimates. Qiu et al proposes a novel and computationally efficient way to solve this issue based on recent advances in MCMC methods.
"

# ╔═╡ 1a5a0086-cfba-470f-955f-82df7c5f19de
md"
- **Note**: This document is a set of interactive notes written to help us understand and re-implement results from the above mentioned paper *Unbiased Contrastive Divergence (...)* by Qiu et al. This is not original content, but just our notes based on the paper and various tutorial as well as the code for our reimplementation.
"

# ╔═╡ d9e2c1de-ee61-4413-8b41-8bcad7206d1d
md"## How to train an RBM with standard CD
Geoffrey Hinton's *A Practical Guide to Training Restricted Boltzmann Machines* is a good starting point for understanding Restricted Boltzmann Machines. Asja Fischer's and Cristian Igel's *Training Restricted Boltzmann Machines: An Introduction* is also very useful, and is highlighted in the *Unbiased Contrastive Divergence (...)* paper which we are studying.
"

# ╔═╡ adb63694-4f58-4d96-84c2-87e3fd69d5ec
md"### The RMB energy function

"

# ╔═╡ 5a5299e2-4a18-4a52-ae87-453380edc682
md" An RBM is governed by the following energy function 
$E(v,h) = \sum_{i\in \textrm{visible}} a_i v_i - \sum_{j\in \textrm{hidden}} b_j h_j - \sum_{i,j}v_i h_j W_{ij}$."

# ╔═╡ d33cc5e2-9135-4dd0-b043-67ff5b5edaf6
md"
Where $W$ is a matrix of learnable synaptic weights and $a$ and $b$ are learnable bias vectors. $v$ and $h$ are the vectors of visible and hidden units respectively. Before we continue to look at how to update the units  how to compute a learning signal let's implement a network struct and an energy function."

# ╔═╡ 5179313d-576f-4433-82cc-bf2cb7907abd
mutable struct rbmstruct
	W::Array{Float32, 2}
	b::Array{Float32, 1}
	a::Array{Float32, 1}
end

# ╔═╡ b775575b-33e7-4708-8c6e-4c28f9cfa79f
md" We will generate some quick random data to try out our energy function with. We will force the hidden units to be binary variables for now using the Heaviside step function."

# ╔═╡ 72a1fd39-2980-4f14-9a67-5362f9bb0775
heaviside(x) = 0.5 * (sign(x) + 1)

# ╔═╡ a9c6ecab-bc6c-4565-9a29-7d07b95c2de9
begin
	numvisible = 30
	numhidden = 50
	# Some initial network parameters
	W = randn(Float32, numhidden, numvisible)
	a = randn(Float32, numvisible)
	b = randn(Float32, numhidden)
	# some dummy state vectors
	h = heaviside.(rand(Float32, numhidden) .- 0.5)
	v = heaviside.(rand(Float32, numvisible) .- 0.5)
end;

# ╔═╡ 4f485f1b-33ff-4ea9-a67e-11e5841fcd62
# Initialize the network
rbm = rbmstruct(W, b, a);

# ╔═╡ 6d210251-d433-43b6-b515-c852ccbc1feb
function energy_single_datapoint(rmb, v, h)
	E = sum(rbm.a .* v) - sum(rbm.b .* h) - sum(h*v' .* W)
end

# ╔═╡ 7d86146b-dee5-477e-b476-4a1d888831aa
md"Let's check what the energy of our randomly initialized network is!"

# ╔═╡ 8b171005-2f80-492a-bf0a-e85088d498a0
energy_single_datapoint(rbm, v, h)

# ╔═╡ 3047d526-f2b6-48f3-b5e1-d5290eddb25a
md"### The training objective

"

# ╔═╡ 85a021d9-eddf-4167-817a-fcee142924ae
md"Similar to in statistical physics the probability of the system being in a certain state is given by the Boltzmann distribution. This part is explained well in Hinton's notes."

# ╔═╡ 5d905b9a-6c86-4361-8c84-39cb320c071e
md"$p(\pmb{v},\pmb{h}) = \frac{e^{-E(\pmb{v},\pmb{h})}}{Z}$"

# ╔═╡ b6fc90cf-ca23-49a3-93dc-8bdb9332faec
md"where $Z$ is given by $Z = \sum_{\pmb{v},\pmb{h}} e^{-E(\pmb{v},\pmb{h})}$. The sum here runs over all possible state pars $(\pmb{v},\pmb{h})$. 
This factor seems intractable for continuous variables, but tractable (and impractical for moderately sized $\pmb{v}$ and $\pmb{h}$) for binary variables. The probability corresponding to a visible vector $\pmb{v}$ is"

# ╔═╡ 0c28089b-96ae-4fc7-a6cd-8d865c0f43de
md"$p(\pmb{v}) = \frac{1}{Z} \sum_{\pmb{h}} e^{-E(\pmb{v},\pmb{h})}$"

# ╔═╡ a9fd23e3-f4d0-4b3b-af36-80953a89431b
md"Fischer and Igel go into more details than Hinton regarding how to compute the gradient of this expression."

# ╔═╡ 01f2b52f-df4a-48cd-90f0-7667f9cff201
md"$\ln{p(\pmb{v})} = \ln\left(\frac{1}{Z}\sum_{\pmb{h}} e^{-E(\pmb{v},\pmb{h})}\right) = \ln\left(\sum_{\pmb{h}} e^{-E(\pmb{v},\pmb{h})}\right) - \ln\left(\sum_{\pmb{v},\pmb{h}} e^{-E(\pmb{v},\pmb{h})}\right)$"

# ╔═╡ 597fe1e4-e59b-4792-9ef9-26c53ef09dfc
md"So the gradient with respect to trainable parameters $\theta=\{W, \pmb{a}, \pmb{b}\}$ then is"

# ╔═╡ b53c7d74-895a-4b86-925f-cc0b60b71b42
md"$-\sum_{\pmb{h}} p(\pmb{h}| \pmb{v})\frac{\partial E(\pmb{v}, \pmb{h})}{\partial \theta} +\sum_{\pmb{v}, \pmb{h}} p(\pmb{v}, \pmb{h})\frac{\partial E(\pmb{v}, \pmb{h})}{\partial \theta}$"

# ╔═╡ 9405e99f-c808-48be-86a2-6448e7575e24
md"where we used $p(h|v) = \frac{p(\pmb{v}, \pmb{h})}{p(\pmb{v})}$. See Fischer and Igelpage 7 for details. The first sum is the expectation values for the gradient under the model distribution and under the conditional distribution of the hidden variables. Evaluating the two sums is not feasible, but we can approximate the expectation values by sampling from the two distributions."

# ╔═╡ 201341eb-7ca7-4f49-abb6-42fea86b6675
md"the derivatives of the energy is given by"

# ╔═╡ 7f4d003b-7cd2-4513-8396-d551d19fbff8
md"$\frac{\partial E(\pmb{v}, \pmb{h})}{\partial θ}: \begin{cases} \frac{\partial E(\pmb{v}, \pmb{h})}{\partial W_{ij}} = v_i h_j\\ \frac{\partial E(\pmb{v}, \pmb{h})}{\partial a_{i}} = v_i\\ \frac{\partial E(\pmb{v}, \pmb{h})}{\partial b_{j}} = h_j \end{cases}$"

# ╔═╡ 46cb0ffa-7fc6-4a77-bd92-b1339eb27a21
md"Hinton uses angle brackets to denote expectation values so in his notation the gradients with respect to $W_{ij}$ would be $\frac{\partial \ln p(\pmb{v})}{\partial W_{ij}} =\langle v_i h_j\rangle_{data} - \langle v_i h_j\rangle_{model}$."

# ╔═╡ 282f9d60-8d8a-40dd-b077-4afe3b3d6313
md"### Performing Inference"

# ╔═╡ e1880f54-cfc9-485e-b219-7849a893a838
md"So to compute a learning signal we need to first compute values for the visible and the hidden units under both the data distribution $\langle\cdot\rangle_{data}$ and the model distribution $\langle\cdot\rangle_{model}$."

# ╔═╡ 2271d6f2-54a4-4f26-a546-8889c227992d
# Fixed inputs: infer hiddens
md"The statistics for the data distribution term is called the positive statistics and the statistics for the model term is called the negative statistics. For the positive phase the input units are clamped at values given by an input datapoint (an image for example). Each hidden unit will then be either in state 0 or state 1 given by the following probability (where $\sigma$ is the sigmoid activation function)"

# ╔═╡ 0cc27c9c-ec58-4ef1-bd58-09954761020b
md"$p(h_j=1|\pmb{v}) = \sigma(b_j + \sum_i v_i W_{ij})$"

# ╔═╡ 74461104-7080-4b58-a159-0b2b6ac5fac5
md"So collecting statistics for the positive phase is easy. In a similar manner we get"

# ╔═╡ 2a36f2df-3f73-4c5d-b154-4a0d1759fcca
md"This is however not what we need to collect for the negative phase. Rather we need $p(\pmb{h},\pmb{v})$ which is less trivial. Hinton states that this can be achieved by initializing the visible units randomly and then performing alternating Gibbs sampling for a very long time. Gibbs sampling is the process of first updating all the hidden units and then updating the visible ones. "

# ╔═╡ 8eb87c5b-c895-43c1-93aa-687275a31c87
md"$p(v_i=1|\pmb{v}) = \sigma(a_i + \sum_j h_j W_{ij})$"

# ╔═╡ 37e880fa-70e7-47d1-b5a8-04cff3f5d828
md"We are finally arriving at the Contrastive divergence algorithm which essentially amounts to cutting the repeated Gibbs sampling short, when collecting the negative statistics. CD-k means contrastive divergence with k iterations of Gibbs sampling. This algorithm is quite fast, but doesn't exactly approximate gradient of the log-probability of the data (which is what we set out to do). To make things even more confusing it is in fact closer to approximating the gradient of a function which itself is called Contrastive Divergence, though a term is omitted (Including this term was found to be both feasible and beneficial by *Du et al* in the paper *Improved Contrastive Divergence Training of Energy-Based Model* from 2021)."

# ╔═╡ 1b0875fd-975c-47f3-a1c1-6278084d77c5


# ╔═╡ 5b982601-9799-41f8-be68-1021c508dc8f
md"- **Comment**: I suppose that for every datapoint one could compute the gradient multiple times (due to the stochastic binary units) to get closer to the expectation value. However, I think that this will be unnecesary when performing mini-batch learning."

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Flux = "~0.12.6"
PlutoUI = "~0.7.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "2e004e61f76874d153979effc832ae53b56c20ee"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.22"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[BFloat16s]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "4af69e205efc343068dc8722b8dfec1ade89254a"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.1.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "DataStructures", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "5e696e37e51b01ae07bd9f700afe6cbd55250bce"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.3.4"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "0902fc7f416c8f1e3b1e014786bb65d0c2241a9b"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "0.8.24"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f53ca8d41e4753c41cdafa6ec5f7ce914b34be54"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.13"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

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
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

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
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "85d2d9e2524da988bffaf2a381864e20d2dae08d"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.2.1"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8c8eac2af06ce35973c3eadb4ab3243076a408e7"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Flux]]
deps = ["AbstractTrees", "Adapt", "ArrayInterface", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "NNlibCUDA", "Pkg", "Printf", "Random", "Reexport", "SHA", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
git-tree-sha1 = "1286e5dd0b4c306108747356a7a5d39a11dc4080"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.12.6"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[Functors]]
deps = ["MacroTools"]
git-tree-sha1 = "4cd9e70bf8fce05114598b663ad79dfe9ae432b3"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.3"

[[GPUArrays]]
deps = ["AbstractFFTs", "Adapt", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "ececbf05f8904c92814bdbd0aafd5540b0bf2e9a"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "7.0.1"

[[GPUCompiler]]
deps = ["DataStructures", "ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "0da0f52fc521ff23b8291e7fda54c61907609f12"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.12.6"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "95215cd0076a150ef46ff7928892bc341864c73c"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.3"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "733abcbdc67337bb6aaf873c6bebbe1e6440a5df"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.1.1"

[[LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b36c0677a0549c7d1dc8719899a4133abbfacf7d"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.6+0"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "d27c8947dab6e3a315f6dcd4d2493ed3ba541791"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.26"

[[NNlibCUDA]]
deps = ["CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "a7de026dc0ff9f47551a16ad9a710da66881b953"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.1.7"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "94bf17e83a0e4b20c8d77f6af8ffe8cc3b386c0a"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Random123]]
deps = ["Libdl", "Random", "RandomNumbers"]
git-tree-sha1 = "0e8b146557ad1c6deb1367655e052276690e71a3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.4.2"

[[RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "a752043df7488ca8bcbe05fa82c831b5e2c67211"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.2"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

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
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "508822dca004bf62e210609148511ad03ce8f1d8"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.0"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "62701892d172a2fa41a1f829f66d2b0db94a9a63"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "885838778bb6f0136f8317757d7803e0d81201e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.9"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "209a8326c4f955e2442c07b56029e88bb48299c7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.12"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "7c53c35547de1c5b9d46a4797cf6d8253807108c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.5"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "c3a5637e27e914a7a445b8d0ad063d701931e9f7"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "8b634fdb4c3c63f2ceaa2559a008da4f405af6b3"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.17"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═03fcb360-421d-441a-bd8c-5372d6bb2be5
# ╠═61ce3068-a319-4f67-b4fd-25745727f0a1
# ╠═8b11badc-2a03-4918-841a-a6459d1aac28
# ╟─ef35700e-8df6-4446-b9f4-2e82bf8801c0
# ╟─9e830b5e-f37f-11eb-083f-277a24c3cd6c
# ╟─1a5a0086-cfba-470f-955f-82df7c5f19de
# ╟─d9e2c1de-ee61-4413-8b41-8bcad7206d1d
# ╠═adb63694-4f58-4d96-84c2-87e3fd69d5ec
# ╟─5a5299e2-4a18-4a52-ae87-453380edc682
# ╟─d33cc5e2-9135-4dd0-b043-67ff5b5edaf6
# ╠═5179313d-576f-4433-82cc-bf2cb7907abd
# ╠═6d210251-d433-43b6-b515-c852ccbc1feb
# ╟─b775575b-33e7-4708-8c6e-4c28f9cfa79f
# ╠═72a1fd39-2980-4f14-9a67-5362f9bb0775
# ╠═a9c6ecab-bc6c-4565-9a29-7d07b95c2de9
# ╠═4f485f1b-33ff-4ea9-a67e-11e5841fcd62
# ╟─7d86146b-dee5-477e-b476-4a1d888831aa
# ╠═8b171005-2f80-492a-bf0a-e85088d498a0
# ╟─3047d526-f2b6-48f3-b5e1-d5290eddb25a
# ╟─85a021d9-eddf-4167-817a-fcee142924ae
# ╟─5d905b9a-6c86-4361-8c84-39cb320c071e
# ╟─b6fc90cf-ca23-49a3-93dc-8bdb9332faec
# ╟─0c28089b-96ae-4fc7-a6cd-8d865c0f43de
# ╟─a9fd23e3-f4d0-4b3b-af36-80953a89431b
# ╟─01f2b52f-df4a-48cd-90f0-7667f9cff201
# ╟─597fe1e4-e59b-4792-9ef9-26c53ef09dfc
# ╟─b53c7d74-895a-4b86-925f-cc0b60b71b42
# ╟─9405e99f-c808-48be-86a2-6448e7575e24
# ╟─201341eb-7ca7-4f49-abb6-42fea86b6675
# ╟─7f4d003b-7cd2-4513-8396-d551d19fbff8
# ╟─46cb0ffa-7fc6-4a77-bd92-b1339eb27a21
# ╟─282f9d60-8d8a-40dd-b077-4afe3b3d6313
# ╟─e1880f54-cfc9-485e-b219-7849a893a838
# ╟─2271d6f2-54a4-4f26-a546-8889c227992d
# ╟─0cc27c9c-ec58-4ef1-bd58-09954761020b
# ╟─74461104-7080-4b58-a159-0b2b6ac5fac5
# ╟─2a36f2df-3f73-4c5d-b154-4a0d1759fcca
# ╟─8eb87c5b-c895-43c1-93aa-687275a31c87
# ╟─37e880fa-70e7-47d1-b5a8-04cff3f5d828
# ╠═1b0875fd-975c-47f3-a1c1-6278084d77c5
# ╟─5b982601-9799-41f8-be68-1021c508dc8f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
