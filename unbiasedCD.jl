### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 03fcb360-421d-441a-bd8c-5372d6bb2be5
using PlutoUI # To enable table of contentss

# ╔═╡ 37c90723-2e16-4e40-a259-74edf117c6e2
using Zygote # For gradient dict

# ╔═╡ 61ce3068-a319-4f67-b4fd-25745727f0a1
using Flux # Julias main Deep Learning library. We don't need autodiff for this project, but Flux is still useful for activation functions and stateful optimizers.

# ╔═╡ 360c13ed-0452-4811-a192-fa21968aae04
using MLDatasets # For loading the FMNIST dataset

# ╔═╡ 118b4c69-d21a-4a35-909a-f1631e83b917
using Plots # For plotting

# ╔═╡ 136b876c-f545-46ac-befd-af7d37ea9d93
using Statistics # For mean

# ╔═╡ a1fb6aad-fb1c-4418-9caa-f7add68bf26e
using LinearAlgebra # For norm

# ╔═╡ f270bf0c-00b6-4308-93bd-2cd0c4dead24
using Random; Random.seed!(3) # for reproducibility

# ╔═╡ 8b11badc-2a03-4918-841a-a6459d1aac28
PlutoUI.TableOfContents(depth = 6)

# ╔═╡ ef35700e-8df6-4446-b9f4-2e82bf8801c0
html"<center><button 
	style='background-color: #4CAF50;  
		border: None ;
		color: white;
		padding: 16px 32px;
		text-align: center;
		text-decoration: none;
		display: inline-block;
		font-size: 20px;
		border-radius: 12px' 
	onclick='present()' >Slideshow</button>"

# ╔═╡ 9e830b5e-f37f-11eb-083f-277a24c3cd6c
md"# Notebook Reimplimentation of Unbiased Contrastive Divergence
The paper *Unbiased Contrastive Divergence Algorithm for Training Energy-Based Latent Variable Models* by Yiuxuan Qiu, Lingsong Zhang and Xiao Wang highlights a known an issue with the Contrastive Divergence algorithm. Namely the fact that it produces biased gradient estimates. Qiu et al proposes a novel and computationally efficient way to solve this issue based on recent advances in MCMC methods.
"

# ╔═╡ 1a5a0086-cfba-470f-955f-82df7c5f19de
md"
- **Note**: This document is a set of interactive notes written to help us understand and re-implement results from the above mentioned paper *Unbiased Contrastive Divergence (...)* by Qiu et al. This is not original content, but just our notes based on the paper and various tutorial as well as the code for our reimplementation.
"

# ╔═╡ d9e2c1de-ee61-4413-8b41-8bcad7206d1d
md"# How to train an RBM with standard CD-k
Geoffrey Hinton's *A Practical Guide to Training Restricted Boltzmann Machines* is a good starting point for understanding Restricted Boltzmann Machines. Asja Fischer's and Cristian Igel's *Training Restricted Boltzmann Machines: An Introduction* is also very useful, and is highlighted in the *Unbiased Contrastive Divergence (...)* paper which we are studying.
"

# ╔═╡ adb63694-4f58-4d96-84c2-87e3fd69d5ec
md"## The RBM energy function
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

# ╔═╡ 6d210251-d433-43b6-b515-c852ccbc1feb
function energy(rbm, v, h, batchsize)
	# TODO: recheck this implementation
	E = sum(rbm.a .* v) - sum(rbm.b .* h) - sum(h*v' .* W)
	return E/batchsize
end

# ╔═╡ b775575b-33e7-4708-8c6e-4c28f9cfa79f
md" We will generate some quick random data to try out our energy function with. We will force the hidden units to be binary variables for now using the Heaviside step function."

# ╔═╡ 72a1fd39-2980-4f14-9a67-5362f9bb0775
heaviside(x) = 0.5 * (sign(x) + 1)

# ╔═╡ a9c6ecab-bc6c-4565-9a29-7d07b95c2de9
function init_rbm(;numvisible=784, numhidden=64, glorot=true)
	if glorot
		# Some initial network parameters
		W = Flux.glorot_normal(numhidden, numvisible)
		a = zeros(numvisible)
		b = zeros(Float32, numhidden)
	else
		# The initialization described in the UCD paper by Qiu on page 8
		W = Flux.rand(Float32, numhidden, numvisible) .-0.5
		a = Flux.zeros(Float32, numvisible)
		b = Flux.zeros(Float32, numhidden)
	end
	return rbmstruct(W, b, a);
end;

# ╔═╡ 3047d526-f2b6-48f3-b5e1-d5290eddb25a
md"## The training objective

"

# ╔═╡ 85a021d9-eddf-4167-817a-fcee142924ae
md"Similar to in statistical physics the probability of the system being in a certain state is given by the Boltzmann distribution. This part is explained well in Hinton's notes."

# ╔═╡ 5d905b9a-6c86-4361-8c84-39cb320c071e
md"$p(\pmb{v},\pmb{h}) = \frac{e^{-E(\pmb{v},\pmb{h})}}{Z}$"

# ╔═╡ b6fc90cf-ca23-49a3-93dc-8bdb9332faec
md"where $Z$ is given by $Z = \sum_{\pmb{v},\pmb{h}} e^{-E(\pmb{v},\pmb{h})}$. The sum here runs over all possible state pars $(\pmb{v},\pmb{h})$. 
This factor seems intractable for continuous variables, but tractable (yet impractical for moderately sized $\pmb{v}$ and $\pmb{h}$) for binary variables. The probability corresponding to a visible vector $\pmb{v}$ is"

# ╔═╡ 0c28089b-96ae-4fc7-a6cd-8d865c0f43de
md"$p(\pmb{v}) = \frac{1}{Z} \sum_{\pmb{h}} e^{-E(\pmb{v},\pmb{h})}$"

# ╔═╡ a9fd23e3-f4d0-4b3b-af36-80953a89431b
md"Fischer and Igel go into more details than Hinton regarding how to compute the gradient of this expression."

# ╔═╡ 01f2b52f-df4a-48cd-90f0-7667f9cff201
md"$\ln{p(\pmb{v})} = \ln\left(\frac{1}{Z}\sum_{\pmb{h}} e^{-E(\pmb{v},\pmb{h})}\right) = \ln\left(\sum_{\pmb{h}} e^{-E(\pmb{v},\pmb{h})}\right) - \ln\left(\sum_{\pmb{v}',\pmb{h}} e^{-E(\pmb{v}',\pmb{h})}\right)$"

# ╔═╡ 597fe1e4-e59b-4792-9ef9-26c53ef09dfc
md"So the gradient with respect to trainable parameters $\theta=\{W, \pmb{a}, \pmb{b}\}$ then is"

# ╔═╡ b53c7d74-895a-4b86-925f-cc0b60b71b42
md"$\frac{\partial \ln p(\pmb{v})}{\partial \theta} = -\sum_{\pmb{h}} p(\pmb{h}| \pmb{v})\frac{\partial E(\pmb{v}, \pmb{h})}{\partial \theta} +\sum_{\pmb{v}', \pmb{h}} p(\pmb{v}', \pmb{h})\frac{\partial E(\pmb{v}', \pmb{h})}{\partial \theta}$"

# ╔═╡ 9405e99f-c808-48be-86a2-6448e7575e24
md"where we used $p(h|v) = \frac{p(\pmb{v}, \pmb{h})}{p(\pmb{v})}$. See Fischer and Igel page 7 for details. The first sum is the expectation values for the gradient of the energy under the data distribution and the second term is the expectation of the gradient of the energy under the model distribution. We can get a good estimate of the first term by averaging the gradient of E produced by a bunch of different datapoints, but the second point is much more tricky!"

# ╔═╡ 201341eb-7ca7-4f49-abb6-42fea86b6675
md"the derivatives of the energy is given by"

# ╔═╡ 7f4d003b-7cd2-4513-8396-d551d19fbff8
md"$\frac{\partial E(\pmb{v}, \pmb{h})}{\partial θ}: \begin{cases} \frac{\partial E(\pmb{v}, \pmb{h})}{\partial W_{ij}} = v_i h_j\\ \frac{\partial E(\pmb{v}, \pmb{h})}{\partial a_{i}} = v_i\\ \frac{\partial E(\pmb{v}, \pmb{h})}{\partial b_{j}} = h_j \end{cases}$"

# ╔═╡ 46cb0ffa-7fc6-4a77-bd92-b1339eb27a21
md"Hinton uses angle brackets to denote expectation values so in his notation the gradients with respect to $W_{ij}$ would be $\frac{\partial \ln p(\pmb{v})}{\partial W_{ij}} =\langle v_i h_j\rangle_{data} - \langle v_i h_j\rangle_{model}$."

# ╔═╡ 64dde393-ac97-4140-a69f-549bd7f7ce85
function ∇E(v,h, batchsize)
	∂E∂W = h*v'/batchsize
	∂E∂a = sum(v, dims=2)[:]/batchsize
	∂E∂b = sum(h, dims=2)[:]/batchsize
	
	#∂E∂W = Flux.σ.(rbm.W'*h .+ rbm.a)*h'/batchsize
	#∂E∂a = sum(Flux.σ.(rbm.W'*h .+ rbm.a), dims=2)[:]/batchsize
	#∂E∂b = sum(h, dims=2)[:]/batchsize
	return [∂E∂W, ∂E∂a, ∂E∂b]
end

# ╔═╡ 282f9d60-8d8a-40dd-b077-4afe3b3d6313
md"## Performing Inference"

# ╔═╡ e1880f54-cfc9-485e-b219-7849a893a838
md"So to compute a learning signal we need to first compute values for the visible and the hidden units under both the data distribution $\langle\cdot\rangle_{data}$ and the model distribution $\langle\cdot\rangle_{model}$."

# ╔═╡ 2271d6f2-54a4-4f26-a546-8889c227992d
# Fixed inputs: infer hiddens
md"The statistics for the data distribution term is called the positive statistics and the statistics for the model term is called the negative statistics. For the positive phase the input units are clamped at values given by an input datapoint (an image for example). Each hidden unit will then be either in state 0 or state 1 given by the following probability (where $\sigma$ is the sigmoid activation function)"

# ╔═╡ 0cc27c9c-ec58-4ef1-bd58-09954761020b
md"$p(h_j=1) = \sigma(b_j + \sum_i v_i W_{ij})$"

# ╔═╡ 74461104-7080-4b58-a159-0b2b6ac5fac5
md"So collecting statistics for the positive phase is easy. In a similar manner we get"

# ╔═╡ 2a36f2df-3f73-4c5d-b154-4a0d1759fcca
md"This is however not what we need to collect for the negative phase. Rather we need $p(\pmb{h},\pmb{v})$ which is less trivial. Hinton states that this can be achieved by initializing the visible units randomly and then performing alternating Gibbs sampling for a very long time. Gibbs sampling is the process of first updating all the hidden units and then updating the visible ones. "

# ╔═╡ 8eb87c5b-c895-43c1-93aa-687275a31c87
md"$p(v_i=1) = \sigma(a_i + \sum_j h_j W_{ij})$"

# ╔═╡ 37e880fa-70e7-47d1-b5a8-04cff3f5d828
md"We are finally arriving at the Contrastive divergence algorithm which essentially amounts to cutting the repeated Gibbs sampling short, when collecting the negative statistics. CD-k means contrastive divergence with k iterations of Gibbs sampling. This algorithm is quite fast, but doesn't exactly approximate gradient of the log-probability of the data (which is what we set out to do). To make things even more confusing it is in fact closer to approximating the gradient of a function which itself is called Contrastive Divergence, though a term is omitted (Including this term was found to be both feasible and beneficial by *Du et al* in the paper *Improved Contrastive Divergence Training of Energy-Based Model* from 2021)."

# ╔═╡ 94fe813e-8047-4372-981d-f1759442975b
md"So we need two functions. One for infering hidden units in the positive phase and one for infering hidden and visible units in the negative phase."

# ╔═╡ 93ca8efa-1bc5-473c-83cc-5994af633659
function inference_pos!(rbm, v, h)
	# Infer the hidden units given fixed visible units
	r = rand(Float32, size(h))
	p = Flux.σ.(rbm.W * v .+ rbm.b)
	h .= p .> r
	# h = Flux.σ.(rbm.W * v .+ rbm.b)
	return h
end

# ╔═╡ 91d233f2-12d3-4594-b68e-5a6b3d8e633f
function inference_neg!(rbm, v, h, k)
	h = Flux.σ.(rbm.W * v .+ rbm.b)
	for i=1:k
		# Infer hidden units given the current visible units
		r = rand(Float32, size(h))
		p = Flux.σ.(rbm.W * v .+ rbm.b)
		h .= p .> r
		# Infer visible units given the current hidden units
		r = rand(Float32, size(v))
		p = Flux.σ.(rbm.W' * h .+ rbm.a)
		v .= p .> r
	end
	# h = Flux.σ.(rbm.W * v .+ rbm.b)
	return v, h
end

# ╔═╡ 1b0875fd-975c-47f3-a1c1-6278084d77c5
md"## Loading the FMNIST dataset
So now we have an initialized RBM, a function for computing the Energy (although we don't really need it) a function for computing gradients and finally functions for computing states of the hidden and visible units in the positive and negative phases. At this point it is probably time to get some real data and start training!"

# ╔═╡ 343ad144-8b26-4c6c-8f3d-4cf042c46cf0
function FMNISTdataloader(batchsize)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.FashionMNIST.traindata(Float32)
    xtest, ytest = MLDatasets.FashionMNIST.testdata(Float32)
	
    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

	# One-hot-encode the labels
    ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9)

    # Create DataLoaders (mini-batch iterators)
    trainloader = Flux.DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true, partial=false)
    testloader = Flux.DataLoader((xtest, ytest), batchsize=batchsize, partial=false)

    return trainloader, testloader
end

# ╔═╡ e8ac304a-84dd-4ed9-80a3-c0786c734f14
md"We will use the FMNISTloader function to load the FMNIST data set. Let's have a look at what this dataset looks like! The images are stored as columns in a matrix, so the function will take a vector as input and than reshape and rotate."

# ╔═╡ 31a4f5c7-9d7b-47a8-a27a-3a421fc0f10f
function imshow(imgvec; w=28, h=28) 
	heatmap(rotr90(reshape(imgvec, (w, h)), 3), c=:grays, colorbar=false, xticks=false, yticks=false, border=:none, aspect_ratio=:equal, size=(100,100))
end

# ╔═╡ 7906a124-67ea-42ae-8f27-80eaba3e3368
trainloader, testloader = FMNISTdataloader(128);

# ╔═╡ 7a782a84-3615-4ad7-b6bb-a500966cb5ac
#= Extrect a number of columns, pass them to the imshow function 
and store the produced heatmaps in an array.=#
H = [imshow(view(trainloader.data[1], : ,i)) for i=1:8]

# ╔═╡ a6a0df67-885d-4799-b3cb-864f09f629a7
md"## Training the RBM
"

# ╔═╡ 0c893210-9bd2-43b8-ab46-d9579707eed2
function reconstruction_loss(rbm , xtest)
	
	vin = xtest
	
	r = rand(Float32, size(rbm.b, size(xtest)[2]))
	h = Flux.σ.(rbm.W * vin .+ rbm.b)
	h = h .> r	
	
	#r = rand(Float32, size(rbm.a))
	vout = Flux.σ.(rbm.W' * h .+ rbm.a) # transpose W?
	#vout = vout .> r
	
	loss = Flux.Losses.mse(vin, vout)
	return loss
end

# ╔═╡ 6b10fdc8-e871-4de1-b2cb-e81c610823e3
function train(rbm; numepochs=5, batchsize=64, k=3)

	trainloader, testloader = FMNISTdataloader(batchsize)
	# Choose optimizer
	η = 0.05; optimizer = Descent(η) # SGD
	# η = 0.02; optimizer = Momentum(η)
	# η = 0.0003; optimizer = ADAM(η)
	
	# We use Zygotes graddict in order to use Flux's optimizers
	θ = Flux.params(rbm.W, rbm.a, rbm.b)
    ∇θ = Zygote.Grads(IdDict(), θ)
	
	recloss = zeros(Float32, numepochs)
	
	# Arrays for storing the variables
	numvisible, numhidden = length(rbm.a), length(rbm.b)
	hpos = zeros(Float32, (numhidden, batchsize))
	vpos = zeros(Float32, (numvisible, batchsize))
	hneg = zeros(Float32, (numhidden, batchsize))
	vneg = zeros(Float32, (numvisible, batchsize))
	
	for epoch=1:numepochs
		t1 = time()
		for (x,y) in trainloader
			vpos = x
			hpos = inference_pos!(rbm, vpos, hpos)

			# Randomly initialize the inputs and perform k Gibbs sampling steps
			vneg = deepcopy(vpos)#heaviside.(rand(Float32, (numvisible, batchsize)) .- 0.5)
			vneg, hneg = inference_neg!(rbm, vneg, hneg, k)
			
			# Compute gradient terms
			∇pos = ∇E(vpos, hpos, batchsize)
			∇neg = ∇E(vneg, hneg, batchsize)
			
			for i=1:3
			∇θ.grads[θ[i]] = -∇pos[i] + ∇neg[i]
			end
			
			Flux.Optimise.update!(optimizer, θ, ∇θ)

		end
		
		recloss[epoch] = reconstruction_loss(rbm , testloader.data[1])
		t2 = time()
		# println output printed to console
		println("Epoch: ", epoch, "/", numepochs, 
				":\t recloss = ", round(recloss[epoch], digits=5),
				"\t runtime: ", round(t2-t1, digits=2), " s")

	end
	return recloss
end

# ╔═╡ 5690fc6d-d3b4-478c-8efe-cd2c03a915af
md"We can now initialize an RBM with random weights and train it!."

# ╔═╡ 944f0cf9-8302-41f4-9b9d-f90523827bac
# Initialize the network
#=begin
	println("\nTraining RBM") # printed to console!
	rbm = init_rbm(numvisible=784, numhidden=64)
	recloss = train(rbm, numepochs=5, batchsize=64, k=3);
end=#

# ╔═╡ 711787c1-f8fc-4fac-92c2-21a01ab4937d
md"## Visualizing the filters"

# ╔═╡ 09852337-608d-4ef4-819d-74437bf978bc
md"Here we visualize the first 64 filters."

# ╔═╡ 6cc91180-c85f-4e46-93bb-668234023328
filters = [imshow(rbm.W'[:,i]) for i=1:64];

# ╔═╡ f0c0bf3b-3329-4619-ba86-366b7abe3c79
plot(filters..., layout=(8, 8), size=(2000, 2000))

# ╔═╡ 0ca12440-3025-48ff-9aa7-aed2ed01d9f6
md"## Reconstruction"

# ╔═╡ 816e0fe7-add7-41e9-8a8c-41d67c44eec8
function reconstruct(rbm, batchsize)

	vin = testloader.data[1][:,1:batchsize];
	
	r = rand(Float32, size(rbm.b, batchsize))
	h = Flux.σ.(rbm.W * vin .+ rbm.b)
	h = h .> r	
	
	#r = rand(Float32, size(rbm.a))
	vout = Flux.σ.(rbm.W' * h .+ rbm.a) # transpose W?
	#vout = vout .> r
	
	return vin, vout
end

# ╔═╡ 9cef19bc-5295-456d-b6fe-a7cb1099fa6f
x, xrec = reconstruct(rbm, 64);

# ╔═╡ 1098fb24-b08a-4598-b44d-8f356877af25
img_orig = [imshow(x[:,i]) for i=1:64]

# ╔═╡ 004f1d73-b909-47da-b25d-aa83787520e9
img_rec = [imshow(xrec[:,i]) for i=1:64]

# ╔═╡ 79a8e531-6c29-4cf5-8429-0d7474b01f29
md"# How to train an RBM with UCD
"

# ╔═╡ a1afe50c-3d5c-4c50-99dd-73aef979e24a
md"## The basic idea"

# ╔═╡ 3794e224-9bba-481d-b072-4abde652c627
md" Here we will use the notation used by Qiu et al (the authors of the *Ubiased Contrastive Divergence (...)* paper). What we want to compute is"

# ╔═╡ 55cee522-7a7d-409e-bcee-5dd6f41c701f
md"
```math
	\begin{equation}
		\frac{\partial l(\theta; \mathcal{D})}{\partial \theta} = 
		-n \left[
		\mathbb{E}(\pmb{v}, \pmb{h})
		\tilde{}p(\pmb{h}|\pmb{v};\theta)\left(\frac{\partial E(\pmb{v}, \pmb{h}; 			\theta)}{\partialθ}\right) 
		- \mathbb{E}(\pmb{v}, \pmb{h})
		\tilde{}p(\pmb{h},\pmb{v};\theta)\left(\frac{\partial E(\pmb{v}, \pmb{h}; 			\theta)}{\partialθ}\right) 
		\right].
	\end{equation}
```
"

# ╔═╡ 6a914ea6-349a-43b9-8152-6e4452786646
md"The tricky part here is to estimate the second term. Typically this has been done with CD-k as mentioned above. Following Qiu's notation we have that $\pmb{x}=(\pmb{v}, \pmb{h})\in\mathbb{X}:=\mathbb{V}\times\mathbb{H}$ 
and $f(\pmb{x}, \theta)=\partial E(\pmb{v},\pmb{h}; \theta)/\partial\theta$, 
which lets us restate the problem as"

# ╔═╡ a039ce76-e5e8-4d22-a23c-c1d03849ada3
md"
```math
	\begin{equation}
		\frac{\partial l(\theta; \mathcal{D})}{\partial \theta} = 
		-n \left[
		\mathbb{E}_\mathcal{D}\{f(\pmb{x},\theta)\}
		- \mathbb{E}_\mathcal{M}\{f(\pmb{x},\theta)\}
		\right].
	\end{equation}
```
"

# ╔═╡ 8b7f2e1c-d21d-4d51-83cc-6a118c34586d
md"As noted before the CD-k algorithm is inherintly biased unless $k\rightarrow\infty$, which is unfeasible. Recall that the $k$ denotes the number of iterations of Gibbs sampling used to find $v$ and $h$ when estimating the model distribution term of the gradient. The different second term can be computed as a Markov chain (via repeated Gibbs sampling). This gives
	$\lim_{t\rightarrow\infty}\mathbb{E}_\mathcal{M}\{f(\pmb{\xi_t},\theta)\}
	= \mathbb{E}_\mathcal{M}\{f(\pmb{x},\theta)\}$.
The CD-k approach truncates the Markov chain at k iterations, which results in a biased estimate of the gradient. 
"

# ╔═╡ 8f64ebad-d946-4f90-bbad-c2923fad1c65
md"## How to get the chain \{$\eta_t\}$"

# ╔═╡ c05d48e3-4b20-4c52-a437-b376ed8c5756
md"The authors state that the second term can be expressed as a telescoping sum under some mild conditions (TODO: check this in the påaper's appendix)."

# ╔═╡ 2588712f-6a5e-4fd1-9b2c-b6c7a20e9199
md"
```math
	\begin{equation}
		\mathbb{E}_\mathcal{M}\{f(\pmb{x})\} 
		= \mathbb{E}\{f(\pmb{\xi}_k)\} 
		+ \sum_{t=k+1}^\infty 
		\left[
		\mathbb{E}\{f(\pmb{\xi}_t)\}
		- \mathbb{E}\{f(\pmb{\xi}_{t-1})\}
		\right]
	\end{equation}
```
"

# ╔═╡ 928fd624-c571-43f4-8bf7-346ec51deeb9
md"They then assume that for any $k\geq 0$ there exists another Markov chain $\{\eta_t\}$, ulfilling the following conditions: \{\eta_t\} and \{\xi_t\} follow the same marginal distribution and $\xi_t=\eta_{t-1}$ for all $t\geq\tau$ for some random time $\tau$. They then flip the order of summation and expectation which yields the followingresult (TODO: check appendix in the paper for detailed derivations)."

# ╔═╡ db748b83-87d9-4354-9a74-b15424119d64
md"
```math
	\begin{equation}
		\mathbb{E}_\mathcal{M}\{f(\pmb{x})\} 
		= \mathbb{E}\left[
		f(\xi_k)
		+ \sum_{t=k+1}^\infty \{f(\xi_t) - f(\eta_{t-1})\}
		\right]		
		= \mathbb{E}\left[
		f(\xi_k)
		+ \sum_{t=k+1}^{\tau - 1} \{f(\xi_t) - f(\eta_{t-1})\}
		\right]
	\end{equation}
```
"

# ╔═╡ c8c8c68a-826e-4877-88a5-59866f422d40
md"In the paper they note that one is free to choose k as one pleases, but that they chose k=1."

# ╔═╡ 77b4d504-7eab-49e8-ab5e-b576250c5411
md" ## UCD RBM implementation"

# ╔═╡ 243c9723-6e33-4e02-baaa-857d4f4cd344
md"- **Note**: Qiu et al used 1000 parallel Markov chains in their experiments in order to get a better gradient estimate. This makes sense, but might not be necessary for a proof of concept implementation."

# ╔═╡ e97a9dde-5e0f-48ae-a72d-56cea3410ad9


# ╔═╡ 0593e773-d877-4c8c-ae7f-826cad5cf75a
function coupled_inference(rbm, vₜ, hₜ, vₜ₋₁´, hₜ₋₁´, # Input variables
						   vₜ₊₁, hₜ₊₁, vₜ´, hₜ´, # Output variables
						   maxtries)
	# Following algorithm 3 in UCD paper
	# Input: ϵₜ = (vₜ, hₜ) and ηₜ₋₁ = (vₜ₋₁´, hₜ₋₁´)
	# Output: ϵₜ₊₁ = (vₜ₊₁, hₜ₊₁) and ηₜ = (vₜ´, hₜ´)
	
	U1 = rand(); Z1 = rand(Float32, size(vₜ)[1])
	p_vₜ₊₁ = Flux.σ.(rbm.W' * hₜ .+ rbm.a)
	vₜ₊₁ = p_vₜ₊₁ .>= Z1
	p_vₜ´ = Flux.σ.(rbm.W' * hₜ .+ rbm.a)
	p_vₜ₋₁´ = Flux.σ.(rbm.W' * hₜ₋₁´ .+ rbm.a)

	lnT1 = transition_density(p_vₜ´, vₜ₊₁)
	lnT2 = transition_density(p_vₜ₊₁, vₜ₊₁)
	
	# Check if chains meet up
	if exp(U1) <= lnT1 - lnT2
		vₜ´ = vₜ₊₁
	# Otherwise repeatedly sample
	else
		accept_vₜ₊₁ = false
		accept_vₜ´ = false
		for i=1:maxtries
			U2 = rand(); U2´ = rand(); Z2 = rand(Float32, size(vₜ)[1])
			# Propose vₜ₊₁
			if accept_vₜ₊₁ == false
				vₜ₊₁ .= p_vₜ₊₁ .>= Z2
				T1 = transition_density(p_vₜ₊₁, vₜ₊₁)
				T2 = transition_density(p_vₜ´, vₜ₊₁)
				accept_vₜ₊₁ = U2 > exp(T1 - T2)
			end
			# Propose vₜ´
			if accept_vₜ´ == false
				vₜ .= p_vₜ´ .>= Z2
				T1 = transition_density(p_vₜ´, vₜ´)
				T2 = transition_density(p_vₜ₋₁´, vₜ´)
				accept_vₜ´ = U2´ > exp(T1 - T2)
			end
		end # end sampling
	end
	
	Z3 = rand(Float32, size(hₜ)[1])
	hₜ₊₁ .= (Flux.σ.(rbm.W * vₜ₊₁ .+ rbm.b) .>= Z3)
	hₜ´ .= (Flux.σ.(rbm.W * vₜ´.+ rbm.b) .>= Z3)
	
	return vₜ₊₁, hₜ₊₁, vₜ´, hₜ´
end

# ╔═╡ c0944ff3-c13a-484d-90ed-8d29bb52bd31
function trainUCD(rbm; numepochs=5, batchsize=64, k=3, tmax, maxtries)

	trainloader, testloader = FMNISTdataloader(batchsize)
	# Choose optimizer
	η = 0.05; optimizer = Descent(η) # SGD
	# η = 0.02; optimizer = Momentum(η)
	# η = 0.0003; optimizer = ADAM(η)
	
	# We use Zygotes graddict in order to use Flux's optimizers
	θ = Flux.params(rbm.W, rbm.a, rbm.b)
    ∇θ = Zygote.Grads(IdDict(), θ)
	
	recloss = zeros(Float32, numepochs)
	tmean = zeros(Float32, numepochs)
	
	# Arrays for storing the variables
	numvisible, numhidden = length(rbm.a), length(rbm.b)
	hpos = zeros(Float32, (numhidden, batchsize))
	vpos = zeros(Float32, (numvisible, batchsize))
	hξₖ = zeros(Float32, (numhidden, batchsize))
	vξₖ = zeros(Float32, (numvisible, batchsize))
	hξₜ = zeros(Float32, (numhidden, batchsize))
	vξₜ = zeros(Float32, (numvisible, batchsize))
	hηₜ₋₁ = zeros(Float32, (numhidden, batchsize))
	vηₜ₋₁ = zeros(Float32, (numvisible, batchsize))
	
	for epoch=1:numepochs
		t1 = time()
		for (x,y) in trainloader
			# The activations for the first term are easy to compute
			vpos = x
			hpos = inference_pos!(rbm, vpos, hpos)

			#= The activations for the second term arer more difficult
			There are three parts:
			(1) The ξₖ term, which we typycally fix to use k=1. 
			We find this by running CD-K.
			(2) The ξₜ terms and (3) the ηₜ₋₁ terms, which really are the tricky parts 
			=#
			
			# (1) get ξₖ term via CD-k
			vξₖ = deepcopy(vpos)
			vξₖ, hξₖ = inference_neg!(rbm, vξₖ, hξₖ, k)
			
			∇combined = -∇E(vpos, hpos, batchsize) .+ ∇E(vξₖ, hξₖ, batchsize)
			# ∇pos = -∇E(vpos, hpos, batchsize)
			# ∇neg1 = ∇E(vξₖ, hξₖ, batchsize)	
			# ∇combined = [-∇pos[i] + ∇neg1[i] for i=1:3]
			
			# (2) get ξₜ terms and (3) the ηₜ₋₁ terms
			# Is this an appropriate initialization of vξₜ, vηₜ₋₁?
			vξₜ, vηₜ₋₁ = deepcopy(vξₖ), deepcopy(vξₖ) 
			
			for t=1:tmax
				vξₜ, hξₜ, vηₜ₋₁, hηₜ₋₁ = coupled_inference(rbm, 
															vξₖ, hξₖ, vηₜ₋₁, hηₜ₋₁, 															vξₜ, hξₜ, vηₜ₋₁, hηₜ₋₁,																maxtries,)
				# Check if the chains have met
				if (vξₜ==vηₜ₋₁ && hξₜ==hηₜ₋₁) || t==tmax
					tmean[epoch] += t
					break
				else
					#= Add contributions to the gradient estimate
					if the chain has not converged yet. =#
					∇ηₜ₋₁ = -∇E(vηₜ₋₁, hηₜ₋₁, batchsize) 
					∇ξₜ = ∇E(vξₜ, hξₜ, batchsize)
					∇combined .+= -∇ηₜ₋₁ .+ ∇ξₜ #∇E(vηₜ₋₁, hηₜ₋₁, batchsize) .+ ∇E(vξₜ, hξₜ, batchsize) 
				end		
			end
			
			hξₜ = Flux.σ.(rbm.W * vξₜ .+ rbm.b)
			hηₜ₋₁ = Flux.σ.(rbm.W * vηₜ₋₁ .+ rbm.b)
			
			# Compute gradient terms
			for i=1:3
				∇θ.grads[θ[i]] = ∇combined[i]
			end
			Flux.Optimise.update!(optimizer, θ, ∇θ)

		end
		
		tmean[epoch] /= round(trainloader.nobs/trainloader.batchsize - 0.5)
		recloss[epoch] = reconstruction_loss(rbm , testloader.data[1])
		t2 = time()
		# println output printed to console
		println("Epoch: ", epoch, "/", numepochs, 
				":\t recloss = ", round(recloss[epoch], digits=5),
				"\t runtime: ", round(t2-t1, digits=2), " s")

	end
	return recloss, tmean
end

# ╔═╡ 431a65ad-152f-4686-bfe1-cf856056dffd
# Initialize the network
#=begin
	println("\nTraining RBM") # printed to console!
	rbmUCD = init_rbm(numvisible=784, numhidden=16)
	(reclossUCD, tmean) = trainUCD(rbmUCD, numepochs=2, batchsize=64, k=1, tmax=10, maxtries=10);=#
end

# ╔═╡ b85a8965-2215-4742-91b5-a60dd4dce499
tmean

# ╔═╡ 643f9028-1f95-4c0c-b842-7cdbb1603914
md"### Visualize the UCD trained RBM's filters and reconstructions
For fun we use sliders to select how many filters and datapoints to visualize.
"

# ╔═╡ d8152e9d-0846-4f20-9242-3f27d80f3243
@bind numfilters Slider(1:length(rbmUCD.b); default=8, show_value=true)

# ╔═╡ e675cc20-01b0-44eb-8307-5f12fa8e47d3
md" $(numfilters) filter(s) selected!"

# ╔═╡ e24aa7ea-211b-4f15-bc17-359478b02cb7
filtersUCD = [imshow(rbmUCD.W'[:,i]) for i=1:numfilters]

# ╔═╡ 248e57a5-a446-4489-8ef6-a840f08d55b9
@bind numreconstructions Slider(1:length(rbmUCD.b); default=8, show_value=true)

# ╔═╡ a64f08d0-c7c6-4575-af8b-40dbdce4cf5d
md" $(numreconstructions) datapoint(s) selected!"

# ╔═╡ 8b5722e3-5a16-4806-b859-6cacb9ee5e13
xUCD, xrecUCD = reconstruct(rbmUCD, numreconstructions);

# ╔═╡ 5cf76965-fabe-4a2e-beab-6abf3a216e90
img_origUCD = [imshow(xUCD[:,i]) for i=1:numreconstructions]

# ╔═╡ 6c96b07d-2f21-45b8-880a-28ae41da3b4b
img_recUCD = [imshow(xrecUCD[:,i]) for i=1:numreconstructions]

# ╔═╡ 277ea081-5088-49c3-b1d9-1ac6272b6d22
md"# Comparing UCD and CD-k"

# ╔═╡ 8b8e35ba-b332-4afc-a8c5-865d73775fa9
md"
- Select which experiments to reproduce
- Reproduce them 
- Stack RBMs"

# ╔═╡ 42aa9749-c4da-4aa9-a89d-c9323bf837ec
md"Even though fMNIST is a small dataset it is still unfeasible to compute $p(\pmb{v}) = \frac{1}{Z} \sum_h e^{-E(\pmb{V}, \pmb{h})}$ for this dataset in practice, so we will work with the much smaller $4\times 4$ *Bars and Stripes* dataset."

# ╔═╡ 2a7c2484-9928-4432-9450-f2823d3a75eb
md"""## Bars and Stripes dataset

The $4\times 4$ Bars and stripes dataset can be generated in a few lines of code. There are 16 possible configurations of stripes examples and the same number of bars examples. Two examples belong to both classes (bars in all/no colums is the same as stripes in all/no columns), so there are 30 unique examples in total. Qiu et al mistakenly state that the dataset contains 36 examples, but it really is 30 (or 32 if you count the duplicate samples mentioned before.
"""

# ╔═╡ d033638d-f0f6-46db-80bf-648b39a52f33
begin
	# Write out the possible masks and generate the bars and stripes examples.
	mask4 = [1 1 1 1]
	mask3 = [1 1 1 0; 1 1 0 1; 1 0 1 1; 0 1 1 1]
	mask2 = [1 1 0 0; 0 1 1 0; 0 0 1 1; 1 0 1 0; 0 1 0 1; 1 0 0 1]
	mask1 = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
	mask0 = [0 0 0 0]
	mask = vcat(mask4, mask3, mask2, mask1, mask0)
	bars = [ones(Float32, 4, 4) .* mask[i,:] for i=1:size(mask, 1)]
	stripes = [ones(Float32, 4, 4) .* mask[i,:]' for i=1:size(mask, 1)]
	
	# Populate columns of BAS matrix 
	BAS = zeros(Float32, 16, 30)
	# With stripes examples
	for i=1:16
		BAS[:, i] = stripes[i][:]
	end
	# With bars examples, but omit degenerate cases
	for i=1:14
		BAS[:, 16+i] = bars[i+1][:]
	end
end;

# ╔═╡ c2e9f61c-2316-4f45-8709-97b9268b0795
img_BAS =  [imshow(BAS[:,i], w=4, h=4) for i=1:30]

# ╔═╡ 5844b476-5827-4ba9-bd70-370326cd71ad
function get_logpv(rbm, v)
	#=
	p(v) = f/Z	
	Z= Σᵥₕe⁻ᴱ⁽ᵛʰ⁾Egelund
	f = Σₕe⁻ᴱ⁽ᵛʰ⁾
	
	⇒ log(p(v)) = log(f) - log(Z)
				  = log(Σₕe⁻ᴱ⁽ᵛʰ⁾) - log(Σᵥₕe⁻ᴱ⁽ᵛʰ⁾)

	Is the following useful?
	log(x₁ + x₂ + x₃) + ... = LogSumExp(log(x₁), log(x₂), log(x₃))
	⇒ log(Σₕe⁻ᴱ⁽ᵛʰ⁾) = LogSumExp(log(exp(-E(v,h₁))), log(exp(-E(v,h₂))), ...)
	=#
	batchsize = size(v, 2)
	wv_plus_b = rbm.W*v .+ rbm.b
	
	# Compute the logarithm of the partition function Z
	# TODO: Check this part again!
	logZ_T1 = sum(v .* rbm.a, dims=1)
	logZ_T2 = sum(log.(1 .+ exp.(wv_plus_b)), dims=1)
	logZ = Flux.logsumexp(logZ_T1 .+ logZ_T2)*batchsize
	
	T1 = sum(rbm.a .* v) 
	T2 = sum(log.(1 .+ exp.(wv_plus_b)))
	f = T1 + T2
	
	logpv = f - logZ
	return logpv
end 

# ╔═╡ a7b081dc-ab39-4ffa-a393-8cb7ce51c076
function lnTᵥ(rbm, v, h)
	pv = Flux.σ.(rbm.W'*h .+ rbm.a)
	lnT = sum(log.(pv).*v .+ log.(1 .- pv).*(1 .- v))
	return lnT
end

# ╔═╡ b87a54bd-3ff9-4fc6-a898-54029658a0b7
"""Following algorithm 3 in Qiu's UCD paper"""
function coupled_inference_2(rbm, vₜ, hₜ, vₜ₋₁´, hₜ₋₁´, # Input variables
						   maxtries)	
	U1 = rand()
	Z1 = rand(Float32, size(rbm.a))
	vₜ₊₁ = Flux.σ.(rbm.W'*hₜ .+ rbm.a) .>= Z1
	
	#=hₜ = Flux.σ.(rbm.W*vₜ₊₁ .+ rbm.b) .>= rand(Float32, size(rbm.b))
	if norm(vₜ - vₜ₋₁´) == 0 && norm(hₜ -  hₜ₋₁´) == 0
		vₜ´ = vₜ₊₁
		hₜ´ = hₜ
		return vₜ₊₁, hₜ₊₁, vₜ´, hₜ´
	end=#
	
	# Check if chains meet up
	if log(U1) <= lnTᵥ(rbm, vₜ₊₁, hₜ₋₁´) - lnTᵥ(rbm, vₜ₊₁, hₜ)  # correct?
		vₜ´ = vₜ₊₁
		
	# Otherwise repeatedly sample
	else
		accept_vₜ₊₁ = false
		accept_vₜ´ = false
		for i=1:maxtries
			U2 = rand(); U2´ = rand(); Z2 = rand(Float32, size(vₜ)[1])
			
			# Propose vₜ₊₁
			if accept_vₜ₊₁ == false
				vₜ₊₁ = Flux.σ.(rbm.W'*hₜ .+ rbm.a) .>= Z2
				accept_vₜ₊₁ = log(U2) > lnTᵥ(rbm, vₜ₊₁, hₜ₋₁´) - lnTᵥ(rbm, vₜ₊₁, hₜ)
			end
			
			# Propose vₜ´
			if accept_vₜ´ == false
				vₜ´ = Flux.σ.(rbm.W'*hₜ₋₁´ .+ rbm.a) .>= Z2
				accept_vₜ´ = log(U2´) > lnTᵥ(rbm, vₜ´, hₜ) - lnTᵥ(rbm, vₜ´, hₜ₋₁´)
			end
			
			if accept_vₜ₊₁==true && accept_vₜ´==true
				break
			end
			
		end # end of for i=1:maxtries		
	end
	
	Z3 = rand(Float32, size(rbm.b))
	hₜ₊₁ = (Flux.σ.(rbm.W * vₜ₊₁ .+ rbm.b) .>= Z3)
	hₜ´ = (Flux.σ.(rbm.W * vₜ´.+ rbm.b) .>= Z3)
	
	return vₜ₊₁, hₜ₊₁, vₜ´, hₜ´
end

# ╔═╡ 535c8a18-b6f3-4a14-b4a4-4c965b00c85d
function reconstruction_loss2(rbm , xtest)
	
	vin = xtest
	
	r = rand(Float32, size(rbm.b, size(xtest)[2]))
	h = Flux.σ.(rbm.W * vin .+ rbm.b)
	h = h .> r	
	
	# r = rand(Float32, size(rbm.a))
	vout = Flux.σ.(rbm.W' * h .+ rbm.a) # transpose W?
	vout = vout .> 0.5
	
	loss = Flux.Losses.mse(vin, vout)
	return loss
end

# ╔═╡ 28621b42-f79c-44e8-8838-0d0717c96cee
function train_BAS_UCD(rbm; numiter=5, batchsize=30, k=1, tmax, maxtries, x, UCD=true)
	
	runtime = 0
	
	# Choose optimizer
	η = 0.01; optimizer = Descent(η) # SGD
	# η = 0.02; optimizer = Momentum(η)
	#η = 0.0003; optimizer = ADAM(η)
	
	# We use Zygotes graddict in order to use Flux's optimizers
	θ = Flux.params(rbm.W, rbm.a, rbm.b)
    ∇θ = Zygote.Grads(IdDict(), θ)
	
	tmean = zeros(Float32, numiter)
	recloss = zeros(Float32, numiter)
	logpv = zeros(Float32, numiter)

	# Arrays for storing the variables
	numvisible, numhidden = length(rbm.a), length(rbm.b)
	hpos = zeros(Float32, (numhidden, batchsize))
	vpos = zeros(Float32, (numvisible, batchsize))
	hξₖ = zeros(Float32, (numhidden, batchsize))
	vξₖ = zeros(Float32, (numvisible, batchsize))
	hξₜ = zeros(Float32, (numhidden, batchsize))
	vξₜ = zeros(Float32, (numvisible, batchsize))
	hηₜ₋₁ = zeros(Float32, (numhidden, batchsize))
	vηₜ₋₁ = zeros(Float32, (numvisible, batchsize))
	
	for iteration=1:numiter
		t1 = time()
		# The activations for the first term are easy to compute
		vpos = x
		hpos = inference_pos!(rbm, vpos, hpos)
		#= The activations for the second term arer more difficult
		There are two parts:
		(1) The ξₖ term, which we typycally fix to use k=1. 
		We find this by running CD-K.
		(2) The ξₜ terms and the ηₜ₋₁ terms, which really are the tricky parts 
		=#
			
		# (1) get ξₖ term via CD-k
		vξₖ = deepcopy(vpos)
		vξₖ, hξₖ = inference_neg!(rbm, vξₖ, hξₖ, k)
		# At this point we have a CD-k gradient estimate
		∇combined = -∇E(vpos, hpos, batchsize) .+ ∇E(vξₖ, hξₖ, batchsize)
		
		# (2) get ξₜ terms and the ηₜ₋₁ terms
		# Is this an appropriate initialization of vξₜ, vηₜ₋₁?
		vξₜ = deepcopy(vξₖ) 
		vηₜ₋₁ = deepcopy(vξₖ) 
		Z1 = rand(Float32, size(rbm.b, batchsize))
		Z2 = rand(Float32, size(rbm.b, batchsize))
		hξₜ = Flux.σ.(rbm.W * vξₜ .+ rbm.b) .> Z1
		hηₜ₋₁ = Flux.σ.(rbm.W * vηₜ₋₁ .+ rbm.b) .> Z2

		# Perform UCD inference
		if UCD == true
			
			# proces one datapoint at a time
			for (vξₜⁱ, hξₜⁱ, vηₜ₋₁ⁱ, hηₜ₋₁ⁱ) in zip(eachcol(vξₜ), eachcol(hξₜ), eachcol(vηₜ₋₁), eachcol(hηₜ₋₁))
				
				# Iterate until Markov chains converged
				for t=1:tmax

					vξₜⁱ, hξₜⁱ, vηₜ₋₁ⁱ, hηₜ₋₁ⁱ  = coupled_inference_2(rbm, vξₜⁱ, hξₜⁱ, vηₜ₋₁ⁱ, hηₜ₋₁ⁱ, maxtries,)
					
					# break if converged or if we have reached tmax
					if (vξₜⁱ == vηₜ₋₁ⁱ && hξₜⁱ == hηₜ₋₁ⁱ) || t==tmax
						tmean[iteration] += t
						break
					# If not converged we add contribution to gradient
					else
						#vηₜ₋₁ⁱ = Flux.σ.(rbm.W' * hηₜ₋₁ⁱ .+ rbm.a)
						#vξₜⁱ = Flux.σ.(rbm.W' * hξₜⁱ .+ rbm.a)
						∇ηₜ₋₁ = ∇E(vηₜ₋₁ⁱ, hηₜ₋₁ⁱ, batchsize) 
						∇ξₜ = ∇E(vξₜⁱ, hξₜⁱ, batchsize)
						∇combined .+= (-∇ηₜ₋₁ .+ ∇ξₜ)
					end
				end
			end
		end

		# update weights
		for i=1:3
			∇θ.grads[θ[i]] = ∇combined[i]
		end
		Flux.Optimise.update!(optimizer, θ, ∇θ)
		
		
	recloss[iteration] = reconstruction_loss2(rbm , x)
	logpv[iteration] = get_logpv(rbm, x)
	tmean[iteration] /= batchsize
	t2 = time()
	runtime += t2-t1
	# println output printed to console (and not to notebook)
	if iteration == 1 || iteration %500 == 0
		runtime = round(runtime, digits=2)
		println("Iteration: ", iteration, "/", numiter, 
				":\t recloss = ", round(recloss[iteration], digits=3),
				":\t ln(p(v)) = ", round(logpv[iteration], digits=3),
				"\t runtime: ", runtime, " s")
		runtime = 0
	end
	end
return recloss, logpv, tmean
end

# ╔═╡ 7dd89ba5-e84f-49a7-8047-94f420998ae3
# Initialize the network
begin
	Random.seed!(3)
	println("\nTraining RBM") # printed to console!
	rbmBAS = init_rbm(numvisible=16, numhidden=16, glorot=false)
	(recloss_BAS, logpv_BAS, tmean_BAS) = train_BAS_UCD(rbmBAS; numiter=20000, batchsize=30, k=1, tmax=100, maxtries=10, x=BAS, UCD=true)
end;

# ╔═╡ 899e6331-faff-4a0a-ae13-7ba6ea32bc6a
tmean_BAS

# ╔═╡ 07353cda-1af9-447f-8801-62363b235c49
md"Here are the learned filters."

# ╔═╡ 298579b2-9e8e-4970-8d60-66e6a0e97ca7
img_rbmBAS =  [imshow(rbmBAS.W[i,:], w=4, h=4) for i=1:16]

# ╔═╡ 90a1b9b3-8486-4174-8ad6-46cee242d135
md"Let's see how the reconstructions look!"

# ╔═╡ 7841210a-1ff2-4ec4-8d82-4aa99cf33ad7
function reconstruct2(rbm, batchsize, x)

	vin = x
	
	r = rand(Float32, size(rbm.b, batchsize))
	h = Flux.σ.(rbm.W * vin .+ rbm.b)
	h = h .> r	
	
	#r = rand(Float32, size(rbm.a))
	vout = Flux.σ.(rbm.W' * h .+ rbm.a) # transpose W?
	#vout = vout .> r
	
	return vout
end

# ╔═╡ abb2d991-f6a9-46b5-8f86-008a72fdeab7
xrecBAS = reconstruct2(rbmBAS, 30, BAS);

# ╔═╡ 543078b8-2445-4acd-a77d-a8429c4cdbef
img_BAS2 =  [imshow(BAS[:,i], w=4, h=4) for i=1:30]

# ╔═╡ 1530b5f8-00eb-4094-8c4a-f3db351f497a
img_BAS_rec =  [imshow(xrecBAS[:,i], w=4, h=4) for i=1:30]

# ╔═╡ c208d6b2-1004-48c0-91b0-486e71848fe9
md"The reconstruction loss is plotted below. The sliders can be used to adjust the plotting range and the *radius* of the averaging filter."

# ╔═╡ f51031a1-2bbe-4c85-bee9-dbd883f22448
md"The dataset has 30 datapoints, so the log-likelihood of a perfectly fit model is $30\ln(1/30)=-102.035$."

# ╔═╡ 96490df9-a129-45a7-9280-7a2e97b25bd7
md"
- start index:    $(@bind a Slider(1:length(recloss_BAS); default=1, show_value=true))
- stop index:    $(@bind b Slider(1:length(recloss_BAS); default=length(recloss_BAS), show_value=true))
- filter radius:    $(@bind c Slider(0 : 1 : 50; default=20, show_value=true))"

# ╔═╡ ac276141-5168-45be-be8f-b8264c35c845
function smoothen(loss, radius)
	smooth = [mean(loss[i-c:i+c]) for i=c+1:length(loss)-c]
	pad_start = smooth[1]*ones(Float32, c)
	pad_end = smooth[end]*ones(Float32, c)
	smooth = hcat(pad_start..., smooth..., pad_end...)[:]
	loss_array = hcat(loss, smooth)
end;

# ╔═╡ 11bea68d-2fbb-4740-85bc-2634f9fbd47e
begin
	p1 = plot(a:b, smoothen(tmean_BAS, c)[a:b,:], w=3, labels=["t" "smoothened t"], ylim=(0,5), xticks=false)
	p2 = plot(a:b, smoothen(recloss_BAS, c)[a:b,:], w=3, labels=["rec loss" "smoothened rec loss"], xticks=false)
	p3 = plot(a:b, smoothen(logpv_BAS, c)[a:b,:], w=3, labels=["log(p(v))" "smoothened log(p(v))"], legend=:bottomright, ylim=(-350, -100))
	plot(p1, p2, p3, layout=(3, 1), size=(600,400))
end

# ╔═╡ 0fa528aa-5933-404e-b58a-25c3a338f7d8


# ╔═╡ afd46512-47ce-41a7-9338-26644d6e4a9b


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Flux = "~0.12.6"
MLDatasets = "~0.5.9"
Plots = "~1.20.0"
PlutoUI = "~0.7.9"
Zygote = "~0.6.19"
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

[[BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "84cf7d0f8fd46ca6f1b3e0305b4b4a37afe50fd6"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.0"

[[Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "e747dac84f39c62aff6956651ec359686490134e"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.0+0"

[[BufferedStreams]]
deps = ["Compat", "Test"]
git-tree-sha1 = "5d55b9486590fdda5905c275bb21ce1f0754020f"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.0.0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "DataStructures", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "5e696e37e51b01ae07bd9f700afe6cbd55250bce"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.3.4"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "11567f2471013449c2fcf119f674c681484a130e"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random", "StaticArrays"]
git-tree-sha1 = "ed268efe58512df8c7e224d2e170afd76dd6a417"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.13.0"

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

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataDeps]]
deps = ["BinaryProvider", "HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "4f0e41ff461d42cfc62ff0de4f1cd44c6e6b3771"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.7"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

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

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

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

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Functors]]
deps = ["MacroTools"]
git-tree-sha1 = "4cd9e70bf8fce05114598b663ad79dfe9ae432b3"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.3"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GPUArrays]]
deps = ["AbstractFFTs", "Adapt", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "ececbf05f8904c92814bdbd0aafd5540b0bf2e9a"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "7.0.1"

[[GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "f26f15d9c353f7091065390ea826df9e03917e58"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.12.8"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "182da592436e287758ded5be6e32c406de3a2e47"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.58.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d59e8320c2747553788e4fc42231489cc602fa50"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.58.1+0"

[[GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HDF5]]
deps = ["Blosc", "Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "83173193dc242ce4b037f0263a7cc45afb5a0b85"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.15.6"

[[HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "fd83fa0bde42e01952757f01149dd968c06c4dba"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.0+1"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "44e3b40da000eab4ccb1aecdc4801c040026aeb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.13"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "95215cd0076a150ef46ff7928892bc341864c73c"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.3"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "d6041ad706cf458b2c9f3e501152488a26451e9c"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.2.0"

[[LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "a9b1130c4728b0e462a1c28772954650039eb847"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.7+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

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

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

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
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

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

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "5c62992f3d46b8dce69bdd234279bb5a369db7d5"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.1"

[[MLDatasets]]
deps = ["BinDeps", "ColorTypes", "DataDeps", "DelimitedFiles", "FixedPointNumbers", "GZip", "MAT", "PyCall", "Requires"]
git-tree-sha1 = "65cb0a663d65d0b782ba74bfc3982ba51eb85485"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.5.9"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

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
git-tree-sha1 = "16520143f067928bb69eee59ac8bca06be1e43b8"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.27"

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

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

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

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "477bf42b4d1496b454c10cce46645bb5b8a0cf2c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.2"

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
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "e39bea10478c6aff5495ab522517fae5134b40e3"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.20.0"

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

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "169bb8ea6b1b143c5cf57df6d34d022a7b60c6db"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.3"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

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
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2a7a2469ed5d94a98dea0e85c46fa653d76be0cd"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.4"

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
git-tree-sha1 = "3fedeffc02e47d6e3eb479150c8e5cd8f15a77a0"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.10"

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

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "000e168f5cc9aded17b6999a560b7c11dda69095"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.0"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

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

[[URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

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

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "c3a5637e27e914a7a445b8d0ad063d701931e9f7"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "f01bac579bb397ab138aed7e9e3f80ef76d055f7"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.19"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═03fcb360-421d-441a-bd8c-5372d6bb2be5
# ╠═37c90723-2e16-4e40-a259-74edf117c6e2
# ╠═61ce3068-a319-4f67-b4fd-25745727f0a1
# ╠═360c13ed-0452-4811-a192-fa21968aae04
# ╠═118b4c69-d21a-4a35-909a-f1631e83b917
# ╠═136b876c-f545-46ac-befd-af7d37ea9d93
# ╠═a1fb6aad-fb1c-4418-9caa-f7add68bf26e
# ╠═f270bf0c-00b6-4308-93bd-2cd0c4dead24
# ╠═8b11badc-2a03-4918-841a-a6459d1aac28
# ╟─ef35700e-8df6-4446-b9f4-2e82bf8801c0
# ╟─9e830b5e-f37f-11eb-083f-277a24c3cd6c
# ╟─1a5a0086-cfba-470f-955f-82df7c5f19de
# ╟─d9e2c1de-ee61-4413-8b41-8bcad7206d1d
# ╟─adb63694-4f58-4d96-84c2-87e3fd69d5ec
# ╟─5a5299e2-4a18-4a52-ae87-453380edc682
# ╟─d33cc5e2-9135-4dd0-b043-67ff5b5edaf6
# ╠═5179313d-576f-4433-82cc-bf2cb7907abd
# ╠═6d210251-d433-43b6-b515-c852ccbc1feb
# ╟─b775575b-33e7-4708-8c6e-4c28f9cfa79f
# ╠═72a1fd39-2980-4f14-9a67-5362f9bb0775
# ╠═a9c6ecab-bc6c-4565-9a29-7d07b95c2de9
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
# ╠═64dde393-ac97-4140-a69f-549bd7f7ce85
# ╟─282f9d60-8d8a-40dd-b077-4afe3b3d6313
# ╟─e1880f54-cfc9-485e-b219-7849a893a838
# ╟─2271d6f2-54a4-4f26-a546-8889c227992d
# ╟─0cc27c9c-ec58-4ef1-bd58-09954761020b
# ╟─74461104-7080-4b58-a159-0b2b6ac5fac5
# ╟─2a36f2df-3f73-4c5d-b154-4a0d1759fcca
# ╟─8eb87c5b-c895-43c1-93aa-687275a31c87
# ╟─37e880fa-70e7-47d1-b5a8-04cff3f5d828
# ╟─94fe813e-8047-4372-981d-f1759442975b
# ╠═93ca8efa-1bc5-473c-83cc-5994af633659
# ╠═91d233f2-12d3-4594-b68e-5a6b3d8e633f
# ╟─1b0875fd-975c-47f3-a1c1-6278084d77c5
# ╠═343ad144-8b26-4c6c-8f3d-4cf042c46cf0
# ╟─e8ac304a-84dd-4ed9-80a3-c0786c734f14
# ╠═31a4f5c7-9d7b-47a8-a27a-3a421fc0f10f
# ╠═7906a124-67ea-42ae-8f27-80eaba3e3368
# ╠═7a782a84-3615-4ad7-b6bb-a500966cb5ac
# ╟─a6a0df67-885d-4799-b3cb-864f09f629a7
# ╠═0c893210-9bd2-43b8-ab46-d9579707eed2
# ╠═6b10fdc8-e871-4de1-b2cb-e81c610823e3
# ╟─5690fc6d-d3b4-478c-8efe-cd2c03a915af
# ╠═944f0cf9-8302-41f4-9b9d-f90523827bac
# ╟─711787c1-f8fc-4fac-92c2-21a01ab4937d
# ╟─09852337-608d-4ef4-819d-74437bf978bc
# ╠═6cc91180-c85f-4e46-93bb-668234023328
# ╠═f0c0bf3b-3329-4619-ba86-366b7abe3c79
# ╟─0ca12440-3025-48ff-9aa7-aed2ed01d9f6
# ╠═816e0fe7-add7-41e9-8a8c-41d67c44eec8
# ╠═9cef19bc-5295-456d-b6fe-a7cb1099fa6f
# ╠═1098fb24-b08a-4598-b44d-8f356877af25
# ╠═004f1d73-b909-47da-b25d-aa83787520e9
# ╟─79a8e531-6c29-4cf5-8429-0d7474b01f29
# ╟─a1afe50c-3d5c-4c50-99dd-73aef979e24a
# ╟─3794e224-9bba-481d-b072-4abde652c627
# ╟─55cee522-7a7d-409e-bcee-5dd6f41c701f
# ╟─6a914ea6-349a-43b9-8152-6e4452786646
# ╟─a039ce76-e5e8-4d22-a23c-c1d03849ada3
# ╟─8b7f2e1c-d21d-4d51-83cc-6a118c34586d
# ╟─8f64ebad-d946-4f90-bbad-c2923fad1c65
# ╟─c05d48e3-4b20-4c52-a437-b376ed8c5756
# ╟─2588712f-6a5e-4fd1-9b2c-b6c7a20e9199
# ╟─928fd624-c571-43f4-8bf7-346ec51deeb9
# ╟─db748b83-87d9-4354-9a74-b15424119d64
# ╟─c8c8c68a-826e-4877-88a5-59866f422d40
# ╟─77b4d504-7eab-49e8-ab5e-b576250c5411
# ╟─243c9723-6e33-4e02-baaa-857d4f4cd344
# ╠═e97a9dde-5e0f-48ae-a72d-56cea3410ad9
# ╠═0593e773-d877-4c8c-ae7f-826cad5cf75a
# ╠═c0944ff3-c13a-484d-90ed-8d29bb52bd31
# ╠═431a65ad-152f-4686-bfe1-cf856056dffd
# ╠═b85a8965-2215-4742-91b5-a60dd4dce499
# ╟─643f9028-1f95-4c0c-b842-7cdbb1603914
# ╟─d8152e9d-0846-4f20-9242-3f27d80f3243
# ╟─e675cc20-01b0-44eb-8307-5f12fa8e47d3
# ╠═e24aa7ea-211b-4f15-bc17-359478b02cb7
# ╟─248e57a5-a446-4489-8ef6-a840f08d55b9
# ╟─a64f08d0-c7c6-4575-af8b-40dbdce4cf5d
# ╠═8b5722e3-5a16-4806-b859-6cacb9ee5e13
# ╟─5cf76965-fabe-4a2e-beab-6abf3a216e90
# ╠═6c96b07d-2f21-45b8-880a-28ae41da3b4b
# ╟─277ea081-5088-49c3-b1d9-1ac6272b6d22
# ╟─8b8e35ba-b332-4afc-a8c5-865d73775fa9
# ╟─42aa9749-c4da-4aa9-a89d-c9323bf837ec
# ╟─2a7c2484-9928-4432-9450-f2823d3a75eb
# ╟─d033638d-f0f6-46db-80bf-648b39a52f33
# ╟─c2e9f61c-2316-4f45-8709-97b9268b0795
# ╠═5844b476-5827-4ba9-bd70-370326cd71ad
# ╠═a7b081dc-ab39-4ffa-a393-8cb7ce51c076
# ╠═b87a54bd-3ff9-4fc6-a898-54029658a0b7
# ╠═535c8a18-b6f3-4a14-b4a4-4c965b00c85d
# ╠═28621b42-f79c-44e8-8838-0d0717c96cee
# ╠═7dd89ba5-e84f-49a7-8047-94f420998ae3
# ╠═899e6331-faff-4a0a-ae13-7ba6ea32bc6a
# ╟─07353cda-1af9-447f-8801-62363b235c49
# ╠═298579b2-9e8e-4970-8d60-66e6a0e97ca7
# ╟─90a1b9b3-8486-4174-8ad6-46cee242d135
# ╟─7841210a-1ff2-4ec4-8d82-4aa99cf33ad7
# ╠═abb2d991-f6a9-46b5-8f86-008a72fdeab7
# ╠═543078b8-2445-4acd-a77d-a8429c4cdbef
# ╠═1530b5f8-00eb-4094-8c4a-f3db351f497a
# ╟─c208d6b2-1004-48c0-91b0-486e71848fe9
# ╟─f51031a1-2bbe-4c85-bee9-dbd883f22448
# ╟─96490df9-a129-45a7-9280-7a2e97b25bd7
# ╟─ac276141-5168-45be-be8f-b8264c35c845
# ╠═11bea68d-2fbb-4740-85bc-2634f9fbd47e
# ╠═0fa528aa-5933-404e-b58a-25c3a338f7d8
# ╠═afd46512-47ce-41a7-9338-26644d6e4a9b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
