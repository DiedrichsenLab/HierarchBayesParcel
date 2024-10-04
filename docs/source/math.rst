Mathematical details
====================

A generative modelling framework for individual brain organization across a number of different data. The Model is partitioned into a model that determines the probability of the spatial arrangement of regions in each subject :math:`s`, :math:`p(\mathbf{U}^{(s)};\theta_A)` and the probability of observing a set of data at each given brain location. We introduce the Markov property that the observations are mutually independent, given the spatial arrangement.

.. math::
	p(\mathbf{Y}^{(s)}|\mathbf{U}^{(s)};\theta_E)=\prod_i p(\mathbf{y}_i^{(s)}|\mathbf{u}_i^{(s)};\theta_E)

Inference and learning
----------------------

We will learn the model, by maximizing the ELBO (Evidence lower bound). For clarity, I am dropping the index for the subject :math:`s` for now,

.. math::
	\begin{align*}
	\log p(\mathbf{Y} | \theta)
	&=\log\sum_{\mathbf{U}}p(\mathbf{Y},\mathbf{U}|\theta) \\
	&=\log\sum_{\mathbf{U}}q(\mathbf{U})\frac{p(\mathbf{Y},\mathbf{U}|\theta)}{q(\mathbf{U})}\\
	&\geqslant \sum_{\mathbf{U}} q(\mathbf{U}) \log \frac{p(\mathbf{Y},\mathbf{U}|\theta)}{q(\mathbf{U})} \tag{Jensen's inequality}\\
	&=\langle \log p(\mathbf{Y},\mathbf{U}|\theta) - \log q(\mathbf{U})\rangle_q
	\triangleq \mathcal{L}(q, \theta) - \log \langle q(\mathbf{U})\rangle_q
	\end{align*}

Given the markov property, we can break the expected complete log likelihood into two pieces, one containing the parameters for the arrangement model and one containing the parameters for the emission model.

.. math::
	\begin{align*}
	\langle \log p(\mathbf{Y},\mathbf{U}|\theta)\rangle_q &=\langle \log(p(\mathbf{Y}|\mathbf{U};\theta_E) p(\mathbf{U}|\theta_A))\rangle_q\\
	&=\langle \log p(\mathbf{Y}|\mathbf{U};\theta_E)\rangle_q + \langle \log p(\mathbf{U}|\theta_A)\rangle_q\\
	&\triangleq \mathcal{L}_E+\mathcal{L}_A
	\end{align*}

We will refer to the first term as the expected emission log-likelihood and the second term as the expected arrangement log-likelihood. We can estimate the parameters of the emission model from maximizing the  expected emission log-likelihood, and we can estimate the parameters of the arrangement model by maximizing the expected arrangement log-likelihood.

Arrangement models
------------------

This is a generative Potts model of brain activity data. The main idea is that the brain consists of :math:`K` regions, each with a specific activity profile :math:`\mathbf{v}_k` for a specific task set. The model consists of a arrangement model that tells us how the :math:`K` regions are arranged in a specific subject :math:`s`, and an emission model that provides a probability of the measured data, given the individual arrangement of regions.

Independent Arrangement model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the simplest spatial arrangement model - it simply learns the probability at each location that the node is part of cluster :math:`K`. These probabilities are simply learned as the parameters :math:`\pi_{ik}=p(u_i=k)`, or after a re-parameterization in log space: :math:`\eta_{ik}=\log \pi_{ik}`. Vice versa (not assuming that the :math:`\eta` are correctly scaled):

.. math::
	\pi_{ik}=\frac{\rm{exp}(\eta_{ik})}{\sum_{j}\rm{exp}(\eta_{ij})}

This independent arrangement model can be estimated using the EM-algorithm. In the Estep, we are integrating the evidence from the data and the prior:

.. math::
	p(u_i=k|\mathbf{y}_i)=\langle u_{ik}\rangle=\frac{\rm{exp}(log(p(\mathbf{y}_i|u_i=k))+\eta_{ik})}{\sum_{j}{\rm{exp}(log(p(\mathbf{y}_i|u_i=j))+\eta_{ij}})}

Or in vector notation:

.. math::
	\begin{align*}
	\langle \mathbf{u}_{i}\rangle =\rm{softmax}(log(p(\mathbf{y}_i|\mathbf{u}_i))+\boldsymbol{\eta}_i)
	\end{align*}

For the M-step, we use the derivative of the expected arrangement log-likelihood in respect to the parameters :math:`\eta`:

.. math::
	\begin{align}
	\mathcal{L}_A&=\sum_{i}\sum_{k}\langle u_{i,k}\rangle  \rm{log}(\pi_{ik})\\
	&=\sum_{i}\sum_{k}\langle u_{ik}\rangle(\eta_{ik}-\rm{log}\sum_j\rm{exp}(\eta_{ij}))\\
	&=\sum_{i}\sum_{k}\langle u_{ik}\rangle\eta_{ik}-\sum_{i}\log\sum_j\exp(\eta_{i,j})\\
	\end{align}

So the derivative is

.. math::
	\begin{align}
	\frac{\partial\mathcal{L}_A}{\partial{\eta_{ik}}}&=\langle u_{ik}\rangle-\frac{\partial}{\partial\eta_{ik}}\log\sum_j\exp(\eta_{ij})\\
	&=\langle u_{ik}\rangle-\frac{1}{\sum_j\exp(\eta_{ij})}\frac{\partial}{\partial\eta_{ik}}\sum_j\exp(\eta_{ij})\\\
	&=\langle u_{ik}\rangle-\frac{\exp(\eta_{ik})}{\sum_j\exp(\eta_{ij})}\\
	&=\langle u_{ik}\rangle-\pi_{ik}
	\end{align}

We can also get the same result directly by the application of chain rule: For a good introduction, see: [https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/].

.. math::
	\frac{\partial{\pi_k}}{\eta_k}=\pi_k(\delta_{kj}-\pi_j)

Simple Potts model
^^^^^^^^^^^^^^^^^^

The brain is sampled in :math:`P` vertices (or voxels). Individual maps are aligned using anatomical normalization, such that each vertex refers to a (roughly) corresponding region in each individual brain. The assignment of each brain location to a specific parcel in subject :math:`s` is expressed as the random variable :math:`u_i^{(s)}`.

Across individual brains, we have the overall probability of a specific brain location being part of parcel :math:`k`.

.. math::
	p(u_i = k) = \pi_{ki}

The spatial interdependence of brain locations is expressed as a Potts model. In this model, the overall probability of a specific assignment of brain locations to parcels (the vector :math:`\mathbf{u}`) is expressed as the product of the overall prior and the product of all possible pairwise potentenials (:math:`\psi_{ij}`).

.. math::
	p(\mathbf{u}) = \frac{1}{Z(\theta)} \prod_{i}\pi_{u_i,i}\prod_{i\neq j}{\psi_{ij}(u_i,u_j) }

Each local potential is defined by an exponential over all other that are connected to node :math:`i`, i.e. nodes with connectivity weights of :math:`w_{ji}=w_{ij}>0`.

.. math::
	\psi_{ij}=  \rm{exp}(\theta_{w}\mathbf{u}_i^T\mathbf{u}_j w_{ij})

Where we have introduced a one-hot encoding of :math:`u_i` with a :math:`K` vector of indicator variables :math:`\mathbf{u}_i`, such that :math:`\mathbf{u}_i^T\mathbf{u}_j = 1` if :math:`u_i = u_j` and :math:`0` otherwise.

The spatial co-dependence across the entire brain is therefore expressed with the pairwise weights :math:`w` that encode how likely two nodes belong to the same parcel. The temperature parameter :math:`\theta_w` determines how strong this co-dependence overall influences the local probabilies (relative to the prior). We can use this notation to express local co-dependencies by using a graph, where we define

.. math::
	w_{ij}=\begin{cases}
	1; \text{if i and j are neighbours}\\
	0; \text{otherwise}
	\end{cases}

This formulation would enforce local smoothness of the map. However, we could also express in these potential more medium range potentials (two specific parietal and premotor areas likely belong to the same parcel), as well as cross-hemispheric symmetry. Given this, the matrix :math:`\mathbf{W}` could be simply derived from the underlying grid or be learned to reflect known brain-connectivity.

The expected arrangement log-likelihood therefore becomes:

.. math::
	\begin{align*}
	\mathcal{L}_A=\sum_i \langle\mathbf{u}_i\rangle^T \log{\boldsymbol{\pi}_{i}}+\theta_w \sum_i \sum_j w_{ij} \langle\mathbf{u}_i^T\mathbf{u}_j\rangle - \log Z
	\end{align*}

Inference using stochastic maximum likelihood / contrastive divergence
**********************************************************************

We can approximate the gradient of the parameters using a contrastive divergence-type algorithm. We view the arrangement log-likelihood as a sum of the unnormalized part (:math:`\tilde{\mathcal{L}}_A`) and the log partition function. For each parameter :math:`\theta` we then follow the gradient

.. math::
	\begin{align*}
	\nabla_\theta \mathcal{L}_A&=\nabla_\theta \tilde{\mathcal{L}}_A-\nabla_\theta \log Z\\
	&=\nabla_\theta \tilde{\mathcal{L}}_A-\mathrm{E}_p [\nabla_\theta \tilde{\mathcal{L}}_A]\\
	&=\nabla_\theta \langle \log \tilde{p}(\mathbf{U}|\theta)\rangle_q -
	\nabla_\theta \langle \log \tilde{p}(\mathbf{U}|\theta)\rangle_p
	\end{align*}

Thus, we can use the gradient of the unnormalized expected log-likelihood (given a distribution :math:`q(\mathbf{U}) = p(\mathbf{U}|\mathbf{Y};\theta)`, minus the gradient of the unnormalized expected log-likelihood in respect to the expectation under the model parameters, without seeing the data, :math:`q(\mathbf{U}) = p(\mathbf{U}|\mathbf{Y};\theta)`. This motivates the use of sampling / approximate inference for both of these steps. See Deep Learning (18.1).

E-step: sampling from prior or posterior distribution
*****************************************************

The problem is that the two expectations under the prior (p) and the posterior (q) distribution of the model cannot be easily be computed. We can evaluate the prior probability of a parcellation :math:`p(\mathbf{U})` or the posterior distribution :math:`p(\mathbf{U}|\mathbf{Y})` up to a constant of proportionality, with for example

.. math::
	p(\mathbf{U}|\mathbf{Y};\theta) = \frac{1}{Z(\theta)}\prod_{i}\mu_{u_i,i}\prod_{i\neq j}{\psi_{ij}(u_i,u_j) }\prod_{i}p(\mathbf{y}_i|u_i)

Calculating the normalization constant :math:`Z(\theta)` (partition function, Zustandssumme, or sum over states) would involve summing this probability over all possible states, which for :math:`P` brain locations and :math:`K` parcels is :math:`K^P`, which is intractable.

However, the conditional probability for each node, given all the other nodes, can be easily computed. Here the normalizaton constant is just the sum of the potential functions over the :math:`K` possible states for this node


.. math::
	p(u_i|u_{j \neq i},\mathbf{y}_i;\theta) = \frac{1}{Z(\theta)}\mu_{u_i,i} \; p(\mathbf{y}_i|u_i) \prod_{i\neq j}{\psi_{ij}(u_i,u_j) }

With Gibbs sampling, we start with a pattern :math:`\mathbf{u}^{(0)}` and then update :math:`u_1^{(1)}` by sampling from :math:`p(u_1|u_2^{(0)}...u_P^{(0)})`. We then sample :math:`u_2^{(1)}` by sampling from :math:`p(u_2|u_1^{(1)}, u_3^{(0)}...u_P^{(0)})` and so on, until we have sampled each node once. Then we return to the beginning and restart the process. After some burn-in period, the samples will come from desired overall distribution. If we want to sample from the prior, rather than from the posterior, we simply drop the :math:`p(\mathbf{y}_i|u_i)` term from the conditional probability above.

Gradient for different parametrization of the Potts model
*********************************************************

For the edge-energy parameters :math:`\theta_w` we clearly want to use the natural parametrization with the derivate:

.. math::
	\frac{\partial \tilde{\mathcal{L}}_A}{\partial \theta_w}=\sum_i\sum_j w_{ij}\langle\mathbf{u}_i^T\mathbf{u}_j\rangle

For the prior probability of each parcel :math:`k` at each location :math:`i`  (:math:`\pi_{ik}`) we have a number of options.

First ,we can use the probabilities themselves as parameters:

.. math::
	\frac{\partial \tilde{\mathcal{L}}_A}{\partial \pi_{ik}}=\frac{\langle u_{ik}\rangle}{\pi_{ik}}


This is unconstrained (that is probabilities do not need to sum to 1), and the normalization would happen through the partition function.

Secondly, we can use a re-parameterization in log space, which is more natural: :math:`\eta_{ik}=\log \pi_{ik}`. In this case the derivative of the non-normalized part just becomes:

.. math::
	\frac{\partial \tilde{\mathcal{L}}_A}{\partial \eta_{ik}}=\langle u_{ik}\rangle

Finally, we can implement the constraint that the probabilities at each location sum to one by the following re-parametrization:

.. math::
	\begin{align*}
	\pi_{iK}&=1-\sum_{k=1}^{K-1}\pi_{ik}\\
	\eta_{ik}&=\log(\frac{\pi_{ik}}{\pi_{iK}})=\log{\pi_{ik}}-\log({1-\sum_{k=1}^{K-1}\pi_{ik}})\\
	\pi_{ik}&=\frac{\exp(\eta_{ik})}{1+\sum_{k=1}^{K-1}\exp(\eta_{ik})}\\
	\pi_{iK}&=\frac{1}{1+\sum_{k=1}^{K-1}\exp(\eta_{ik})}
	\end{align*}

In the implementation, we can achieve this parametrization easily by defining a non-flexible parameter :math:`\eta_{iK}\triangleq0`. Then we can treat the last probability like all the other ones.

With this constrained parameterization, we can rewrite the unnormalized part of the expected log-likelihood as:


.. math::
	\begin{align*}
	\tilde{\mathcal{L}}_A&=\sum_i \sum_{k}^{K-1}\langle u_{ik}\rangle  \log \pi_{ik}+[1-\sum_{k}^{K-1}\langle u_{ik}\rangle]\log{\pi_{iK}}+C\\
	&=\sum_i \sum_{k}^{K-1}\langle u_{ik}\rangle  (\log \pi_{ik}-\log \pi_{iK})+\log{\pi_{iK}}+C\\
	&=\sum_i \sum_{k}^{K-1}\langle u_{ik}\rangle \eta_{ik}-\log(1+\sum_{k=1}^{K-1}\exp(\eta_{ik}))+C\\
	\end{align*}

where C is the part of the normalized log-likelihood that does not depend on :math:`\pi`. Taking derivative in respect to :math:`\eta_{ik}` yields:

.. math::
	\begin{align*}
	\frac{\partial\tilde{\mathcal{L}}_A}{\partial\eta_{ik}}&=\langle u_{ik}\rangle - \frac{\exp(\eta_{ik})}{1+\sum_k^{K-1}\exp(\eta_{ik})}\\
	&=\langle u_{ik}\rangle - \pi_{ik}
	\end{align*}

So in this parameterization in the iid case, :math:`Z=1` and we don't need the negative step. In general, however, we cannot simply set the above derivative to zero and solve it, as the parameter :math:`\theta_w` will also have an influences on :math:`\langle u_{ik} \rangle`.


Probabilistic multinomial restricted Boltzmann machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an alternative to a Potts model, we are introducing here a multivariate version of a restricted Boltzmann machine. A restricted Boltzmann machine consists typically out a layer of binary visible and a layer of binary hidden units (:math:`\mathbf{h}`) with :math:`J` nodes :math:`h_j`. Here, we are replacing the input with the spatial arrangement matrix :math:`\mathbf{U}`, with each column of the matrix :math:`\mathbf{u}_i` representing a one-hot encoded multinomial random variable, that assigns the brain location :math:`i` to parcel :math:`k`.

The hidden variables is still a vector of binary latent variables

.. math::
	p(h_j|\mathbf{U})=\sigma(vec(\mathbf{U})^T\mathbf{W}_{.,j}+\mathbf{b}_j)

Where :math:`\sigma` is the sigmoid function.

The probability of a brain location then is given by:

.. math::
	p(\mathbf{u}_i|\mathbf{h})=\rm{softmax}([\mathbf{h}^T\mathbf{W}^T]_i+\boldsymbol{\eta}_i)


Where :math:`[.]_i` selects the element for :math:`\mathbf{u}_i` from vectorized version of :math:`\mathbf{U}`.

Positive Estep: Expectation given the data (RBM)
************************************************

The advantage of a Boltzmann machine is that we can efficiently do inference and sampling in a blocked fashion. In the positive E-step, the expectation can be passed - and we can do one or more iteration between the :math:`\mathbf{h}` and the :math:`\mathbf{U}` layer.

We intialize the hidden layer with

.. math::
	\langle\mathbf{h}\rangle^{(0)}_q=\mathbf{0}

An then alternate:

.. math::
	\langle\mathbf{u}_i\rangle^{(t)}_q=\rm{softmax}([\mathbf{W} \langle \mathbf{h}\rangle^{(t)}]_i+\boldsymbol{\eta}_i + \log p(\mathbf{y}_i|\mathbf{u}_i))

.. math::
	\langle h_j\rangle^{(t+1)}_q =\sigma(vec(\langle \mathbf{U} \rangle^{(t)}_q)^T\mathbf{W}_{.,j}+\mathbf{b}_j)


Negative Estep: Expectation given the model (RBM)
*************************************************

For the negative e-step, we are using sampling alternating for :math:`\mathbf{h}` and :math:`\mathbf{U}`, using the main equations. The expectations are then probabilities before the last sampling step. These give us the expectations :math:`\langle . \rangle_p` that we need for subsequent learning.

Gradient step for parameter estimation (RBM)
********************************************

Given the expectation of the hidden and latent variable for the positive and negative phase of the expectation.

.. math::
	\begin{align*}
	\nabla_W = \langle \mathbf{h} \rangle_q^T vec(\langle \mathbf{U} \rangle_q)-\langle \mathbf{h} \rangle_p^T vec(\langle \mathbf{U} \rangle_p)\\
	\nabla_b =\langle \mathbf{h} \rangle_q - \langle \mathbf{h} \rangle_p\\
	\nabla_\eta =\langle \mathbf{U} \rangle_q - \langle \mathbf{U} \rangle_p
	\end{align*}

Convolutional multinomial probabilistic restricted Boltzmann machine (cmpRBM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another approach is to make both the hidden (:math:`\mathbf{H}`) and the intermedicate (:math:`\mathbf{U}`) nodes are multinomial version of a restricted Boltzmann machine. So with Q hidden nodes, H is the KxQ matrix with the one-hot  encoded state of the hidden variables, and U is a KxP matrix of the one-hot encoded clustering. :math:`\mathbf{W}` is the :math:`QxP` matrix of connectivity that connects the respective nodes.

The hidden variables is still a vector of binary latent variables

.. math::
	p(\mathbf{h}_j|\mathbf{U})=\rm{softmax}(\mathbf{U}\mathbf{W}_{j,.}^T)

The probability of a brain location then is given by:

.. math::
	p(\mathbf{u}_i|\mathbf{h})=\rm{softmax}(\mathbf{H}\mathbf{W}_{.,i}+\boldsymbol{\eta}_i).

Positive Estep: Expectation given the data (cmpRBM)
***************************************************

The advantage of a Boltzmann machine is that we can efficiently do inference and sampling in a blocked fashion. In the positive E-step, the expectation can be passed - and we can do one or more iteration between the :math:`\mathbf{H}` and the :math:`\mathbf{U}` layer.

We intialize the hidden layer with

.. math::
	\langle\mathbf{H}\rangle^{(0)}_q=\mathbf{0}

An then alternate:

.. math::
	\langle\mathbf{u}_i\rangle^{(t)}_q=\rm{softmax}(\langle \mathbf{H}\rangle^{(t)}\mathbf{W}_{.,i} +\boldsymbol{\eta}_i + \log p(\mathbf{y}_i|\mathbf{u}_i))

.. math::
	\langle \mathbf{h}_j\rangle^{(t+1)}_q =\rm{softmax}(\langle \mathbf{U} \rangle^{(t)}_q\mathbf{W}_{j,.}^T)

Negative Estep: Expectation given the model (cmpRBM)
****************************************************

For the negative e-step, nwe are using sampling alternating for :math:`\mathbf{h}` and :math:`\mathbf{U}`, using the main equations. The expectations are then probabilities before the last sampling step. These give us the expectations :math:`\langle . \rangle_p` that we need for subsequent learning.

Gradient step for parameter estimation (cmpRBM)
***********************************************

The unnormalized log-probability of the model (negative Energy function) of the model is:

.. math::
	\log\tilde{p}(\mathbf{U},\mathbf{H}|\mathbf{Y})=\sum_i\eta_i^T\mathbf{u}_i+\rm{tr}(\mathbf{H}\mathbf{W}\mathbf{U}^T)

Given the expectation of the hidden and latent variable for the positive and negative phase of the expectation, the gradients are:

.. math::
	\begin{align*}
	\nabla_W = \langle \mathbf{H} \rangle_q^T \langle \mathbf{U} \rangle_q-\langle \mathbf{H} \rangle_p^T \langle \mathbf{U} \rangle_p\\
	\nabla_\eta =\langle \mathbf{U} \rangle_q - \langle \mathbf{U} \rangle_p
	\end{align*}


Emission models
---------------

Given the Markov property, the emission models specify the log probability of the observed data as a function of :math:`\mathbf{u}`.

.. math::
	\log p(\mathbf{Y}|\mathbf{U};\theta_E)=\sum_i \log p(\mathbf{y}_i|\mathbf{u}_i;\theta_E)

Furthermore, assuming that :math:`\mathbf{u}_i` is a one-hot encoded indicator variable (parcellation model), we can write the expected emission log-likelihood as:

.. math::
	\langle \log p(\mathbf{Y}|\mathbf{U};\theta_E)\rangle =\sum_i \sum_k \langle u_i^{(k)}\log p(\mathbf{y}_i|u_i=k;\theta_E) \rangle

In the E-step the emission model simply passes :math:`p(\mathbf{y}_i|\mathbf{u}_i;\theta_E)` as a message to the arrangement model. In the M-step, :math:`q(\mathbf{u}_i) = \langle \mathbf{u}_i \rangle` is passed back, and the emission model optimizes the above quantity in respect to :math:`\theta_E`.

Emission model 1: Multinomial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple (but instructive) emission model is that the observed data simpy has a multinomial distribution, like the latent variables :math:`\mathbf{u}`. The coupling between the latent and the observed variable is stochastic, using a Potts model between the two nodes:

.. math::
	p(\mathbf{y_i}|\mathbf{u}_i;\theta_E) =  \frac{\exp(\mathbf{y}_i^T \mathbf{u}_i w)}{(K-1)+\exp(w)}

The expected emission loglikelihood therefore is:

.. math::
	\begin{align*}
	\mathcal{L}_E=\sum_{i} (\mathbf{y}_i^T \langle \mathbf{u}_i \rangle w - \log((K-1)+\exp(w)))
	\end{align*}

The derivative in respect to w then becomes:

.. math::
	\frac{\partial \mathcal{L}_E}{\partial w} =
	\sum_{i}^P \mathbf{y}_i^T \langle \mathbf{u}_i \rangle  - P\frac{\exp(w)}{(K-1)+\exp(w)}

After setting the derivate to zero and solving for :math:`w`, we obrain for the M-step:

.. math::
	\frac{\partial \mathcal{L}_E}{\partial w} =
	\sum_{i}^P \mathbf{y}_i^T \langle \mathbf{u}_i \rangle  - P\frac{\exp(w)}{(K-1)+\exp(w)} = 0

So that, we have

.. math::
	\begin{align*}
	\sum_{i}^P \mathbf{y}_i^T \langle \mathbf{u}_i \rangle &= P\frac{\exp(w)}{(K-1)+\exp(w)}\\
	\sum_{i}^P \mathbf{y}_i^T \langle \mathbf{u}_i \rangle / P &= 1-\frac{(K-1)}{(K-1)+\exp(w)}\\
	1-\sum_{i}^P \mathbf{y}_i^T \langle \mathbf{u}_i \rangle / P &= \frac{(K-1)}{(K-1)+\exp(w)}\\
	\frac{(K-1)}{1-\sum_{i}^P \mathbf{y}_i^T \langle \mathbf{u}_i \rangle / P} &= (K-1)+\exp(w)\\
	1-K+\frac{(K-1)}{1-\sum_{i}^P \mathbf{y}_i^T \langle \mathbf{u}_i \rangle / P} &= \exp(w)\\
	w&=\log(1-K+\frac{(K-1)}{1-\sum_{i}^P \mathbf{y}_i^T \langle \mathbf{u}_i \rangle / P})
	\end{align*}


Emission model 2: Mixture of Gaussians
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under the Gaussian mixture model, we model the emissions as a Gaussian with a parcel-specific mean (:math:`\mathbf{v}_k`), and with equal isotropic variance across parcels and observations:

.. math::
	p(\mathbf{y_i}|u^{(k)};\theta_E) = \frac{1}{(2\pi)^{N/2}(\sigma^{2})^{N/2}}\rm{exp}\{-\frac{1}{2\sigma^{2}}(y_{i}-\mathbf{X}\mathbf{v}_k)^T(y_{i}-\mathbf{X}\mathbf{v}_k)\}

The expected emission log-likelihood therefore is:

.. math::
	\begin{align*}
	\mathcal{L}_E&=\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}[-\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}(\mathbf{y}_{i}-\mathbf{X}\mathbf{v}_{k})^T(\mathbf{y}_{i}-\mathbf{X}\mathbf{v}_{k})]\\
	&=-\frac{PN}{2}\log(2\pi)-\frac{PN}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}[(\mathbf{y}_{i}-\mathbf{X}\mathbf{v}_{k})^T(\mathbf{y}_{i}-\mathbf{X}\mathbf{v}_{k})]
	\end{align*}

Now, with the above expected emission log likelihood by hand, we can update the parameters :math:`\theta_E = \{\mathbf{v}_1,...,\mathbf{v}_K,\sigma^2\}` in the M-step.

1. Updating :math:`\mathbf{v}_k`, we take derivative of *expected emission log likelihood* with respect to :math:`\mathbf{v}_{k}` and set it to 0:

.. math::
	\frac{\partial \mathcal{L}_E}{\partial \mathbf{v}_{k}} =\frac{1}{\sigma^{2}}\sum_{i}\langle u_{i}^{(k)}\rangle_{q}(\mathbf{y}_{i}-\mathbf{X}\mathbf{v}_{k}) = 0

Thus, we get the updated :math:`\mathbf{v}_{k}` in current M-step as,

.. math::
	\mathbf{v}_{k}^{(t)} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\frac{\sum_{i}\langle u_{i}^{(k)}\rangle_{q}^{(t)}\mathbf{y}_{i}}{\sum_{i}\langle u_{i}^{(k)}\rangle_{q}^{(t)}}


2. Updating :math:`\sigma^{2}`, we take derivative of *expected emission log likelihood* :math:`\mathcal{L}(q, \theta)` with respect to :math:`\sigma^{2}`  and set it to  0:

.. math::
	\frac{\partial \mathcal{L}_E}{\partial \sigma^{2}} =-\frac{PN}{2\sigma^{2}}+\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}[\frac{1}{2\sigma^{4}}(\mathbf{y}_{i}-\mathbf{X}\mathbf{v}_{k}^{(t)})^T(\mathbf{y}_{i}-\mathbf{X}\mathbf{v}_{k}^{(t)})] = 0

Thus, we get the updated :math:`\sigma^{2}` for parcel :math:`k` in the current M-step as,

.. math::
	{\sigma^{2}}^{(t)} = \frac{1}{PN}\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}^{(t)}(\mathbf{y}_{i}-\mathbf{X}\mathbf{v}_{k}^{(t)})^T(\mathbf{y}_{i}-\mathbf{X}\mathbf{v}_{k}^{(t)})

where :math:`P` is the total number of voxels :math:`i`.

The updated parameters :math:`\theta_{k}^{(t)}` from current :math:`\mathbf{M}`-step will be passed to the next :math:`\mathbf{E}`-step :math:`(t+1)`  until convergence.


Emission model 3: Mixture of Von-Mises Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a :math:`M`-dimensional data :math:`\mathbf{y}` the probability density function of von Mises-Fisher distribution is defined as following,

.. math::
	V_M(\mathbf{y}|\mathbf{v}_k,\kappa) = C_M(\kappa)exp(\kappa\mathbf{v}_k^{T}\mathbf{y})

where :math:`\mathbf{v}_k` denotes the mean direction for parcel k (a unit vector), :math:`\mathbf{y}` has unit length, :math:`\kappa` indicates the concentration parameter (:math:`\kappa\geqslant0`), which is joint over all parcels. :math:`C_M(\kappa) = \frac{\kappa^{M/2-1}}{(2\pi)^{M/2}I_{M/2-1}(\kappa)}` is the normalization constant where :math:`I_r(.)` refers to the modified Bessel function of the :math:`r` order. Thus, the *expected emission log-likelihood* of a mixture of :math:`K`-classes von-Mises fisher distributions is defined as:

.. math::
	\begin{align*}
	\mathcal{L}_E &=\langle \sum_i \log p(\mathbf{y}_i|\mathbf{u}_i;\theta_E)\rangle_q\\
	&=P\log C_M(\kappa)+\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle\kappa\mathbf{v}_{k}^T\mathbf{y}_i
	\end{align*}

If the design has repeated measurements of the same :math:`M` conditions, the user can specify this over the :math:`N \times M` design matrix  :math:`X` (:math:`N` is number of observation, :math:`M` is number of conditions). If we combine across the different repetitions, the resultant data would be :math:`\mathbf{y}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\tilde{\mathbf{y}}`, and then normalized. However, we can also treat the different repetitions as independent observations, meaning that the resultant data is normalized to length 1 for each of the :math:`J` independent partitions.  The likelihood is then the sum over partitions and voxels :

.. math::
	\begin{align*}
	\mathcal{L}_E=PJ\log C_N(\kappa)+\sum_{i}^P\sum_{j}^J\sum_{k}^K\langle u_{i}^{(k)}\rangle\kappa\mathbf{v}_{k}^T\mathbf{y}_{i,j}\\
	=PJ\log C_N(\kappa)+\sum_{i}^P\sum_{k}^K\langle u_{i}^{(k)}\rangle\kappa\mathbf{v}_{k}^T\sum_{j}^J\mathbf{y}_{i,j}
	\end{align*}

Effectively in the code, the user passes the unnormalized data, a design matrix, and a partition vector. We first compute :math:`\mathbf{y}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\tilde{\mathbf{y}}`  for each partition and then normalize the resultant data in each partition. Finally, we sum the vectors across partitions. :math:`\mathbf{y}_i = \sum_j \mathbf{y}_{i,j}`, and retain the number of observations for voxel i: :math:`J_i`. The resultant summed vectors are not length 1 anymore, but will be fine as a sufficient statistics.

Now, we update the parameters :math:`\theta` of the von-Mises mixture in the M-step by maximizing :math:`\mathcal{L}_E`  in respect to the parameters in vn-Mises mixture :math:`\theta_{k}=\{\mathbf{v}_{k},\kappa\}`. (Note: the updates only consider a single subject).

1. First, we calculated the weighted sum of the data vectors (:math:`\tilde{\mathbf{y}}`) in each subject and parcel, and the number of observations (:math:`\tilde{u}`) underlying that weighted sum.

.. math::
	\begin{align*}
	\tilde{\mathbf{y}}_{s,k} &= \sum_{i}^P\langle u_{s,i}^{(k)}\rangle_{q}\mathbf{y}_{s,i} \\
	\tilde{u}_{s,k} &= \sum_{i}^P\langle u_{s,i}^{(k)}\rangle_{q} J_{s,i}
	\end{align*}

2. In general, given these two estimates, we can updated the v parameter for the von-Mises Fisher distribution as follows:

.. math::
	\begin{align*}
	\tilde{\mathbf{v}} &= \frac{\tilde{\mathbf{y}}}{\tilde{u}} \\
	\mathbf{v}^{(t)} &= \frac{\tilde{\mathbf{v}}}{||\tilde{\mathbf{v}}||}\\
	\end{align*}

For the concentration parameter :math:`\kappa` updating is more difficult in particularly for high dimensional problems since it involves inverting ratio of two Bessel functions. Here we use approximate solutions suggested in (Banerjee et al., 2005) and (Hornik et al., 2014 "movMF: An R Package for Fitting Mixtures of von Mises-Fisher Distributions").

.. math::
	\begin{align*}
	r &= ||\tilde{\mathbf{v}}|| = \tilde{\mathbf{v}}^T\mathbf{v}\\
	\kappa^{(t)} &= \frac{\tilde{r}M-\tilde{r}^3}{1-\tilde{r}^2}
	\end{align*}

1. There are now different ways to integrated the sufficient statistics across subjects and parcels. For the estimation of the v-direction, we can either combine all the voxels across subjects in a fixed-effects analysis:

.. math::
	\tilde{\mathbf{v}}_k = \frac{\sum_s\tilde{\mathbf{y}}_{s,k}}{\sum_s\tilde{u}_{s,k}}

Or we can weight each subjects equally (note that this makes noisier subjects less important - just the number of assigned voxels per parcel does not matter anymore):

.. math::
	\tilde{\mathbf{v}}_k = \frac{1}{S} \sum_s{\frac{\tilde{\mathbf{y}}_{s,k}}{\tilde{u}_{s,k}}}

1. Finally for :math:`\kappa` estimation we have a number of options. In all of those, we want to take into account the deviation from the assumed mean direction (Vv). The most straightforward to understand is the subjects and parcel-specific kappa:

.. math::
	r_{s,k}=\tilde{\mathbf{v}}_{s,k}^{T}\mathbf{v}_k.

and for parcel-specific :math:`\kappa` :

.. math::
	\tilde{\mathbf{v}}_k = \frac{\sum_s\tilde{\mathbf{y}}_{s,k}}{\sum_s\tilde{u}_{s,k}}\\
	r_{k}=\tilde{\mathbf{v}}_{k}^{T}\mathbf{v}_k

for a subjects-specific kappa:

.. math::
	r_s=\frac{1}{K}\tilde{\mathbf{v}}_{s,k}^{T}\mathbf{v}_k

and for the overall kappa:

.. math::
	r=\frac{1}{K}\tilde{\mathbf{v}}_{k}^{T}\mathbf{v}_k



The update of kappa then follows point 2.


Model Evaluation
----------------

After model fitting, we need a fair way to quantitatively compare different emission models between a Gaussian mixture model (GMM), a Gaussian Mixture with exponential signal strength (GMM_exp), and a directional model (VMF). Unfortunately, the three models are defined in different space: the GMM and GMM_exp are defined in :math:`\mathbb{R}^N` state space while the VMF is defined in :math:`(N-1)`-hypersphere surface. Therefore, the traditional marginal log-likelihood based criterion (BIC, Bayes Factor) cannot provide a fair comparison, as the probability densities would cover different spaces. The main purpose of this section is trying to find evaluation criteria  that would be suitable to compare model defined in different space.

Comparing the true :math:`\mathbf{U}` and the inferred :math:`\hat{\mathbf{U}}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that these criteria only have value for simulations, for which we have the true parcellation :math:`\mathbf{U}`.

1. The absolute error between :math:`\mathbf{U}` and :math:`\hat{\mathbf{U}}`
*****************************************************************************

The first evaluation criterion is to calculate the absolute error between the true parcellation :math:`\mathbf{U}` and the expected :math:`\mathbf{\hat{U}}` which inferred on the training data. It defined as,

.. math::
	\bar{U}_{error}=\frac{\sum_i|\mathbf{u_i}-\langle \mathbf{u}_{i}\rangle_{q}|}{P}

where the :math:`\mathbf{u_i}` represents the true cluster label of :math:`i` and :math:`\langle \mathbf{u}_{i}\rangle_{q}` is the expected cluster label of brain location :math:`i` under the expectation :math:`q`. Both are multinomial encoded vectors. We can also replace the expectation with a the hard parcellation, again coded as a one-hot vector.

Note, this calculation of the mean absolute error is subject to the premutation of the parcellation, so that a loop over all possible permutations and find the minimum error is applied.



2. Normalized Mutual information (NMI) between :math:`\mathbf{U}` and :math:`\hat{\mathbf{U}}`
**********************************************************************************************

The second criteria is the normalized mutual information which examine the actual amount of "mutual information" between two parcellations :math:`\mathbf{U}` and :math:`\hat{\mathbf{U}}`.  A NMI value closes to 0 indicate two parcellations are largely independent, while values close to 1 indicate significant agreement. It defined as:

.. math::
	NMI(\mathbf{U},\mathbf{\hat{U}})=\frac{2\sum_{i=1}^{k_\mathbf{u}}\sum_{j=1}^{k_\mathbf{\hat{u}}}\frac{|\mathbf{u}=i|\cap|\mathbf{\hat{u}}=j|}{P}\log (P\frac{||\mathbf{u}=i|\cap|\mathbf{\hat{u}}=j||}{|\mathbf{u}=i|\cdot|\mathbf{\hat{u}}=j|})}{\sum_{i=1}^{k_\mathbf{u}}\frac{|\mathbf{u}=i|}{P}\log(\frac{|\mathbf{u}=i|}{P})+\sum_{j=1}^{k_{\mathbf{\hat{u}}}}\frac{|\hat{\mathbf{u}}=j|}{P}\log(\frac{|\hat{\mathbf{u}}=j|}{P})}

where :math:`k_{\mathbf{u}}=\{1,2,3,...,k\}` and :math:`k_{\mathbf{\hat{u}}}=\{1,2,3,...,k\}` represents the cluster labels of :math:`\mathbf{U}` and :math:`\hat{\mathbf{U}}` respectively. The term :math:`|\mathbf{u}=i|` and :math:`|\hat{\mathbf{u}}=j|` are the number of brain locations that belongs to cluster :math:`k_\mathbf{u}=i` in parcellation :math:`\mathbf{U}` or to cluster :math:`k_\mathbf{\hat{u}}=j` in :math:`\mathbf{\hat{U}}`, in other words, the terms :math:`\frac{|\mathbf{u}=i|}{P}` and :math:`\frac{|\mathbf{\hat{u}}=j|}{P}` represents the probability that a brain location picked at random from :math:`\mathbf{U}` falls into class :math:`k_{\mathbf{u}}=i`, or from :math:`\mathbf{\hat{U}}` falls into class :math:`k_{\mathbf{\hat{u}}}=j`.

Similarly, the :math:`||\mathbf{u}=i|\cap|\mathbf{\hat{u}}=j||` means the total number of a brain locations that both falls into classes :math:`k_{\mathbf{u}}=i` and :math:`k_{\mathbf{\hat{u}}}=j`. Note, the NMI calculation would not suffer from the permutation.



3. Adjusted rand index (ARI) between :math:`\mathbf{U}` and :math:`\hat{\mathbf{U}}`
************************************************************************************

The third one is the commonly used adjust rand index to test how similar the two given parcellations are. It defined as:

.. math::
	ARI(\mathbf{U},\mathbf{\hat{U}})=\frac{2\times(M_{11}M_{00}-M_{10}M_{01})}{(M_{00}+M_{10})(M_{10}+M_{11})+(M_{00}+M_{01})(M_{01}+M_{11})}

where :math:`M_{11}` corresponds to the number of pairs that are assigned to the same parcel in both :math:`\mathbf{U}` and :math:`\mathbf{\hat{U}}`, :math:`M_{00}` corresponds to the number of pairs that are assigned to different clusters in both :math:`\mathbf{U}` and :math:`\mathbf{\hat{U}}`, :math:`M_{10}` corresponds to the number of pairs that are assigned to the same parcel in :math:`\mathbf{U}`, but different parcels in :math:`\mathbf{\hat{U}}`, and :math:`M_{01}` corresponds to the number of pairs that are assigned to the same parcel in :math:`\mathbf{\hat{U}}`, but different parcels in :math:`\mathbf{\hat{U}}`.

Intuitively, :math:`M_{00}` and :math:`M_{11}` account for the agreement of parcellations, whereas :math:`M_{10}` and :math:`M_{01}` indicate their disagreement. Note, the ARI calculation would not suffer from the permutation.


Prediction error on independent test data (:math:`\mathbf{Y}_{test}`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One way to evaluate parcellation models is to test how well they can predict new test data. In general we have test data :math:`\mathbf{Y}_{test}`, a :math:`N \times P` matrix with *N* measurements (tasks, timepoints) and *P* brain locations (voxels, vertices) for each subject. Therefore, :math:`\mathbf{y}_{i}` is the response profile (a N-long vector) for each brain location :math:`i`. 

Because fMRI data (task activities or time series) have very different signal to noise levels across different voxels and subjects (and because the model does not necessarily predict the amplitude of responses), a natural evaluation criterion is the cosine error between predicted and observed data. 

.. math::
	\bar{\epsilon}_{cosine} = \frac{1}{P}\sum_i^P \left( 1-\frac{\hat{\mathbf{y}}_{i}^{T}\mathbf{y}_i}{||\hat{\mathbf{y}}_i||||\mathbf{y}_i||} \right)

For deriving a prediction from a probabilistic parcellation model, we have three options: 

Cosine error using a hard parcellation
**************************************
A simple evaluation is to set the prediction of the model to the response profile for the most likely parcel, :math:`\mathbf{v}_{\underset{k}{\operatorname{argmax}}}`. Assuming that the predicted response profiles are already length 1, the cosine error is then:

.. math::
	\bar{\epsilon}_{cosine} = \frac{1}{P}\sum_i^P \left( 1-{\mathbf{v}_\underset{k}{\operatorname{argmax}}}^{T}\frac{\mathbf{y}_i}{||\mathbf{y}_i||} \right)


Cosine Error for the average prediction
***************************************

Alternatively, we can set the prediction to the average of the response profiles across all the parcels, weighted by the probability that the voxels belongs to that parcel: 

.. math::
	\hat{\mathbf{y}}_{i} = \sum_k \hat{u}_i^{(k)}\mathbf{v}_k

where :math:`\hat{u}_i^{(k)}` is a probability that brain location *i* belongs to parcel *k*. This probabilistic parcellation should of course be estimated on independent training data.  

The entire cosine error is then: 

.. math::
	\bar{\epsilon}_{cosine} = \frac{1}{P}\sum_i^P \left( 1-\frac{ \left( \sum_k \hat{u}_i^{(k)}\mathbf{v}_k \right)^{T}\mathbf{y}_i}{|| \sum_k \hat{u}_i^{(k)}\mathbf{v}_k||\:||\mathbf{y}_i||} \right)

Expected cosine error
*********************

Rather than calculating the **cosine error for the average prediction** of the probabilistic model, we can also compute the **average cosine error across all possible predictions**. The expected cosine error is defined as: 

.. math::
	\begin{align*}
	\langle\bar{\epsilon}_{cosine}\rangle_q &= \frac{1}{P}\sum_i^P \left( 1-\frac{\left( \sum_k \hat{u}_i^{(k)}\mathbf{v}_k \right)^{T}\mathbf{y}_i}{||\mathbf{y}_i||} \right) \\\\
	&= \frac{1}{P}\sum_i \sum_k \hat{u}_i^{(k)} \left( 1-{\mathbf{v}_k}^{T}\frac{\mathbf{y}_i}{||\mathbf{y}_i||} \right)
	\end{align*}

When we compare the first expression of the expected cosine error with the expression of the cosine error for the average prediction, we conclude that the prediction term is normalized to unit length 
for the former (i.e. for each voxel :math:`i`, the sum across :math:`k` parcels is 1) whereas that is not the case for the latter.


Adjusted vs. non-adjusted cosine error
************************************** 

For all three types of cosine error mentioned so far, a possible problem is that a voxel which has very little signal count as much as a voxel with a lot of signal. To address this, we can change how we average the cosine error across different voxels. An interesting choice is to weight each error by the squared length of the data vector:

.. math::
	\begin{align*}
	\bar{\epsilon}_{Acosine} &= \frac{1}{\sum_i^P ||\mathbf{y}_i||^2} \sum_i^P ||\mathbf{y}_i||^2 \left( 1-\frac{\hat{\mathbf{y}}_{i}^{T}\mathbf{y}_i}{||\hat{\mathbf{y}}_i||\,||\mathbf{y}_i||} \right) \\\\
	&= \frac{1}{\sum_i^P ||\mathbf{y}_i||^2} \sum_i^P  \left( ||\mathbf{y}_i||^2-\frac{\hat{\mathbf{y}}_{i}^{T}\mathbf{y}_i ||\mathbf{y}_i||}{||\hat{\mathbf{y}}_i||} \right)
	\end{align*}

Weighting the error by the squared length of the vector effectively calculates the squared error between :math:`\mathbf{y}_i` and the prediction scaled to the amplitude of the data (:math:`\mathbf{v}_k\,||\mathbf{y}_i||`). For simplicity, we replace :math:`\mathbf{v}_k` by :math:`\mathbf{v}_i` to represent the predicted mean direction for voxel :math:`i` (already normalized to unit length). Therefore, :math:`1-R^2` between :math:`\mathbf{y}_i` and the prediction scaled to the amplitude of the data (:math:`\mathbf{v}_i\,||\mathbf{y}_i||`) is defined as:

.. math::
	\begin{align*}
	1-R^2 &= \frac{RSS}{TSS}\\
	&=\frac{1}{\sum_i||\mathbf{y}_i||^2}\sum_i (\mathbf{y}_i-\mathbf{v}_i||\mathbf{y}_i||)^2\\
	&=\frac{1}{\sum_i||\mathbf{y}_i||^2}\sum_i[(\mathbf{y}_i-\mathbf{v}_i||\mathbf{y}_i||)^T(\mathbf{y}_i-\mathbf{v}_k||\mathbf{y}_i||)]\\
	&=\frac{1}{\sum_i||\mathbf{y}_i||^2}\sum_i(\mathbf{y}_i^T\mathbf{y}_i-2\mathbf{y}_i^T\mathbf{v}_k||\mathbf{y}_i||+\mathbf{v}_i^T\mathbf{v}_i||\mathbf{y}_i||^2)\\
	&=\frac{1}{\sum_i||\mathbf{y}_i||^2}\sum_i(||\mathbf{y}_i||^2-2\mathbf{y}_i^T\mathbf{v}_i||\mathbf{y}_i||+||\mathbf{y}_i||^2)\\
	&=\frac{2}{\sum_i||\mathbf{y}_i||^2}\sum_i(||\mathbf{y}_i||^2-\mathbf{y}_i^T\mathbf{v}_i||\mathbf{y}_i||)
	\end{align*}

Adjusting for the length of the data vector can be done for any of the previously mentioned types of cosine error, i.e. for the hard-parcellation cosine error, the cosine error for the average prediction, and the expected cosine error.