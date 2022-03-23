# Generative Framework
A generative modelling framework for individual brain organization across a number of different data. The Model is partitioned into a model that determines the probability of the spatial arrangement of regions in each subject s, $p(\mathbf{U}^{(s)};\theta_A)$ and the probability of observing a set of data at each given brain location. We introduce the Markov property that the observations are mutually independent, given the spatial arrangement. 
$$
p(\mathbf{Y}^{(s)}|\mathbf{U}^{(s)};\theta_E)=\prod_i p(\mathbf{y}_i^{(s)}|\mathbf{u}_i^{(s)};\theta_E)
$$

### Inference and learning 

We will learn the model, by maximizing the ELBO (Evidence lower bound). For clarity, I am dropping the index for the subject (s) for now, 
$$
\begin{align*}
\log p(\mathbf{Y} | \theta)
&=\log\sum_{\mathbf{U}}p(\mathbf{Y},\mathbf{U}|\theta) \\ 
&=\log\sum_{\mathbf{U}}q(\mathbf{U})\frac{p(\mathbf{Y},\mathbf{U}|\theta)}{q(\mathbf{U})}\\
&\geqslant \sum_{\mathbf{U}} q(\mathbf{U}) \log \frac{p(\mathbf{Y},\mathbf{U}|\theta)}{q(\mathbf{U})} \tag{Jensen's inequality}\\ 
&=\langle \log p(\mathbf{Y},\mathbf{U}|\theta) - \log q(\mathbf{U})\rangle_q
\triangleq \mathcal{L}(q, \theta) - \log \langle q(\mathbf{U})\rangle_q
\end{align*}
$$

Given the markov property, we can break the expected complete log likelihood into two pieces, one containing the parameters for the arrangement model and one containing the parameters for the emission model. 

$$
\begin{align*}
\langle \log p(\mathbf{Y},\mathbf{U}|\theta)\rangle_q &=\langle \log(p(\mathbf{Y}|\mathbf{U};\theta_E) p(\mathbf{U}|\theta_A))\rangle_q\\
&=\langle \log p(\mathbf{Y}|\mathbf{U};\theta_E)\rangle_q + \langle \log p(\mathbf{U}|\theta_A)\rangle_q\\
&\triangleq \mathcal{L}_E+\mathcal{L}_A
\end{align*}
$$

We will refer to the first term as the expected emission log-likelihood and the second term as the expected arrangement log-likelihood. We can estimate the parameters of the emission model from maximizing the  expected emission log-likelihood, and we can estimate the parameters of the arrangement model by maximizing the expected arrangement log-likelihood.  

## Arrangement models

This is a generative Potts model of brain activity data. The main idea is that the brain consists of $K$ regions, each with a specific activity profile $\mathbf{v}_k$ for a specific task set. The model consists of a arrangement model that tells us how the $K$ regions are arranged in a specific subject $s$, and an emission model that provides a probability of the measured data, given the individual arrangement of regions.

### Independent Arrangement model 
This is the simplest spatial arrangement model - it simply learns the probability at each location that the node is part of cluster K. These probabilities are simply learned as the parameters $\pi_{ik}=p(u_i=k)$, or after a re-parameterization in log space: $\eta_{ik}=\log \pi_{ik}$.  Vice versa (not assuming that the etas are correctly scaled): 
$$
\pi_{ik}=\frac{\rm{exp}(\eta_{ik})}{\sum_{j}\rm{exp}(\eta_{ij})}
$$

This independent arrangement model can be estimated using the EM-algorithm.

In the Estep, we are integrating the evidence from the data and the prior: 
$$
p(u_i=k|\mathbf{y}_i)=\langle u_{ik}\rangle=\frac{\rm{exp}(log(p(\mathbf{y}_i|u_i=k))+\eta_{ik})}{\sum_{j}{\rm{exp}(log(p(\mathbf{y}_i|u_i=j))+\eta_{ij}})}
$$

Or in vector notation:

$$
\begin{align*}
\langle \mathbf{u}_{i}\rangle =\rm{softmax}(log(p(\mathbf{y}_i|\mathbf{u}_i))+\boldsymbol{\eta}_i)
\end{align*}
$$
For the M-step, we use the derivative of the expected arrangement log-likelihood in respect to the parameters $\eta$:

$$
\mathcal{L}_A=\sum_{i}\sum_{k}\langle u_{i,k}\rangle  \rm{log}(\pi_{ik})\\
=\sum_{i}\sum_{k}\langle u_{ik}\rangle(\eta_{ik}-\rm{log}\sum_j\rm{exp}(\eta_{ij}))\\
=\sum_{i}\sum_{k}\langle u_{ik}\rangle\eta_{ik}-\sum_{i}\log\sum_j\exp(\eta_{i,j})\\
$$

So the derivative is 

$$
\begin{align}
\frac{\part\mathcal{L}_A}{\part{\eta_{ik}}}&=\langle u_{ik}\rangle-\frac{\part}{\part\eta_{ik}}\log\sum_j\exp(\eta_{ij})\\
&=\langle u_{ik}\rangle-\frac{1}{\sum_j\exp(\eta_{ij})}\frac{\part}{\part\eta_{ik}}\sum_j\exp(\eta_{ij})\\\
&=\langle u_{ik}\rangle-\frac{\exp(\eta_{ik})}{\sum_j\exp(\eta_{ij})}\\
&=\langle u_{ik}\rangle-\pi_{ik}
\end{align}
$$



We can also get the same result directly by the application of chain rule: For a good introduction, see: [https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/].  
$$
\frac{\part{\pi_k}}{\eta_k}=\pi_k(\delta_{kj}-\pi_j)
$$



### Simple Potts model

The brain is sampled in $P$ vertices (or voxels). Individual maps are aligned using anatomical normalization, such that each vertex refers to a (roughly) corresponding region in each individual brain. The assignment of each brain location to a specific parcel in subject $s$ is expressed as the random variable $u_i^{(s)}$.

Across individual brains, we have the overall probability of a specific brain location being part of parcel $k$.
$$
p(u_i = k) = \pi_{ki}
$$

The spatial interdependence of brain locations is expressed as a Potts model. In this model, the overall probability of a specific assignment of brain locations to parcels (the vector $\mathbf{u}$) is expressed as the product of the overall prior and the product of all possible pairwise potentenials ($\psi_{ij}$). 

$$
p(\mathbf{u}) = \frac{1}{Z(\theta)} \prod_{i}\pi_{u_i,i}\prod_{i\neq j}{\psi_{ij}(u_i,u_j) }
$$

Each local potential is defined by an exponential over all other that are connected to node $i$, i.e. nodes with connectivity weights of $w_{ji}=w_{ij}>0$.

$$
\psi_{ij}=  \rm{exp}(\theta_{w}\mathbf{u}_i^T\mathbf{u}_j w_{ij})
$$

Where we have introduced a one-hot encoding of $u_i$ with a $K$ vector of indicator variables $\mathbf{u}_i$ , such that $\mathbf{u}_i^T\mathbf{u}_j = 1$ if $u_i = u_j$ and $0$ otherwise.

The spatial co-dependence across the entire brain is therefore expressed with the pairwise weights $w$ that encode how likely two nodes belong to the same parcel. The temperature parameter $\theta_w$ determines how strong this co-dependence overall influences the local probabilies (relative to the prior). We can use this notation to express local co-dependencies by using a graph, where we define 
$$
w_{ij}=\begin{cases}
1; \text{if i and j are neighbours}\\
0; \text{otherwise}
\end{cases}
$$
This formulation would enforce local smoothness of the map. However, we could also express in these potential more medium range potentials (two specific parietal and premotor areas likely belong to the same parcel), as well as cross-hemispheric symmetry. Given this, the matrix $\mathbf{W}$ could be simply derived from the underlying grid or be learned to reflect known brain-connectivity. 

The expected arrangement log-likelihood therefore becomes: 
$$
\begin{align*}
\mathcal{L}_A=\sum_i \langle\mathbf{u}_i\rangle^T \log{\boldsymbol{\pi}_{i}}+\theta_w \sum_i \sum_j w_{ij} \langle\mathbf{u}_i^T\mathbf{u}_j\rangle - \log Z 
\end{align*}
$$

####  Inference using stochastic maximum likelihood / contrastive divergence

We can approximate the gradient of the parameters using a contrastive divergence-type algorithm. We view the arrangement log likelihood as a sum of the unnormalized part ($\tilde{\mathcal{L}}_A$) and the log partition function. For each parameter $\theta$ we then follow the gradient 
$$
\begin{align*}
\grad_\theta \mathcal{L}_A&=\grad_\theta \tilde{\mathcal{L}}_A-\grad_\theta \log Z\\
&=\grad_\theta \tilde{\mathcal{L}}_A-\mathrm{E}_p [\grad_\theta \tilde{\mathcal{L}}_A]\\
&=\grad_\theta \langle \log \tilde{p}(\mathbf{U}|\theta)\rangle_q -
\grad_\theta \langle \log \tilde{p}(\mathbf{U}|\theta)\rangle_p 
\end{align*}
$$
Thus, we can use the gradient of the unnormalized expected log-likelihood (given a distribution $q(\mathbf{U}) = p(\mathbf{U}|\mathbf{Y};\theta)$, minus the  gradient of the unnormalized expected log-likelihood in respect to the expectation under the model parameters, without seeing the data, $q(\mathbf{U}) = p(\mathbf{U}|\mathbf{Y};\theta)$. This motivates the use of sampling / approximate inference for both of these steps. See Deep Learning (18.1).

#### Sampling from the prior or posterior distribution

The problem is that the two expectations under the prior (p) and the posterior (q) distribution of the model cannot be easily be computed. We can evaluate the prior probability of a parcellation $p(\mathbf{U})$ or the posterior distribution $p(\mathbf{U}|\mathbf{Y})$ up to a constant of proportionality, with for example 
$$
p(\mathbf{U}|\mathbf{Y};\theta) = \frac{1}{Z(\theta)}\prod_{i}\mu_{u_i,i}\prod_{i\neq j}{\psi_{ij}(u_i,u_j) }\prod_{i}p(\mathbf{y}_i|u_i)
$$
Calculating the normalization constant $Z(\theta)$ (partition function, Zustandssumme, or sum over states) would involve summing this probability over all possible states, which for $P$ brain locations and $K$ parcels is $K^P$, which is intractable. 

However, the conditional probability for each node, given all the other nodes, can be easily computed. Here the normalizaton constant is just the sum of the potential functions over the $K$ possible states for this node

$$
p(u_i|u_{j \neq i},\mathbf{y}_i;\theta) = \frac{1}{Z(\theta)}\mu_{u_i,i} \; p(\mathbf{y}_i|u_i) \prod_{i\neq j}{\psi_{ij}(u_i,u_j) }
$$
With Gibbs sampling, we start with a pattern $\mathbf{u}^{(0)}$ and then update $u_1^{(1)}$ by sampling from $p(u_1|u_2^{(0)}...u_P^{(0)})$. We then sample $u_2^{(1)}$ by sampling from $p(u_2|u_1^{(1)}, u_3^{(0)}...u_P^{(0)})$ and so on, until we have sampled each node once. Then we return to the beginning and restart the process. After some burn-in period, the samples will come from desired overall distribution. If we want to sample from the prior, rather than from the posterior, we simply drop the $p(\mathbf{y}_i|u_i)$ term from the conditional probability above. 

#### Gradient for different parametrization of the Potts model

For the edge-energy parameters $\theta_w$ we clearly want to use the natural parametrization with the derivate: 
$$
\frac{\part \tilde{\mathcal{L}}_A}{\part \theta_w}=\sum_i\sum_j w_{ij}\langle\mathbf{u}_i^T\mathbf{u}_j\rangle
$$
For the prior probability of each parcel $k$ at each location $i$  ($\pi_{ik}$) we have a number of options. 

First ,we can use the probabilities themselves as parameters:
$$
\frac{\part \tilde{\mathcal{L}}_A}{\part \pi_{ik}}=\frac{\langle u_{ik}\rangle}{\pi_{ik}}
$$

This is unconstrained (that is probabilities do not need to sum to 1), and the normalization would happen through the partition function. 

Secondly, we can use a re-parameterization in log space, which is more natural: $\eta_{ik}=\log \pi_{ik}$. In this case the derivative of the non-normalized part just becomes: 
$$
\frac{\part \tilde{\mathcal{L}}_A}{\part \eta_{ik}}=\langle u_{ik}\rangle
$$
Finally, we can implement the constraint that the probabilities at each location sum to one by the following re-parametrization:
$$
\begin{align*}
\pi_{iK}&=1-\sum_{k=1}^{K-1}\pi_{ik}\\
\eta_{ik}&=\log(\frac{\pi_{ik}}{\pi_{iK}})=\log{\pi_{ik}}-\log({1-\sum_{k=1}^{K-1}\pi_{ik}})\\
\pi_{ik}&=\frac{\exp(\eta_{ik})}{1+\sum_{k=1}^{K-1}\exp(\eta_{ik})}\\
\pi_{iK}&=\frac{1}{1+\sum_{k=1}^{K-1}\exp(\eta_{ik})}
\end{align*}
$$
In the implementation, we can achieve this parametrization easily by defining a non-flexible parameter $\eta_{iK}\triangleq0$. Then we can treat the last probability like all the other ones. 

With this constrained parameterization, we can rewrite the unnormalized part of the expected log-likelihood as: 

$$
\begin{align*}
\tilde{\mathcal{L}}_A&=\sum_i \sum_{k}^{K-1}\langle u_{ik}\rangle  \log \pi_{ik}+[1-\sum_{k}^{K-1}\langle u_{ik}\rangle]\log{\pi_{iK}}+C\\
&=\sum_i \sum_{k}^{K-1}\langle u_{ik}\rangle  (\log \pi_{ik}-\log \pi_{iK})+\log{\pi_{iK}}+C\\
&=\sum_i \sum_{k}^{K-1}\langle u_{ik}\rangle \eta_{ik}-\log(1+\sum_{k=1}^{K-1}\exp(\eta_{ik}))+C\\
\end{align*}
$$
where C is the part of the normalized log-likelihood that does not depend on $\pi$. Taking derivative in respect to $\eta_{ik}$ yields: 
$$
\begin{align*}
\frac{\part\tilde{\mathcal{L}}_A}{\part\eta_{ik}}&=\langle u_{ik}\rangle - \frac{\exp(\eta_{ik})}{1+\sum_k^{K-1}\exp(\eta_{ik})}\\
&=\langle u_{ik}\rangle - \pi_{ik}
\end{align*}
$$
So in this parameterization in the iid case, $Z=1$ and we don't need the negative step. In general, however, we cannot simply set the above derivative to zero and solve it, as the parameter $\theta_w$ will also have an influences on $\langle u_{ik} \rangle$.

## Emission models

Given the Markov property, the emission models specify the log probability of the observed data as a function of $\mathbf{u}$.  

$$
\log p(\mathbf{Y}|\mathbf{U};\theta_E)=\sum_i \log p(\mathbf{y}_i|\mathbf{u}_i;\theta_E)
$$

Furthermore, assuming that $\mathbf{u}_i$ is a one-hot encoded indicator variable (parcellation model), we can write the expected emission log-likelihood as: 
$$
\langle \log p(\mathbf{Y}|\mathbf{U};\theta_E)\rangle =\sum_i \sum_k \langle u_i^{(k)}\log p(\mathbf{y}_i|u_i=k;\theta_E) \rangle
$$
In the E-step the emission model simply passes $p(\mathbf{y}_i|\mathbf{u}_i;\theta_E)$ as a message to the arrangement model. In the M-step, $q(\mathbf{u}_i) = \langle \mathbf{u}_i \rangle$ is passed back, and the emission model optimizes the above quantity in respect to $\theta_E $.


### Emission model 1: Mixture of Gaussians

Under the Gaussian mixture model, we model the emissions as a Gaussian with a parcel-specific mean ($\mathbf{v}_k$), and with equal isotropic variance across parcels and observations: 
$$
p(\mathbf{y_i}|u^{(k)};\theta_E) = \frac{1}{(2\pi)^{N/2}(\sigma^{2})^{N/2}}\rm{exp}\{-\frac{1}{2\sigma^{2}}(y_{i}-\mathbf{v}_k)^T(y_{i}-\mathbf{v}_k)\}
$$
The expected emission log-likelihood therefore is: 
$$
\begin{align*}
\mathcal{L}_E=\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}[-\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}(\mathbf{y}_{i}-\mathbf{v}_{k})^T(\mathbf{y}_{i}-\mathbf{v}_{k})]\\
=-\frac{PN}{2}\log(2\pi)-\frac{PN}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}[(\mathbf{y}_{i}-\mathbf{v}_{k})^T(\mathbf{y}_{i}-\mathbf{v}_{k})]
\end{align*}
$$
Now, with the above expected emission log likelihood by hand, we can update the parameters $\theta_E = \{\mathbf{v}_1,...,\mathbf{v}_K,\sigma^2\}$ in the $\Mu$ step. 

1. Updating $\mathbf{v}_k$, we take derivative of *expected emission log likelihood* with respect to $\mathbf{v}_{k}$ and set it to 0:
   $$
   \frac{\partial \mathcal{L}_E}{\partial \mathbf{v}_{k}} =\frac{1}{\sigma^{2}}\sum_{i}\langle u_{i}^{(k)}\rangle_{q}(\mathbf{y}_{i}-\mathbf{v}_{k}) = 0
   $$
   Thus, we get the updated $\mathbf{v}_{k}$ in current $\Mu$ step as, 
   $$
   \mathbf{v}_{k}^{(t)} = \frac{\sum_{i}\langle u_{i}^{(k)}\rangle_{q}^{(t)}\mathbf{y}_{i}}{\sum_{i}\langle u_{i}^{(k)}\rangle_{q}^{(t)}}
   $$

2. Updating $\sigma^{2}$, we take derivative of *expected emission log likelihood* $\mathcal{L}(q, \theta)$ with respect to $\sigma^{2}$  and set it to  0: 
   $$
   \frac{\partial \mathcal{L}_E}{\partial \sigma^{2}} =-\frac{PN}{2\sigma^{2}}+\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}[\frac{1}{2\sigma^{4}}(\mathbf{y}_{i}-\mathbf{v}_{k}^{(t)})^T(\mathbf{y}_{i}-\mathbf{v}_{k}^{(t)})] = 0
   $$
   Thus, we get the updated $\sigma^{2}$ for parcel $k$ in the current $\Mu$ step as,
   $$
   {\sigma^{2}}^{(t)} = \frac{1}{PN}\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}^{(t)}(\mathbf{y}_{i}-\mathbf{v}_{k}^{(t)})^T(\mathbf{y}_{i}-\mathbf{v}_{k}^{(t)})
   $$
   where $P$ is the total number of voxels $i$.

   The updated parameters $\theta_{k}^{(t)}$ from current $\mathbf{M}$-step will be passed to the next $\mathbf{E}$-step $(t+1)$  until convergence.

### Emission model 2: Mixture of Gaussians with Exponential signal strength

The emission model should depend on the type of data that is measured. A common application is that the data measured at location $i$ are the task activation in $N$ tasks, arranged in the $N\times1$ data vector $\mathbf{y}_i$. The averaged expected response for each of the parcels is $\mathbf{v}_k$. One issue of the functional activation is that the signal-to-noise ratio (SNR) can be quite different across different participants, and voxels, with many voxels having relatively low SNR. We model this signal to noise for each brain location (and subject) as $s_i \sim exponential(\beta_s)$. Therefore the probability model for gamma is defined as:
$$
p(s_i|\theta) = \beta e^{-\beta s_i}
$$
Overall, the expected signal at each brain location is then 
$$
\rm{E}(\mathbf{y}_i)=\mathbf{u}_i^T \mathbf{V}s_i
$$


Finally, relative to the signal, we assume that the noise is distributed i.i.d Gaussian with: 
$$
\boldsymbol{\epsilon}_i \sim Normal(0,\mathbf{I}_K\theta_{\sigma s})
$$
 Here, the proposal distribution $q(u_{i}^{(k)},s_{i}|\mathbf{y}_{i})$ is now a multivariate distribution across $u_i$ and $s_i$. Thus, the *expected emission log likelihood* $\mathcal{L}_E(q, \theta)$ is defined as:
$$
\begin{align*}
\mathcal{L}_E &= \langle\sum_i\log p(\mathbf{y}_i, s_i|u_i; \theta_E)\rangle_{q}\\
&=\sum_{i}\sum_{k}\langle u_{i}^{(k)}[-\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}(\mathbf{y}_{i}-\mathbf{v_k}s_i)^T(\mathbf{y}_{i}-\mathbf{v_k}s_i)]\rangle_{q}  \\ &+\sum_{i}\sum_{k}\langle u_{i}^{(k)}[ \log \beta-\beta s_i] \rangle_q\\

&=-\frac{NP}{2}\log(2\pi)-\frac{NP}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i}\sum_{k}\langle u_{i}^{(k)}(\mathbf{y}_{i}-\mathbf{v_k}s_i)^T(\mathbf{y}_{i}-\mathbf{v_k}s_i)\rangle_{q} \\ &+ P\log\beta-\sum_{i}\sum_k\beta\langle u_{i}^{(k)} s_i\rangle_q\\

&=-\frac{NP}{2}\log(2\pi)-\frac{NP}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i} \mathbf{y}_i^T\mathbf{y}_i-\frac{1}{2\sigma^{2}}\sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q}) \\ &+\log\beta-\beta \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q
\end{align*}
$$
Now, we can update the parameters $\theta$ of the Gaussians/Exponential mixture in the $\Mu$ step. The parameters of the gaussian mixture model are $\theta_{E} = \{\mathbf{v}_{1},...,\sigma^{2},\beta\}$ . 

1. We start with updating the $\mathbf{v}_k$ (Note: the updates only consider a single subject). We take the derivative of *expected emission log likelihood* $\mathcal{L}_E$ with respect to $\mathbf{v}_{k}$ and make it equals to 0 as following:
   $$
   \frac{\partial \mathcal{L}_E}{\partial \mathbf{v}_{k}} =-\frac{1}{\sigma^{2}}\sum_{i}-\mathbf{y}_{i}^{T}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\langle u_{i}^{(k)}s_{i}^{2}\rangle_{q} = 0
   $$
   Thus, we get the updated $\mathbf{v}_{k}$ in current $\Mu$ step as, 
   $$
   \mathbf{v}_{k}^{(t)} = \frac{\sum_{i}\langle u_{i}^{(k)}s_{i}\rangle_{q}^{(t)}\mathbf{y}_{i}}{\sum_{i}\langle u_{i}^{(k)}s_{i}^{2}\rangle_{q}^{(t)}}
   $$

2. Updating $\sigma^{2}$ , we take derivative of with respect to $\sigma^{2}$ and set it equals to 0 as following:
   $$
   \frac{\partial \mathcal{L}_E}{\partial \sigma^{2}} =-\frac{NP}{2\sigma^2}+\frac{1}{2\sigma^{4}}\sum_{i}\mathbf{y}_{i}^T\mathbf{y}_{i}+\frac{1}{2\sigma^{4}}\sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q}) = 0
   $$
   Thus, we get the updated $\sigma^{2}$ for parcel $k$ in the current $\Mu$ step as,
   $$
   {\sigma^2}^{(t)} = \frac{1}{NP}(\sum_{i}\mathbf{y}_i^T\mathbf{y}_i+
   \sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q})
   $$
   
3. Updating $\beta$, we take derivative of  $\mathcal{L}_E(q, \theta)$ with respect to $\beta$ and set it equal to 0 as following:


$$
\begin{align*}
\frac{\partial \mathcal{L}_E}{\partial \beta} &=\frac{\partial [P\log\beta-\beta \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q]}{\partial \beta} \\
&= \frac{P\alpha}{\beta}-\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q = 0
\end{align*}
$$

​	   Thus, we get the updated $\beta_{k}$ in current $\Mu$ step as, 
$$
\beta_{k}^{(t)} =  	\frac{P}{\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q}
$$


### Emission model 2b: Mixture of Gaussians with Gamma signal strength

The emission model should depend on the type of data that is measured. A common application is that the data measured at location $i$ are the task activation in $N$ tasks, arranged in the $N\times1$ data vector $\mathbf{y}_i$. The averaged expected response for each of the parcels is $\mathbf{v}_k$. One issue of the functional activation is that the signal-to-noise ratio (SNR) can be quite different across different participants, and voxels, with many voxels having relatively low SNR. We model this signal to noise for each brain location (and subject) as $s_i \sim Gamma(\theta_\alpha,\theta_{\beta s})$. Therefore the probability model for gamma is defined as:
$$
p(s_i|\theta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)}s_i^{\alpha-1}e^{-\beta s_i}
$$
Overall, the expected signal at each brain location is then 
$$
\rm{E}(\mathbf{y}_i)=\mathbf{u}_i^T \mathbf{V}s_i
$$


Finally, relative to the signal, we assume that the noise is distributed i.i.d Gaussian with: 
$$
\boldsymbol{\epsilon}_i \sim Normal(0,\mathbf{I}_K\theta_{\sigma s})
$$
 Here, the proposal distribution $q(u_{i}^{(k)},s_{i}|\mathbf{y}_{i})$ is now a multivariate distribution across $u_i$ and $s_i$. Thus, the *expected emission log likelihood* $\mathcal{L}_E(q, \theta)$ is defined as:
$$
\begin{align*}
\mathcal{L}_E &= \langle\sum_i\log p(\mathbf{y}_i, s_i|u_i; \theta_E)\rangle_{q}\\
&=\sum_{i}\sum_{k}\langle u_{i}^{(k)}[-\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}(\mathbf{y}_{i}-\mathbf{v_k}s_i)^T(\mathbf{y}_{i}-\mathbf{v_k}s_i)]\rangle_{q}  \\ &+\sum_{i}\sum_{k}\langle u_{i}^{(k)}[\alpha \log \beta-\log\Gamma(\alpha)+(\alpha-1)\log s_i-\beta s_i] \rangle_q\\

&=-\frac{NP}{2}\log(2\pi)-\frac{NP}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i}\sum_{k}\langle u_{i}^{(k)}(\mathbf{y}_{i}-\mathbf{v_k}s_i)^T(\mathbf{y}_{i}-\mathbf{v_k}s_i)\rangle_{q} \\ &+ P\alpha\log\beta-P\log\Gamma(\alpha)+\sum_{i}\sum_k\langle u_{i}^{(k)}(\alpha-1)\log s_i-u_{i}^{(k)}\beta s_i\rangle_q\\

&=-\frac{NP}{2}\log(2\pi)-\frac{NP}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i} \mathbf{y}_i^T\mathbf{y}_i-\frac{1}{2\sigma^{2}}\sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q}) \\ &+P\alpha\log\beta-P\log\Gamma(\alpha)+(\alpha-1)\sum_{i}\sum_k \langle u_{i}^{(k)}\log s_i\rangle_q-\beta \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q
\end{align*}
$$
Now, we can update the parameters $\theta$ of the Gaussians/Gamma mixture in the $\Mu$ step. The parameters of the gaussian mixture model are $\theta_{E} = \{\mathbf{v}_{1},...,\sigma^{2},\alpha,\beta\}$ . 

1. We start with updating the $\mathbf{v}_k$ (Note: the updates only consider a single subject). We take the derivative of *expected emission log likelihood* $\mathcal{L}_E$ with respect to $\mathbf{v}_{k}$ and make it equals to 0 as following:
   $$
   \frac{\partial \mathcal{L}_E}{\partial \mathbf{v}_{k}} =-\frac{1}{\sigma^{2}}\sum_{i}-\mathbf{y}_{i}^{T}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\langle u_{i}^{(k)}s_{i}^{2}\rangle_{q} = 0
   $$
   Thus, we get the updated $\mathbf{v}_{k}$ in current $\Mu$ step as, 
   $$
   \mathbf{v}_{k}^{(t)} = \frac{\sum_{i}\langle u_{i}^{(k)}s_{i}\rangle_{q}^{(t)}\mathbf{y}_{i}}{\sum_{i}\langle u_{i}^{(k)}s_{i}^{2}\rangle_{q}^{(t)}}
   $$

2. Updating $\sigma^{2}$ , we take derivative of with respect to $\sigma^{2}$ and set it equals to 0 as following:
   $$
   \frac{\partial \mathcal{L}_E}{\partial \sigma^{2}} =-\frac{NP}{2\sigma^2}+\frac{1}{2\sigma^{4}}\sum_{i}\mathbf{y}_{i}^T\mathbf{y}_{i}+\frac{1}{2\sigma^{4}}\sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q}) = 0
   $$
   Thus, we get the updated $\sigma^{2}$ for parcel $k$ in the current $\Mu$ step as,
   $$
   {\sigma^2}^{(t)} = \frac{1}{NP}(\sum_{i}\mathbf{y}_i^T\mathbf{y}_i+
   \sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q})
   $$
   
3. Updating $\beta$, we take derivative of  $\mathcal{L}_E(q, \theta)$ with respect to $\beta$ and set it equal to 0 as following:


$$
\begin{align*}
\frac{\partial \mathcal{L}_E}{\partial \beta} &=\frac{\partial [P\alpha\log\beta-\beta \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q]}{\partial \beta} \\
&= \frac{P\alpha}{\beta}-\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q = 0
\end{align*}
$$

​	   Thus, we get the updated $\beta_{k}$ in current $\Mu$ step as, 
$$
\beta_{k}^{(t)} =  	\frac{P\alpha_k^{(t)}}{\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q}
$$

4. Updating $\alpha_{k}$ is comparatively hard since we cannot derive closed-form, we take derivative of  $\mathcal{L}_E(q, \theta)$ with respect to $\alpha$ and make it equals to 0 as following:

$$
\begin{align*}
\frac{\partial \mathcal{L}_E}{\partial \alpha} &=\frac{\partial [P\alpha\log\beta-P\log\Gamma(\alpha)+(\alpha-1)\sum_{i}\sum_k \langle u_{i}^{(k)}\log{s_i}\rangle_q]}{\partial \alpha}\\
&=P\log\beta-P\frac{\Gamma'(\alpha)}{\Gamma(\alpha)}+\sum_{i}\sum_k \langle u_{i}^{(k)} \log {s_i}\rangle_q = 0
\end{align*}
$$
The term $\frac{\Gamma'(\alpha)}{\Gamma(\alpha)}$ in above equation is exactly the *digamma function* and we use $\digamma(\alpha)$ to represent. Also from (4), we know $\beta=\frac{P\alpha_k^{(t)}}{\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q}$ Thus, we get the updated $\alpha$ in current $\Mu$ step as, 
$$
\begin{align*}
\digamma(\alpha)^{(t)} &= \log \frac{P\alpha_k^{(t)}}{\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q}+\frac{1}{P}\sum_{i}\sum_k \log \langle u_{i}^{(k)}s_i\rangle_q\\
&=\log P\alpha^{(t)} - \log \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q + \frac{1}{P}\sum_{i}\sum_k\log\langle u_{i}^{(k)}s_i\rangle_q
\end{align*}
$$
By applying "generalized Newton" approximation form, the updated $\alpha$ is as follows: 
$$
\alpha^{(t)} \approx \frac{0.5}{\log \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q - \frac{1}{P}\sum_{i}\sum_k\langle u_{i}^{(k)} \log{s_i}\rangle_q}
$$
Note that $\log \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q \geqslant \frac{1}{P}\sum_{i}\sum_k\log\langle u_{i}^{(k)}s_i\rangle_q$ is given by Jensen's inequality.

### Emission model 3: Mixture of Von-Mises Distributions

For a $N$-dimensional data $\mathbf{y} \in $ the probability density function of von Mises-Fisher distribution is defined as following, 
$$
V_N(\mathbf{y}|\mathbf{v}_k,\kappa) = C_N(\kappa)exp(\kappa\mathbf{v}_k^{T}\mathbf{y})
$$
where $\mathbf{v}_k$ denotes the mean direction (unit vectors for each parcels), $\kappa$ indicates the concentration parameter ($\kappa\geqslant0$), which is joint over all parcels. $C_N(\kappa) = \frac{\kappa^{N/2-1}}{(2\pi)^{N/2}I_{N/2-1}(\kappa)}$ is the normalization constant where $I_r(.)$ refers to the modified Bessel function of the $r$ order. Thus, the *expected emission log-likelihood* of a mixture of $K$-classes von-Mises fisher distributions is defined as:
$$
\begin{align*}
\mathcal{L}_E &=\langle \sum_i \log p(\mathbf{y}_i|\mathbf{u}_i;\theta_E)\rangle_q\\
&=P\log C_N(\kappa)+\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle\kappa{\mu^{(k)}}^{T}\mathbf{y}_i
\end{align*}
$$
Now, we update the parameters $\theta$ of the von-Mises mixture in the $\Mu$ step by maximizing $\mathcal{L}_E$  in respect to the parameters in von-Mises mixture $\theta_{k}=\{\mathbf{v}_{k},\kappa\}$. (Note: the updates only consider a single subject).

1. Updating mean direction $\mathbf{v}_k$, we take derivative in respect to $\mathbf{v}_{k}$ and set it to 0. Thus, we get the updated $\mathbf{v}_{k}$ in current $\Mu$ step as, 
   $$
   \begin{align*}
   
   \mathbf{v}_{k}^{(t)} &=\frac{\bar{\mathbf{y}}_k}{r_k}, \;\;\;\;\;\;\text{where}\;\; \bar{\mathbf{y}}_{k} = \sum_{i}\langle u_{i}^{(k)}\rangle_{q}\mathbf{y}_{i}; \;\; r_k=||\bar{\mathbf{y}}_{k}||
   
   \end{align*}
   $$

2. Updating concentration parameter $\kappa$ is difficult in particularly for high dimensional problems since it involves inverting ratio of two Bessel functions. Here we use approximate solutions suggested in (Banerjee et al., 2005): 

$$
\kappa_k^{(t)} \approx \frac{\overline{r}_kN-\overline{r}_k^3}{1-\overline{r}_k^2}\\
\bar{r}_k=\frac{r_k}{\sum_i \langle u_i^{(k)} \rangle_q}
$$

The updated parameters from current $\mathbf{M}$-step will be passed to the $\mathbf{E}$-step of $(t+1)$ times for calculating the expectation.

## Model Evaluation

After model fitting, we need a fair way to quantitatively compare different emission models between a Gaussian mixture model (GMM), a Gaussian Mixture with exponential signal strength (GMM_exp), and a directional model (VMF). Unfortunately, the three models are defined in different space: the GMM and GMM_exp are defined in $\mathbb{R}^N$ state space while the VMF is defined in $(N-1)$-hypersphere surface. Therefore, the traditional marginal log-likelihood based criterion (BIC, Bayes Factor) cannot provide a fair comparison, as the probability densities would cover different spaces. The main purpose of this section is trying to find evaluation criteria  that would be suitable to compare model defined in different space.

### Comparing the true $\mathbf{U}$ and the inferred $\hat{\mathbf{U}}$

Note that these criteria only have value for simulations, for which we have the true 

#### 1. the absolute error between $\mathbf{U}$ and $\hat{\mathbf{U}}$

the first evaluation criterion is to calculate the absolute error between the true parcellation $\mathbf{U}$ and the expected $\mathbf{\hat{U}}$ which inferred on the training data. It defined as,
$$
\bar{U}_{error}=\frac{\sum_i|\mathbf{u_i}-\langle \mathbf{u}_{i}\rangle_{q}|}{P}
$$
where the $\mathbf{u_i}$ represents the true cluster label of $i$ and $\langle \mathbf{u}_{i}\rangle_{q}$ is the expected cluster label of brain location $i$ under the expectation $q$. Both are multinomial encoded vectors. We can also replace the expectation with a the hard parcellation, again coded as a one-hot vector. 

Note, this calculation of the mean absolute error is subject to the premutation of the parcellation, so that a loop over all possible permutations and find the minimum error is applied.

#### 2. Normalized Mutual information (NMI) between $\mathbf{U}$ and $\hat{\mathbf{U}}$

the second criteria is the normalized mutual information which examine the actual amount of "mutual information" between two parcellations $\mathbf{U}$ and $\hat{\mathbf{U}}$.  A NMI value closes to 0 indicate two parcellations are largely independent, while values close to 1 indicate significant agreement. It defined as:
$$
NMI(\mathbf{U},\mathbf{\hat{U}})=\frac{2\sum_{i=1}^{k_\mathbf{u}}\sum_{j=1}^{k_\mathbf{\hat{u}}}\frac{|\mathbf{u}=i|\cap|\mathbf{\hat{u}}=j|}{P}\log (P\frac{||\mathbf{u}=i|\cap|\mathbf{\hat{u}}=j||}{|\mathbf{u}=i|\cdot|\mathbf{\hat{u}}=j|})}{\sum_{i=1}^{k_\mathbf{u}}\frac{|\mathbf{u}=i|}{P}\log(\frac{|\mathbf{u}=i|}{P})+\sum_{j=1}^{k_{\mathbf{\hat{u}}}}\frac{|\hat{\mathbf{u}}=j|}{P}\log(\frac{|\hat{\mathbf{u}}=j|}{P})}
$$
where $k_{\mathbf{u}}=\{1,2,3,...,k\}$ and $k_{\mathbf{\hat{u}}}=\{1,2,3,...,k\}$ represents the cluster labels of $\mathbf{U}$ and $\hat{\mathbf{U}}$ respectively. The term$|\mathbf{u}=i|$ and $|\hat{\mathbf{u}}=j|$ are the number of brain locations that belongs to cluster $k_\mathbf{u}=i$ in parcellation $\mathbf{U}$ or to cluster $k_\mathbf{\hat{u}}=j$ in $\mathbf{\hat{U}}$, in other words, the terms $\frac{|\mathbf{u}=i|}{P}$ and $\frac{|\mathbf{\hat{u}}=j|}{P}$ represents the probability that a brain location picked at random from $\mathbf{U}$ falls into class $k_{\mathbf{u}}=i$, or from $\mathbf{\hat{U}}$ falls into class $k_{\mathbf{\hat{u}}}=j$. 

Similarly, the $||\mathbf{u}=i|\cap|\mathbf{\hat{u}}=j||$ means the total number of a brain locations that both falls into classes $k_{\mathbf{u}}=i$ and $k_{\mathbf{\hat{u}}}=j$. Note, the NMI calculation would not suffer from the permutation.

#### 3. Adjusted rand index (ARI) between $\mathbf{U}$ and $\hat{\mathbf{U}}$

the third one is the commonly used adjust rand index to test how similar the two given parcellations are.



### Evaluation on independent test data ($\mathbf{Y}_{test}$)

#### 1a. Cosine error

the first evaluation criterion based on the difference between some observed activity profiles $\mathbf{Y}_{test}$ and the predicted mean directions from the model  $\mathbf{v}_k$ given some expectation of which voxel belongs to what cluster $\langle \mathbf{u}_i \rangle$ . One possibility is to use for each voxel the most likely predicted mean direction. 
$$
\bar{\epsilon}_{cosine} = \frac{1}{P}\sum_i^P (1-{\mathbf{v}_\underset{k}{\operatorname{argmax}}}^{T}\frac{\mathbf{y}_i}{||\mathbf{y}_i||})
$$
where $||\mathbf{y}_i||$ is the length of the data at brain location $i$, $\mathbf{v}_\underset{k}{\operatorname{argmax}}$ represents the $\mathbf{v_k}$ with the maximum expectation for that voxel. We then compute the mean cosine distance across all $P$ brain locations. We can also compute the  *expected* mean cosine distance under the $q(\mathbf{u}_i)$ which defined as below:
$$
\langle\bar{\epsilon}_{cosine}\rangle_q = \frac{1}{P}\sum_i \sum_k \hat{u}_i^{(k)} (1-{\mathbf{v}_k}^{T}\frac{\mathbf{y}_i}{||\mathbf{y}_i||})
$$
where $\hat{u}_i^{(k)}$ is the inferred expectation on the training data using the fitted model.


#### 1b. the adjusted Cosine error

A possible problem with the cosine error is that voxel that have very little signal count as much as voxel with a lot of signal. To address this, we can weight each error by the length of the data vector. This effectively calculates the root mean squared error between $\mathbf{y}_i$ and the prediction scaled to the amplitude of the data ($\||\mathbf{y}_i||\mathbf{v}_k$):
$$
\bar{\epsilon}_{cosine} = \frac{1}{P}\sum_i^P (||\mathbf{y}_i||-{\mathbf{v}_\underset{k}{\operatorname{argmax}}}^{T}\mathbf{y}_i)
$$
where $||\mathbf{y}_i||$ is the length of the data at brain location $i$, $\mathbf{v}_\underset{k}{\operatorname{argmax}}$ represents the $\mathbf{v_k}$ with the maximum expectation. We then compute the mean cosine distance across all $P$ brain locations. Another option is to calculate the *expected* mean cosine distance under the $q(\mathbf{u}_i)$ which defined as below:
$$
\langle\bar{\epsilon}_{cosine}\rangle_q = \frac{1}{P}\sum_i \sum_k  \hat{u}_i^{(k)} (||\mathbf{y}_i||-{\mathbf{v}_k}^{T}\mathbf{y}_i)
$$
where $\hat{u}_i^{(k)}$ is the inferred expectation on the training data using the fitted model.



#### 2. the adjusted RMSE

the second evaluation criterion is to calculate the adjusted square root of mean squared error, which defined as:
$$
\bar{\epsilon}_{RMSE} = \sqrt{\frac{\sum_i (\mathbf{y}_i-||\mathbf{y}_i||\cdot{\mathbf{v}_\underset{k}{\operatorname{argmax}}})^2}{P}}
$$
where $||\mathbf{y}_i||$ is the length of the data at brain location $i$, $\mathbf{v}_\underset{k}{\operatorname{argmax}}$ represents the $\mathbf{v_k}$ with the maximum expectation (argmax). We then compute the root of mean squared error across all $P$ brain locations. Another option is to calculate the *expected* RMSE under the $q(\mathbf{u}_i)$ which defined as below:
$$
\langle\bar{\epsilon}_{RMSE}\rangle_q = \sqrt{\frac{\sum_i \sum_k\hat{\mathbf{u}}_i^{(k)} (\mathbf{y}_i-||\mathbf{y}_i|| \cdot\mathbf{v}_k)^2}{P}}
$$
where $\hat{\mathbf{u}}_i^{(k)}$ is the inferred expectation on the training data using the fitted model.
