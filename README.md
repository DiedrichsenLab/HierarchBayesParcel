# generativeMRF
Collection of functions to play around with different Markov random field models as generative processes for brain activity data.

## Generative Potts Model
This is a generative Potts model of brain activity data. The main idea is that the brain consists of $K$ regions, each with a specific activity profile $\mathbf{v}_k$ for a specific task set. The model consists of a arrangement model that tells us how the $K$ regions are arranged in a specific subject $s$, and an emission model that provides a probability of the measured data, given the individual arrangement of regions.

### Arrangement model
The brain is sampled in $P$ vertices (or voxels). Individual maps are aligned using anatomical normalization, such that each vertex refers to a (roughly) corresponding region in each individual brain. The assignment of each brain location to a specific parcel in subject $s$ is expressed as the random variable $u_i^{(s)}$.

Across individual brains, we have the overall probability of a specific brain location being part of parcel $k$.
$$
p(u_i = k) = \mu_{ki}
$$

The spatial interdependence of brain locations is expressed as a Potts model. In this model, the overall probability of a specific assignment of brain locations to parcels (the vector $\mathbf{u}$) is expressed as the product of the overall prior and the product of all possible pairwise potentenials ($\psi_{ij}$). 

$$
p(\mathbf{u}) \propto \prod_{i}\mu_{u_i,i}\prod_{i\neq j}{\psi_{ij}(u_i,u_j) }
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

In summary, we can express the prior probability of a specific arrangement in terms of a set of conditional probabilities 
$$
p(u_i|u_{j\neq i}) \propto \prod_{i}\mu_{u_i,i}\prod_{i\neq j}{\psi_{ij}(u_i,u_j) }
$$
  and we the corresponding conditional log-probability
$$
l(u_i|u_{j\neq i}) \propto \rm{log}\mu_{u_i,i}+ \theta_{w}\sum_{i\neq j}{\mathbf{u}_i^T\mathbf{u}_j w_{ij}}
$$

### Evidence lower bound 

By using the Jensen's inequality, we define a lower bound to the complete log likelihood:
$$
\begin{align*}
\log p(\mathbf{y},u | \theta)
&=\sum_{i}\log \sum_{k}p(\mathbf{y}_{i},u_{i}^{(k)}|\theta) \\ 
&=\sum_{i}\log\sum_{k}q(u_{i}^{(k)}|\mathbf{y}_{i})\frac{p(\mathbf{y}_{i},u_{i}^{(k)}|\theta)}{q(u_{i}^{(k)}|\mathbf{y}_{i})}\\
&\geqslant\sum_{i}\sum_{k}q(u_{i}^{(k)}|\mathbf{y}_{i})\log\frac{p(\mathbf{y}_{i},u_{i}^{(k)}|\theta)}{q(u_{i}^{(k)}|\mathbf{y}_{i})} \triangleq \mathcal{L}(q, \theta)
\end{align*}
$$


### Emission model 1: Mixture of Gaussians

We first try the Gaussian Mixtures as our emission probability model. The original probability model for a $N$-dimensional multivariate Gaussian mixture is as follow:
$$
p(x|\theta) = \sum_{i}{\pi_{i}}\frac{1}{(2\pi)^{N/2}|\sum_{i}|^{1/2}}\rm{exp}\{-\frac{1}{2}(x-\mu_{i})^T\Sigma_{i}^{-1}(x-\mu_{i})\}
$$
In our emission model, for all $i$ voxel and the associated $N$-dimensional data vector $y_{i}$, the above equation can be re-written as:
$$
p(\mathbf{y}, u^{(k)}|\theta) = \sum_{i}\sum_{k}u_{i}^{(k)}\pi_{i}^{(k)}\frac{1}{(2\pi)^{N/2}|\sum_{i}|^{1/2}}\rm{exp}\{-\frac{1}{2}(y_{i}-\mu_{i})^T\Sigma_{i}^{-1}(y_{i}-\mu_{i})\}
$$
Since we define $p(y_{i}|u_{i}^{(k)}) \triangleq \mathcal{N}(\mathbf{v}_k,\,I\sigma^{2})$, where $\mathbf{v}_k$ is the averaged expected response for each of the parcels. Note that $|\mathbf{I}_N \sigma^2|=\sigma^{2N}$. Then we plug back these parameters back to the model and get:
$$
p(\mathbf{y}, u^{(k)}|\theta) = \sum_{i}\sum_{k}u_{i}^{(k)}\pi_{i}^{(k)}\frac{1}{(2\pi)^{N/2}(\sigma^{2})^{N/2}}\rm{exp}\{-\frac{1}{2\sigma^{2}}(y_{i}-\mathbf{v}_k)^T(y_{i}-\mathbf{v}_k)\}
$$
Thus, the maximization of the complete log likelihood is equivalent to maximize its lower bound: the *expected log likelihood* with respect to the expectation $q$. Here, we use $\langle u_{i}^{(k)}\rangle_{q}$ to represent the expected distribution $q(u_{i}^{(k)}|\mathbf{y}_{i})$. Thus, the *expected log likelihood* $\mathcal{L}(q, \theta)$ is defined as following:
$$
\begin{align*}
\mathcal{L}(q, \theta) &= \langle\log p(\mathbf{y}, u|\theta)\rangle_{q}\\
&=\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}[-\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}(\mathbf{y}_{i}-\mathbf{v}_{k})^T(\mathbf{y}_{i}-\mathbf{v}_{k})]+\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}\pi_{i}^{(k)}
\end{align*}
$$
Now, with the above expected log likelihood by hand, we can update the parameters $\theta$ of the Gaussians mixture in the $\Mu$ step. There are three parameters in Gaussians mixture $\theta_{k}\sim(\mu_{k},\sigma^{2}_{k},\pi_{k})$. Now, we start with updating the $\mu$. (Note: the updates only consider a single subject)

1. Updating $\mu$, we take derivative of *expected log likelihood* $\mathcal{L}(q, \theta)$ with respect to $\mathbf{v}_{k}$ and set it to 0:
   $$
   \frac{\partial \mathcal{L}}{\partial \mathbf{v}_{k}} =\frac{1}{\sigma^{2}}\sum_{i}\langle u_{i}^{(k)}\rangle_{q}(\mathbf{y}_{i}-\mathbf{v}_{k}) = 0
   $$
   Thus, we get the updated $\mathbf{v}_{k}$ in current $\Mu$ step as, 
   $$
   \mathbf{v}_{k}^{(t)} = \frac{\sum_{i}\langle u_{i}^{(k)}\rangle_{q}^{(t)}\mathbf{y}_{i}}{\sum_{i}\langle u_{i}^{(k)}\rangle_{q}^{(t)}}
   $$

2. Updating $\sigma^{2}$, we take derivative of *expected log likelihood* $\mathcal{L}(q, \theta)$ with respect to $\sigma^{2}$  and set it to  0: 
   $$
   \frac{\partial \mathcal{L}}{\partial \sigma^{2}} =\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}[-\frac{N}{2\sigma^{2}}+\frac{1}{2\sigma^{4}}(\mathbf{y}_{i}-\mathbf{v}_{k})^T(\mathbf{y}_{i}-\mathbf{v}_{k})] = 0
   $$
   Thus, we get the updated $\sigma^{2}$ for parcel $k$ in the current $\Mu$ step as,
   $$
   {\sigma^{2}}^{(t)} = \frac{1}{NP}\sum_{i}\sum_{k}\langle u_{i}^{(k)}\rangle_{q}^{(t)}(\mathbf{y}_{i}-\mathbf{v}_{k})^T(\mathbf{y}_{i}-\mathbf{v}_{k})
   $$
   where $P$ is the total number of voxels $i$.

3. Updating $\pi_{k}$ for parcel $k$, we take derivative of *expected log likelihood* $\mathcal{L}(q, \theta)$ with respect to $\pi$ and make it equals to 0 as following:
   $$
   \frac{\partial \mathcal{L}}{\partial \pi_{k}} =\sum_{i}\langle u_{i}^{(k)}\rangle_{q}= 0
   $$
   Thus, we get the updated $\pi_{k}$ in current $\Mu$ step as, 
   $$
   \pi_{k}^{(t)} = \frac{1}{P}\sum_{i}\langle u_{i}^{(k)}\rangle_{q}^{(t)}
   $$

The updated parameters $\theta_{k}^{(t)}\sim(\mu_{k}^{(t)},{\sigma^{2}_{k}}^{(t)},\pi_{k}^{(t)})$ from current $\mathbf{M}$-step will be passed to the $\mathbf{E}$-step of $(t+1)$ times for calculating the expectation.

### Emission model 2: Van Mises Distribution

### Emission model 3: Mixture of Gaussians with signal strength

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
 Here, we use $\langle u_{i}^{(k)},s_{i}\rangle_{q}$ to represent the expected distribution $q(u_{i}^{(k)},s_{i}|\mathbf{y}_{i})$. Thus, the *expected log likelihood* $\mathcal{L}(q, \theta)$ is defined as following:
$$
\begin{align*}
\mathcal{L}(q, \theta) &= \langle\log p(\mathbf{y}, s|u, \theta)\rangle_{q}\\
&=\sum_{i}\sum_{k}\langle u_{i}^{(k)}[-\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}(\mathbf{y}_{i}-\mathbf{v_k}s_i)^T(\mathbf{y}_{i}-\mathbf{v_k}s_i)\rangle_{q} \\
&=-\frac{NP}{2}\log(2\pi)-\frac{NP}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}(\sum_{i}\sum_{k}\mathbf{y}_{i}^T\mathbf{y}_{i}\langle u_i^{(k)\rangle-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q}))]+\sum_{i}\sum_{k}\langle u_{i}^{(k)},s_{i}\rangle_{q}\pi_{i}^{(k)}
\end{align*}
$$
Now, we can update the parameters $\theta$ of the Gaussians/Gamma mixture in the $\Mu$ step. There are total five parameters in this Gaussians/Gamma mixture $\theta_{k}\sim(\mu_{k},\sigma^{2}_{k},\pi_{k},\alpha_{k},\beta_{k})$ for each parcel $k$. Now, we start with updating the $\mu$ ($\mathbf{v}_k$ in our case). (Note: the updates only consider a single subject)

1. Updating $\mathbf{v}_k$, we take derivative of *expected log likelihood* $\mathcal{L}(q, \theta)$ with respect to $\mathbf{v}_{k}$ and make it equals to 0 as following:
   $$
   \frac{\partial \mathcal{L}}{\partial \mathbf{v}_{k}} =-\frac{1}{\sigma^{2}}\sum_{i}-\mathbf{y}_{i}^{T}\langle u_{i}^{(k)},s_{i}\rangle_{q}+\mathbf{v}_{k}^T\langle u_{i}^{(k)},s_{i}^{2}\rangle_{q} = 0
   $$
   Thus, we get the updated $\mathbf{v}_{k}$ in current $\Mu$ step as, 
   $$
   \mathbf{v}_{k}^{(t)} = \frac{\sum_{i}\langle u_{i}^{(k)},s_{i}\rangle_{q}^{(t)}\mathbf{y}_{i}}{\sum_{i}\langle u_{i}^{(k)},s_{i}^{2}\rangle_{q}^{(t)}}
   $$

2. Updating $\sigma_{k}^{2}$ for parcel $k$, we take derivative of *expected log likelihood* $\mathcal{L}(q, \theta)$ with respect to $\sigma^{2}$ ( $I\sigma^{2}$ in this case) and make it equals to 0 as following:
   $$
   \frac{\partial \mathcal{L}}{\partial \sigma_{k}^{2}} =\sum_{i}\langle u_{i}^{(k)},s_{i}\rangle_{q}[-\frac{N}{2\sigma^{2}}+\frac{1}{2\sigma^{4}}(\mathbf{y}_{i}-\mathbf{v}_{k}s_{i})^T(\mathbf{y}_{i}-\mathbf{v}_{k}s_{i})] = 0
   $$
   Thus, we get the updated $\sigma^{2}$ for parcel $k$ in the current $\Mu$ step as,
   $$
   {\sigma_{k}^{2}}^{(t)} = \frac{1}{\mathbf{U}}\sum_{i}\langle u_{i}^{(k)},s_{i}\rangle_{q}^{(t)}(\mathbf{y}_{i}-\mathbf{v}_{k}s_{i})^T(\mathbf{y}_{i}-\mathbf{v}_{k}s_{i})
   $$

3. Updating $\pi_{k}$ for parcel $k$, we take derivative of *expected log likelihood* $\mathcal{L}(q, \theta)$ with respect to $\pi$ and make it equals to 0 as following:
   $$
   \frac{\partial \mathcal{L}}{\partial \pi_{k}} =\sum_{i}\langle u_{i}^{(k)},s_{i}\rangle_{q}= 0
   $$
   Thus, we get the updated $\pi_{k}$ in current $\Mu$ step as, 
   $$
   \pi_{k}^{(t)} = \frac{1}{\mathbf{U}}\sum_{i}\langle u_{i}^{(k)},s_{i}\rangle_{q}^{(t)}
   $$

4. Updating $\alpha_{i}$, we take derivative of *expected log likelihood* $\mathcal{L}(q, \theta)$ with respect to $\alpha$​ and make it equals to 0 as following:
   $$
   \begin{align*}
   \frac{\partial \mathcal{L}}{\partial \alpha_{i}} &=\sum_{k}\langle u_{i}^{(k)},s_{i}\rangle_{q}\frac{\mathbf{v}_{k}}{\sigma_{k}^{2}}(\mathbf{y}_{i}-\mathbf{v}_{k}s_{i})^T\frac{\partial \log s_{i}}{\partial \alpha_{i}} \tag{Chain rule}\\
   &= \sum_{k}\langle u_{i}^{(k)},s_{i}\rangle_{q}\frac{\mathbf{v}_{k}}{\sigma_{k}^{2}}(\mathbf{y}_{i}-\mathbf{v}_{k}s_{i})^T\frac{\partial[(\alpha-1)\log s_{i}-\beta s_{i}]}{\partial \alpha_{i}} \\
   &= \sum_{k}\langle u_{i}^{(k)},s_{i}\rangle_{q}\frac{\mathbf{v}_{k}}{\sigma_{k}^{2}}(\mathbf{y}_{i}-\mathbf{v}_{k}s_{i})^T\log s_{i} =0\\
   \end{align*}
   $$
   Thus, we get the updated $\alpha_{i}$ in current $\Mu$ step as, 
   $$
   \alpha_{i}^{(t)} =
   $$

5. Updating $\beta_{i}$, we take derivative of *expected log likelihood* $\mathcal{L}(q, \theta)$ with respect to $\beta$​ and make it equals to 0 as following:

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \beta_{i}} &=\sum_{k}\langle u_{i}^{(k)},s_{i}\rangle_{q}\frac{\mathbf{v}_{k}}{\sigma_{k}^{2}}(\mathbf{y}_{i}-\mathbf{v}_{k}s_{i})^T\frac{\partial \log s_{i}}{\partial \beta_{i}} \tag{Chain rule}\\
&= \sum_{k}\langle u_{i}^{(k)},s_{i}\rangle_{q}\frac{\mathbf{v}_{k}}{\sigma_{k}^{2}}(\mathbf{y}_{i}-\mathbf{v}_{k}s_{i})^T\frac{\partial[(\alpha-1)\log s_{i}-\beta s_{i}]}{\partial \beta_{i}} \\
&= \sum_{k}\langle u_{i}^{(k)},s_{i}\rangle_{q}\frac{\mathbf{v}_{k}}{\sigma_{k}^{2}}(\mathbf{y}_{i}-\mathbf{v}_{k}s_{i})^T s_{i} =0\\
\end{align*}
$$

​		Thus, we get the updated $\beta_{i}$ in current $\Mu$ step as, 
$$
\beta_{i}^{(t)} =
$$


### Sampling from the prior or posterior distribution

The problem with determine the overall prior or posterior distribution of the model (for purposes of data generation or inference) cannot be easily be computed. We can evaluate the prior probability of a parcellation $p(\mathbf{U})$ or the posterior distribution $p(\mathbf{U}|\mathbf{Y})$ up to a constant of proportionality, with for example 
$$
p(\mathbf{U}|\mathbf{Y};\theta) = \frac{1}{Z(\theta)}\prod_{i}\mu_{u_i,i}\prod_{i\neq j}{\psi_{ij}(u_i,u_j) }\prod_{i}p(\mathbf{y}_i|u_i)
$$
Calculating the normalization constant $Z(\theta)$ (partition function, Zustandssumme, or sum over states) would involve summing this probability over all possible states, which for $P$ brain locations and $K$ parcels is $K^P$, which is intractable. 

However, the conditional probability for each node, given all the other nodes, can be easily computed. Here the normalizaton constant is just the sum of the potential functions over the $K$ possible states for this node

$$
p(u_i|u_{j \neq i},\mathbf{y}_i;\theta) = \frac{1}{Z(\theta)}\mu_{u_i,i} \; p(\mathbf{y}_i|u_i) \prod_{i\neq j}{\psi_{ij}(u_i,u_j) }
$$
With Gibbs sampling, we start with a pattern $\mathbf{u}^{(0)}$ and then update $u_1^{(1)}$ by sampling from $p(u_1|u_2^{(0)}...u_P^{(0)})$. We then sample $u_2^{(1)}$ by sampling from $p(u_2|u_1^{(1)}, u_3^{(0)}...u_P^{(0)})$ and so on, until we have sampled each node once. Then we return to the beginning and restart the process. After some burn-in period, the samples will come from desired overall distribution. If we want to sample from the prior, rather than from the posterior, we simply drop the $p(\mathbf{y}_i|u_i)$ term from the conditional probability above. 



