Emission model 3a: Mixture of Gaussians with Exponential signal strength
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The emission model should depend on the type of data that is measured. A common application is that the data measured at location :math:`i` are the task activation in :math:`N` tasks, arranged in the :math:`N\times1` data vector :math:`\mathbf{y}_i`. The averaged expected response for each of the parcels is :math:`\mathbf{v}_k`. One issue of the functional activation is that the signal-to-noise ratio (SNR) can be quite different across different participants, and voxels, with many voxels having relatively low SNR. We model this signal to noise for each brain location (and subject) as :math:`s_i \sim exponential(\beta_s)`. Therefore the probability model for gamma is defined as:

.. math::
	p(s_i|\theta) = \beta e^{-\beta s_i}

Overall, the expected signal at each brain location is then 

.. math::
	\rm{E}(\mathbf{y}_i)=\mathbf{u}_i^T \mathbf{V}s_i

Finally, relative to the signal, we assume that the noise is distributed i.i.d Gaussian with: 

.. math::
	\boldsymbol{\epsilon}_i \sim Normal(0,\mathbf{I}_K\theta_{\sigma s})

Here, the proposal distribution :math:`q(u_{i}^{(k)},s_{i}|\mathbf{y}_{i})` is now a multivariate distribution across :math:`u_i` and :math:`s_i`. Thus, the *expected emission log likelihood* :math:`\mathcal{L}_E(q, \theta)` is defined as:

.. math::
	\begin{align*}
	\mathcal{L}_E &= \langle\sum_i\log p(\mathbf{y}_i, s_i|u_i; \theta_E)\rangle_{q}\\
	&=\sum_{i}\sum_{k}\langle u_{i}^{(k)}[-\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}(\mathbf{y}_{i}-\mathbf{v_k}s_i)^T(\mathbf{y}_{i}-\mathbf{v_k}s_i)]\rangle_{q}  \\ &+\sum_{i}\sum_{k}\langle u_{i}^{(k)}[ \log \beta-\beta s_i] \rangle_q\\
	&=-\frac{NP}{2}\log(2\pi)-\frac{NP}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i}\sum_{k}\langle u_{i}^{(k)}(\mathbf{y}_{i}-\mathbf{v_k}s_i)^T(\mathbf{y}_{i}-\mathbf{v_k}s_i)\rangle_{q} \\ &+ P\log\beta-\sum_{i}\sum_k\beta\langle u_{i}^{(k)} s_i\rangle_q\\
	&=-\frac{NP}{2}\log(2\pi)-\frac{NP}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i} \mathbf{y}_i^T\mathbf{y}_i-\frac{1}{2\sigma^{2}}\sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q}) \\ &+\log\beta-\beta \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q
	\end{align*}

Now, we can update the parameters :math:`\theta` of the Gaussians/Exponential mixture in the M-step. The parameters of the gaussian mixture model are :math:`\theta_{E} = \{\mathbf{v}_{1},...,\sigma^{2},\beta\}` . 

1. We start with updating the :math:`\mathbf{v}_k` (Note: the updates only consider a single subject). We take the derivative of *expected emission log likelihood* :math:`\mathcal{L}_E` with respect to :math:`\mathbf{v}_{k}` and make it equals to 0 as following:

.. math::
	\frac{\partial \mathcal{L}_E}{\partial \mathbf{v}_{k}} =-\frac{1}{\sigma^{2}}\sum_{i}-\mathbf{y}_{i}^{T}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\langle u_{i}^{(k)}s_{i}^{2}\rangle_{q} = 0

Thus, we get the updated :math:`\mathbf{v}_{k}` in current M-step as, 

.. math::
	\mathbf{v}_{k}^{(t)} = \frac{\sum_{i}\langle u_{i}^{(k)}s_{i}\rangle_{q}^{(t)}\mathbf{y}_{i}}{\sum_{i}\langle u_{i}^{(k)}s_{i}^{2}\rangle_{q}^{(t)}}

2. Updating :math:`\sigma^{2}` , we take derivative of with respect to :math:`\sigma^{2}` and set it equals to 0 as following:

.. math::
	\frac{\partial \mathcal{L}_E}{\partial \sigma^{2}} =-\frac{NP}{2\sigma^2}+\frac{1}{2\sigma^{4}}\sum_{i}\mathbf{y}_{i}^T\mathbf{y}_{i}+\frac{1}{2\sigma^{4}}\sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q}) = 0

Thus, we get the updated :math:`\sigma^{2}` for parcel :math:`k` in the current M-step as,

.. math::
	{\sigma^2}^{(t)} = \frac{1}{NP}(\sum_{i}\mathbf{y}_i^T\mathbf{y}_i+
	\sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q})

3. Updating :math:`\beta`, we take derivative of  :math:`\mathcal{L}_E(q, \theta)` with respect to :math:`\beta` and set it equal to 0 as following:

.. math::
	\begin{align*}
	\frac{\partial \mathcal{L}_E}{\partial \beta} &=\frac{\partial [P\log\beta-\beta \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q]}{\partial \beta} \\
	&= \frac{P\alpha}{\beta}-\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q = 0
	\end{align*}

Thus, we get the updated :math:`\beta_{k}` in current M-step as, 

.. math::
	\beta_{k}^{(t)} =  	\frac{P}{\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q}


Emission model 3b: Mixture of Gaussians with Gamma signal strength
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The emission model should depend on the type of data that is measured. A common application is that the data measured at location :math:`i` are the task activation in :math:`N` tasks, arranged in the :math:`N\times1` data vector :math:`\mathbf{y}_i`. The averaged expected response for each of the parcels is :math:`\mathbf{v}_k`. One issue of the functional activation is that the signal-to-noise ratio (SNR) can be quite different across different participants, and voxels, with many voxels having relatively low SNR. We model this signal to noise for each brain location (and subject) as :math:`s_i \sim Gamma(\theta_\alpha,\theta_{\beta s})`. Therefore the probability model for gamma is defined as:

.. math::
	p(s_i|\theta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)}s_i^{\alpha-1}e^{-\beta s_i}

Overall, the expected signal at each brain location is then 

.. math::
	\rm{E}(\mathbf{y}_i)=\mathbf{u}_i^T \mathbf{V}s_i


Finally, relative to the signal, we assume that the noise is distributed i.i.d Gaussian with: 

.. math::
	\boldsymbol{\epsilon}_i \sim Normal(0,\mathbf{I}_K\theta_{\sigma s})

Here, the proposal distribution :math:`q(u_{i}^{(k)},s_{i}|\mathbf{y}_{i})` is now a multivariate distribution across :math:`u_i` and :math:`s_i`. Thus, the *expected emission log likelihood* :math:`\mathcal{L}_E(q, \theta)` is defined as:

.. math::
	\begin{align*}
	\mathcal{L}_E &= \langle\sum_i\log p(\mathbf{y}_i, s_i|u_i; \theta_E)\rangle_{q}\\
	&=\sum_{i}\sum_{k}\langle u_{i}^{(k)}[-\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}(\mathbf{y}_{i}-\mathbf{v_k}s_i)^T(\mathbf{y}_{i}-\mathbf{v_k}s_i)]\rangle_{q}  \\ &+\sum_{i}\sum_{k}\langle u_{i}^{(k)}[\alpha \log \beta-\log\Gamma(\alpha)+(\alpha-1)\log s_i-\beta s_i] \rangle_q\\
	&=-\frac{NP}{2}\log(2\pi)-\frac{NP}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i}\sum_{k}\langle u_{i}^{(k)}(\mathbf{y}_{i}-\mathbf{v_k}s_i)^T(\mathbf{y}_{i}-\mathbf{v_k}s_i)\rangle_{q} \\ &+ P\alpha\log\beta-P\log\Gamma(\alpha)+\sum_{i}\sum_k\langle u_{i}^{(k)}(\alpha-1)\log s_i-u_{i}^{(k)}\beta s_i\rangle_q\\
	&=-\frac{NP}{2}\log(2\pi)-\frac{NP}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i} \mathbf{y}_i^T\mathbf{y}_i-\frac{1}{2\sigma^{2}}\sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q}) \\ &+P\alpha\log\beta-P\log\Gamma(\alpha)+(\alpha-1)\sum_{i}\sum_k \langle u_{i}^{(k)}\log s_i\rangle_q-\beta \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q
	\end{align*}

Now, we can update the parameters :math:`\theta` of the Gaussians/Gamma mixture in the M-step. The parameters of the gaussian mixture model are :math:`\theta_{E} = \{\mathbf{v}_{1},...,\sigma^{2},\alpha,\beta\}` . 

1. We start with updating the :math:`\mathbf{v}_k` (Note: the updates only consider a single subject). We take the derivative of *expected emission log likelihood* :math:`\mathcal{L}_E` with respect to :math:`\mathbf{v}_{k}` and make it equals to 0 as following:

.. math::
	\frac{\partial \mathcal{L}_E}{\partial \mathbf{v}_{k}} =-\frac{1}{\sigma^{2}}\sum_{i}-\mathbf{y}_{i}^{T}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\langle u_{i}^{(k)}s_{i}^{2}\rangle_{q} = 0

Thus, we get the updated :math:`\mathbf{v}_{k}` in current M-step as, 

.. math::
	\mathbf{v}_{k}^{(t)} = \frac{\sum_{i}\langle u_{i}^{(k)}s_{i}\rangle_{q}^{(t)}\mathbf{y}_{i}}{\sum_{i}\langle u_{i}^{(k)}s_{i}^{2}\rangle_{q}^{(t)}}

2. Updating :math:`\sigma^{2}` , we take derivative of with respect to :math:`\sigma^{2}` and set it equals to 0 as following:

.. math::
	\frac{\partial \mathcal{L}_E}{\partial \sigma^{2}} =-\frac{NP}{2\sigma^2}+\frac{1}{2\sigma^{4}}\sum_{i}\mathbf{y}_{i}^T\mathbf{y}_{i}+\frac{1}{2\sigma^{4}}\sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q}) = 0

Thus, we get the updated :math:`\sigma^{2}` for parcel :math:`k` in the current M-step as,

.. math::
	{\sigma^2}^{(t)} = \frac{1}{NP}(\sum_{i}\mathbf{y}_i^T\mathbf{y}_i+
	\sum_{i}\sum_{k}(-2\mathbf{y}_{i}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}\rangle_{q}+\mathbf{v}_{k}^T\mathbf{v}_{k}\langle u_{i}^{(k)}s_{i}^2\rangle_{q})


3. Updating :math:`\beta`, we take derivative of  :math:`\mathcal{L}_E(q, \theta)` with respect to :math:`\beta` and set it equal to 0 as following:

.. math::
	\begin{align*}
	\frac{\partial \mathcal{L}_E}{\partial \beta} &=\frac{\partial [P\alpha\log\beta-\beta \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q]}{\partial \beta} \\
	&= \frac{P\alpha}{\beta}-\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q = 0
	\end{align*}

Thus, we get the updated :math:`\beta_{k}` in current M-step as, 

.. math::
	\beta_{k}^{(t)} =  	\frac{P\alpha_k^{(t)}}{\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q}


4. Updating :math:`\alpha_{k}` is comparatively hard since we cannot derive closed-form, we take derivative of  :math:`\mathcal{L}_E(q, \theta)` with respect to :math:`\alpha` and make it equals to 0 as following:


.. math::
	\begin{align*}
	\frac{\partial \mathcal{L}_E}{\partial \alpha} &=\frac{\partial [P\alpha\log\beta-P\log\Gamma(\alpha)+(\alpha-1)\sum_{i}\sum_k \langle u_{i}^{(k)}\log{s_i}\rangle_q]}{\partial \alpha}\\
	&=P\log\beta-P\frac{\Gamma'(\alpha)}{\Gamma(\alpha)}+\sum_{i}\sum_k \langle u_{i}^{(k)} \log {s_i}\rangle_q = 0
	\end{align*}

The term :math:`\frac{\Gamma'(\alpha)}{\Gamma(\alpha)}` in above equation is exactly the *digamma function* and we use :math:`\digamma(\alpha)` to represent. Also from (4), we know :math:`\beta=\frac{P\alpha_k^{(t)}}{\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q}` Thus, we get the updated :math:`\alpha` in current M-step as, 

.. math::
	\begin{align*}
	\digamma(\alpha)^{(t)} &= \log \frac{P\alpha_k^{(t)}}{\sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q}+\frac{1}{P}\sum_{i}\sum_k \log \langle u_{i}^{(k)}s_i\rangle_q\\
	&=\log P\alpha^{(t)} - \log \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q + \frac{1}{P}\sum_{i}\sum_k\log\langle u_{i}^{(k)}s_i\rangle_q
	\end{align*}

By applying "generalized Newton" approximation form, the updated :math:`\alpha` is as follows: 

.. math::
	\alpha^{(t)} \approx \frac{0.5}{\log \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q - \frac{1}{P}\sum_{i}\sum_k\langle u_{i}^{(k)} \log{s_i}\rangle_q}

Note that :math:`\log \sum_{i}\sum_k\langle u_{i}^{(k)}s_i\rangle_q \geqslant \frac{1}{P}\sum_{i}\sum_k\log\langle u_{i}^{(k)}s_i\rangle_q` is given by Jensen's inequality.
