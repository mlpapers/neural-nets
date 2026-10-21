# Neural networks, deep learning papers

## Overviews and approximation

- **[Deep Learning](https://www.deeplearningbook.org/)** — 2016. Textbook covering feedforward, convolutional, recurrent, and generative networks; optimization, regularization, representation learning, and approximate inference.
- **[Historical overview](https://arxiv.org/abs/1404.7828)** — 2014. Surveys deep supervised, unsupervised, reinforcement, and evolutionary learning, organized around credit-assignment paths and their history.
- **[Perceptron](https://doi.org/10.1037/h0042519)** — 1958. Rosenblatt's probabilistic model of recognition, learning, and memory. Studies statistical separability and the capabilities and limitations of early neural systems.
- **[Universal approximation](https://cognitivemedium.com/magic_paper/assets/Hornik.pdf)** — 1989. Establishes approximation capacity for sufficiently wide feedforward networks with squashing activations. An existence result, not a guarantee of efficient training or generalization.
- **[Large multilayer perceptrons](https://arxiv.org/abs/1003.0358)** — 2010. Studies deep, wide MLPs trained by online backpropagation on augmented handwritten digits using GPUs. Results concern MNIST and the specified training procedure.

## Feedforward and convolutional architectures

- **[GMDH polynomial networks](https://www.gmdh.net/articles/history/polynomial.pdf)** — 1971. Builds multilayer polynomial models, selecting useful combinations before constructing more complex ones. Uses an external criterion to control model complexity.
- **[GMDH algorithm families](https://gmdh.net/articles/review/algorith.pdf)** — 1995. Reviews polynomial and nonparametric GMDH algorithms for forecasting, clustering, and system identification. Choice depends on noise, sample size, and variable types.
- **[Neocognitron](https://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf)** — 1980. A self-organizing visual hierarchy of simple- and complex-cell-like layers, designed for pattern recognition tolerant of position changes.
- **[Backpropagation for digit recognition](https://doi.org/10.1162/neco.1989.1.4.541)** — 1989. Encodes task constraints in network architecture and learns the mapping from normalized handwritten digit images to classifications. An early end-to-end recognition system.
- **[GPU-trained CNNs](https://people.idsia.ch/~juergen/ijcai2011.pdf)** — 2011. Presents a configurable GPU implementation of CNNs with supervised feature learning. Evaluates architecture variants on digit and object classification.
- **[CNN survey](https://arxiv.org/abs/2004.02806)** — 2020. Reviews convolutional architectures, component choices, and applications across one-, two-, and higher-dimensional inputs.
- **[ResNet](https://arxiv.org/abs/1512.03385)** — 2015. Learns residual functions relative to layer inputs. Studies how this formulation enables optimization of substantially deeper image-recognition networks.
- **[MLP-Mixer](https://arxiv.org/abs/2105.01601)** — 2021. Alternates MLP operations across patches and across channels. Demonstrates an image-classification architecture built without convolution or attention.
- **[Kolmogorov–Arnold networks](https://arxiv.org/abs/2404.19756)** — 2024. Replaces scalar edge weights with learnable univariate spline functions. Evaluates function fitting and scientific problems; broader superiority over MLPs is not established.

## Sequence models, attention, and implicit depth

- **[LSTM](https://doi.org/10.1162/neco.1997.9.8.1735)** — 1997. Uses memory cells and multiplicative gates to preserve error flow over long delays. The original design targets vanishing gradients in recurrent sequence learning.
- **[Bidirectional LSTM](https://www.cs.toronto.edu/~graves/nn_2005.pdf)** — 2005. Combines forward and backward sequence context for framewise phoneme classification. Access to later inputs must match the deployment setting.
- **[Transformer](https://arxiv.org/abs/1706.03762)** — 2017. Builds sequence transduction around attention, removing recurrence and convolution from the architecture. The original encoder–decoder is evaluated on translation and parsing.
- **[S4](https://arxiv.org/abs/2111.00396)** — 2021. Uses a structured state-space parameterization to model long sequences efficiently, reducing its core computation to a Cauchy kernel.
- **[Mamba](https://arxiv.org/abs/2312.00752)** — 2023. Makes state-space parameters depend on the input, allowing selective retention and forgetting. Uses a hardware-aware recurrent computation with linear scaling in sequence length.
- **[Mamba-2](https://arxiv.org/abs/2405.21060)** — 2024. Connects structured state-space models and attention variants through semiseparable matrices. Uses this duality to derive a revised selective state-space layer.
- **[Neural ODEs](https://arxiv.org/abs/1806.07366)** — 2018. Parameterizes hidden-state derivatives with a neural network and computes outputs with an ODE solver. Numerical tolerances and solver behavior become part of model execution.
- **[Deep equilibrium models](https://arxiv.org/abs/1909.01377)** — 2019. Finds a weight-tied network's equilibrium by root finding and differentiates it implicitly. Replaces an explicit stack of layers with a fixed-point computation.
- **[Rotary position embeddings](https://arxiv.org/abs/2104.09864)** — 2021. Encodes position through rotations of query and key vectors, making their dot products depend on relative position offsets; used inside attention rather than as added position vectors.

## Learning operators between functions

- **[DeepONet](https://arxiv.org/abs/1910.03193)** — 2019/2021. A branch network encodes an input function at sensors; a trunk network encodes output locations. Their combination learns a nonlinear operator. Universal approximation concerns representability, not successful fitting or generalization on new functions.
- **[Fourier Neural Operator](https://arxiv.org/abs/2010.08895)** — 2020/2021. Parameterizes integral kernels in Fourier space to learn maps between functions, such as PDE coefficients and solutions. Demonstrates transfer across resolutions on studied equations; a learned operator still needs validation on new regimes.

## Conditional computation

- **[Sparsely gated mixture of experts](https://arxiv.org/abs/1701.06538)** — 2017. A learned gate selects a sparse combination of feedforward experts per example. Separates total parameter capacity from the subset activated for each input.
- **[Switch Transformer](https://arxiv.org/abs/2101.03961)** — 2021. Simplifies mixture-of-experts routing and addresses communication costs and training instability. Studies sparse conditional computation in language models.

## Unsupervised and generative networks

- **[Competitive learning](https://doi.org/10.1207/s15516709cog0901_5)** — 1985. Rumelhart and Zipser study how competition among units discovers feature detectors without target labels, including multilayer representations of nonlinearly separable patterns.
- **[Self-organizing maps](https://link.springer.com/article/10.1007/BF00337288)** — 1982. Organizes adaptive units into a low-dimensional array whose responses reflect input topology. The original study also examines conditions where ordering fails.
- **[Cresceptron](https://experts.illinois.edu/en/publications/cresceptron-a-self-organizing-neural-network-which-grows-adaptive/)** — 1992. Grows a hierarchy of neurons and connections as new concepts appear. Uses hierarchical tolerance for recognition and traces response paths for segmentation.
- **[Modular autoassociative learning](https://www.aaai.org/Papers/AAAI/1987/AAAI87-050.pdf)** — 1987. Couples autoassociative modules into hierarchies to learn internal representations. An early approach to organizing learning in networks with multiple hidden levels.
- **[Autoencoder objectives](https://www.deeplearningbook.org/contents/autoencoders.html)** — 2016. Learns an encoder and reconstruction decoder under restrictions that prevent unrestricted copying. Covers bottlenecks, regularization, and links to latent-variable models.
- **[Denoising autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf)** — 2008. Learns representations by reconstructing inputs from corrupted versions. Stacked denoising modules can initialize deeper networks.
- **[Contractive autoencoders](https://www.iro.umontreal.ca/~lisa/pointeurs/ICML2011_explicit_invariance.pdf)** — 2011. Adds a penalty on the encoder's input Jacobian to reconstruction loss, encouraging local insensitivity of the representation to input changes.
- **[Variational autoencoders](https://arxiv.org/abs/1312.6114)** — 2013. Learns a latent-variable generative model and an approximate posterior encoder by optimizing a variational lower bound with reparameterized gradients.
- **[VAE tutorial](https://arxiv.org/abs/1606.05908)** — 2016. Develops the intuition, mathematics, and empirical behavior of variational autoencoders, including their use for modeling complex data distributions.
- **[VAE foundations and extensions](https://arxiv.org/abs/1906.02691)** — 2019. Reviews deep latent-variable models, learned inference models, and extensions of the variational-autoencoder framework.
- **[VAEs for images, labels, and captions](https://papers.nips.cc/paper/6528-variational-autoencoder-for-deep-learning-of-images-labels-and-captions.pdf)** — 2016. Combines an image encoder and generative decoder with label or caption models. Supports learning from images with or without associated annotations.
- **[Deep autoregressive networks](https://arxiv.org/abs/1310.8499)** — 2013. Adds autoregressive connections to stochastic hidden layers, enabling exact ancestral sampling. Learns through an approximate variational/description-length objective.
- **[Boltzmann machines](https://papers.cnl.salk.edu/PDFs/Learning%20and%20Relearning%20in%20Boltzmann%20Machines%201986-3239.pdf)** — 1986. Studies stochastic networks with symmetric connections, energy-based relaxation, and learning of hidden representations. Distinct from sequence-processing RNNs such as LSTM.
- **[GANs](https://arxiv.org/abs/1406.2661)** — 2014. Trains a generator against a discriminator in an adversarial game. The idealized distribution-matching equilibrium does not guarantee convergence of finite-network training.
- **[TimeGAN](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks.pdf)** — 2019. Jointly learns an embedding space with supervised temporal and adversarial objectives to generate time series that preserve temporal relationships.
- **[ForGAN](https://arxiv.org/abs/1903.12549)** — 2019. Uses a conditional GAN for one-step probabilistic forecasting. Learns a distribution over future values rather than only a point prediction.

## Bayesian neural networks and uncertainty

- **[Bayesian backpropagation framework](https://authors.library.caltech.edu/13793/1/MACnc92b.pdf)** — 1992. Uses Bayesian evidence for regularization and model comparison, with uncertainty estimates for weights and outputs. Conclusions depend on the likelihood and prior assumptions.
- **[Bayesian learning for neural networks](https://www.cs.toronto.edu/~radford/ftp/thesis.pdf)** — 1995. Studies network priors, infinite-width limits, and posterior prediction using hybrid Monte Carlo. Includes automatic relevance determination for inputs.
- **[Practical Bayesian approximations](https://doi.org/10.1088/0954-898X_6_3_011)** — 1995. Reviews Gaussian approximations for parameter inference, model comparison, regularization, and prediction in supervised neural networks.
- **[Stochastic variational inference for networks](https://www.cs.toronto.edu/~graves/nips_2011.pdf)** — 2011. Approximates variational expectations numerically to train a broad range of networks. Connects the objective to description length, regularization, and pruning.
- **[Bayes by Backprop](https://arxiv.org/abs/1505.05424)** — 2015. Learns a distribution over network weights by optimizing variational free energy with backpropagation-compatible gradients.
- **[Monte Carlo dropout](https://arxiv.org/abs/1506.02142)** — 2015. Interprets dropout as approximate Bayesian inference. Stochastic predictions can represent model uncertainty; collecting multiple passes adds inference cost.
- **[SGD as approximate inference](https://www.cs.columbia.edu/~blei/papers/MandtHoffmanBlei2017.pdf)** — 2017. Under a stochastic-process approximation, tunes constant-step SGD so its stationary distribution approximates a posterior. Ordinary SGD is not automatically exact Bayesian sampling.
- **[Deep networks as Gaussian processes](https://arxiv.org/abs/1711.00165)** — 2017. Derives a Gaussian-process correspondence for infinitely wide deep networks under specified parameter priors. The limit supplies a kernel for Bayesian prediction.
- **[Noisy natural gradients](https://arxiv.org/abs/1712.02390)** — 2017. Combines natural-gradient updates with adaptive weight noise to fit Gaussian variational posteriors. Includes factorized, full-covariance, and matrix-variate forms.
- **[Weight perturbation in Adam](https://arxiv.org/abs/1806.04854)** — 2018. Implements Gaussian mean-field variational inference through natural-gradient updates and perturbed weights, adapting familiar optimizer machinery for uncertainty estimation.
- **[Priors at the unit level](https://arxiv.org/abs/1810.05193)** — 2018. Shows that Gaussian weight priors with ReLU-like activations can induce increasingly heavy-tailed unit distributions with depth. Weight priors do not directly describe function-space behavior.
- **[Bayesian generalization perspective](https://arxiv.org/abs/2002.08791)** — 2020. Studies marginalization across neural-network solutions, functional priors, calibration, and double descent. Relates ensembles to approximate model averaging.

## Activations and normalization

- **[Rectified units in RBMs](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)** — 2010. Uses noisy rectified linear units to approximate replicated binary units in restricted Boltzmann machines, preserving information about relative input intensity.
- **[Deep rectifier networks](https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)** — 2011. Studies rectifying activations that produce exact zeros, including supervised learning without mandatory unsupervised pretraining.
- **[Leaky rectifiers](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)** — 2013. Retains a nonzero gradient for negative inputs. Compares rectifier variants and sigmoid-based acoustic models on conversational speech recognition.
- **[PReLU and rectifier initialization](https://arxiv.org/abs/1502.01852)** — 2015. Introduces a parametric rectifier and an initialization derived for rectifier networks. Studies their use in deep image classification.
- **[Randomized leaky ReLU](https://arxiv.org/abs/1505.00853)** — 2015. Randomizes the negative slope during training. Compares standard, leaky, parametric, and randomized rectifiers on image-classification tasks.
- **[ELU](https://arxiv.org/abs/1511.07289)** — 2015. Uses an identity branch for positive inputs and a saturating negative branch. Motivated by shifting average activations closer to zero.
- **[Parametric ELU](https://arxiv.org/abs/1605.09332)** — 2016. Learns the shape of the exponential-linear activation instead of fixing its parameters. Different layers can learn different nonlinear responses.
- **[S-shaped ReLU](https://arxiv.org/abs/1512.07030)** — 2015. Learns a three-segment piecewise-linear activation with four parameters, jointly with the network. Can represent convex and nonconvex shapes.
- **[Maxout](https://arxiv.org/abs/1302.4389)** — 2013. Takes the maximum over a set of inputs to form a learnable nonlinear unit. Designed and evaluated in conjunction with dropout.
- **[GELU](https://arxiv.org/abs/1606.08415)** — 2016. Weights an input by the standard Gaussian cumulative probability at that value. Provides a smooth alternative to ReLU's sign-based gating.
- **[SiLU](https://arxiv.org/abs/1702.03118)** — 2017. Multiplies an input by its sigmoid. The paper studies SiLU and its derivative as activations for reinforcement-learning function approximation.
- **[Swish](https://arxiv.org/abs/1710.05941)** — 2017. Searches activation functions and identifies input times sigmoid of a scaled input. The scale parameter generalizes the SiLU form.
- **[Mish](https://arxiv.org/abs/1908.08681)** — 2019. Uses input times tanh of softplus, producing a smooth nonmonotonic activation. Evaluated across image-classification and detection architectures.
- **[SELU](https://arxiv.org/abs/1706.02515)** — 2017. Uses scaled exponential-linear activations to obtain self-normalizing behavior under the paper's assumptions. The result is not a guarantee for arbitrary architectures or initialization.
- **[Gated feedforward layers](https://arxiv.org/abs/2002.05202)** — 2020. Compares GLU variants in Transformer feedforward sublayers. Combines two projections by elementwise multiplication, with a nonlinearity applied to one branch.
- **[Batch normalization](https://arxiv.org/abs/1502.03167)** — 2015. Normalizes layer inputs using mini-batch statistics as part of the architecture. Studies effects on training, learning-rate choice, and regularization.
- **[Why batch normalization helps](https://arxiv.org/abs/1805.11604)** — 2018. Challenges internal covariate shift as the main explanation and analyzes smoother optimization behavior. An analysis of mechanism, not a universal account of every normalization setup.
- **[Layer normalization](https://arxiv.org/abs/1607.06450)** — 2016. Computes normalization statistics across a layer for each example, with learned gain and bias. Avoids dependence on other examples in a mini-batch.
- **[RMSNorm](https://arxiv.org/abs/1910.07467)** — 2019. Normalizes activations by their root mean square with learned rescaling, omitting LayerNorm's mean subtraction; preserves rescaling invariance without recentering activations.

## Training, regularization, and generalization

- **[Unsupervised pretraining](https://jmlr.csail.mit.edu/papers/volume11/erhan10a/erhan10a.pdf)** — 2010. Tests how pretraining affects optimization and generalization. Experiments support a regularization interpretation through the basins reached during supervised training.
- **[Dropout](https://arxiv.org/abs/1207.0580)** — 2012. Randomly omits feature detectors during training to reduce co-adaptation. Trains shared weights across many perturbed network configurations.
- **[Dropout analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC3996711/)** — 2014. Analyzes Bernoulli gating, ensemble averaging, and regularization in linear and nonlinear networks. Distinguishes exact identities from approximations used for nonlinear activations.
- **[Adversarial training](https://arxiv.org/abs/1706.06083)** — 2017/2018. Optimizes prediction loss against worst-case perturbations in a specified threat set. Madry et al. approximate the inner attack with projected gradients; resistance to tested attacks is not a certificate against every possible perturbation.
- **[Differentially private SGD](https://arxiv.org/abs/1607.00133)** — 2016. Clips each example’s gradient, adds Gaussian noise and accounts for privacy loss over training. Abadi et al. trade accuracy against a stated privacy budget; the protected unit here is an example, not automatically an entire user.
- **[Adaptive dropout](https://papers.nips.cc/paper/5032-adaptive-dropout-for-training-deep-neural-networks.pdf)** — 2013. Learns a binary belief network that determines which hidden activities to suppress. The dropout mechanism is trained jointly with the predictive network.
- **[Fast dropout](https://nlp.stanford.edu/pubs/sidaw13fast.pdf)** — 2013. Approximates dropout's random contributions with a Gaussian distribution, sampling or integrating that approximation instead of repeatedly sampling unit masks.
- **[Neural tangent kernel](https://arxiv.org/abs/1806.07572)** — 2018. Describes function evolution during gradient training through a kernel. In the specified infinite-width limit the kernel becomes fixed; finite feature-learning networks need not follow that regime.
- **[Lazy training](https://arxiv.org/abs/1812.07956)** — 2018. Shows how parameter scaling can make training resemble a model's linearization at initialization. Its kernel-like behavior is not a complete explanation of neural feature learning.
- **[Double descent](https://arxiv.org/abs/1912.02292)** — 2019. Documents nonmonotonic test error as model size or training duration increases. The observed interpolation regimes qualify simple claims that more capacity always helps or hurts.
- **[Language-model scaling laws](https://arxiv.org/abs/2001.08361)** — 2020. Fits empirical relationships among cross-entropy loss, parameter count, data, and training compute. Extrapolations inherit the assumptions of the studied model and data regime.
- **[Compute-optimal training](https://arxiv.org/abs/2203.15556)** — 2022. Studies model size and token allocation under a fixed training budget, motivating Chinchilla. The measured trade-off concerns training compute, not a universal deployment optimum.

## Credit assignment and alternatives to backpropagation

- **[Backpropagation](https://www.nature.com/articles/323533a0)** - 1986. Applies the chain rule backward through network layers to compute weight gradients. Credit assignment determines the learning signal; an optimizer such as SGD or Adam uses that signal to update weights.

### Perturbation-based learning

- **[Node versus weight perturbation](https://proceedings.neurips.cc/paper/2003/file/f8b932c70d0b2e6bf071729a4fa68dfc-Paper.pdf)** - 2003. Uses loss changes after perturbing activations or weights. In linear teacher-student networks, maximum stable learning rates scale inversely with noise dimension; this is not a transformer scaling law.
- **[MeZO](https://arxiv.org/abs/2305.17333)** - 2023. Fine-tunes pretrained language models with [SPSA-style](https://mlpapers.org/optimization/#perturbation-based-gradient-estimation) loss differences. Replays random seeds for in-place weight perturbations, avoiding stored gradients. The memory saving does not establish efficient training from scratch.
- **[EGGROLL](https://arxiv.org/abs/2511.16652)** - 2025 preprint. Uses low-rank weight perturbations for evolution strategies, reducing perturbation storage and evaluation cost relative to full-rank ES. Its reported ES speedups do not establish an advantage over backpropagation.
- **[Dust](https://qlabs.sh/research/dust)** - 2026 technical report. Perturbs activations independently per token and estimates updates from loss changes. Large populations approach backprop with SGD at extra compute cost; the reported runs still trail Adam.

### Feedback signals and learned targets

- **[Feedback alignment](https://arxiv.org/abs/1411.0247)** - 2014/2016. Sends errors through fixed random feedback weights instead of transposed forward weights. [Direct feedback alignment](https://proceedings.neurips.cc/paper_files/paper/2016/file/d490d7b4576290fa60eb31b5fc917ad1-Paper.pdf) (2016) projects output errors to each hidden layer; both retain local activation derivatives.
- **[Difference target propagation](https://arxiv.org/abs/1412.7525)** - 2014/2015. Learns approximate inverse maps to send activation targets backward. A difference correction compensates for imperfect inverses; local fitting and the original top-layer update still use derivatives.
- **[Synthetic gradients](https://proceedings.mlr.press/v70/jaderberg17a.html)** - 2016/2017. Learns predictors of downstream error gradients so modules can update without waiting for downstream computation. Gradient predictors need training signals; this decouples backpropagation rather than eliminating it.

### Local objectives, forward differentiation, and equilibrium learning

- **[Forward-Forward](https://arxiv.org/abs/2212.13345)** - 2022. Each layer learns to raise a goodness score on positive data and lower it on negative data. Replaces end-to-end error propagation with local objectives; negative-example construction is part of the method.
- **[Forward gradients](https://arxiv.org/abs/2202.08587)** - 2022. Forward-mode autodiff computes a directional derivative; multiplying by its random direction gives an unbiased estimate for independent zero-mean, unit-variance tangent components. Removes reverse-mode differentiation, not derivatives.
- **[Local forward gradients](https://www.mengyeren.com/research/2023/scaling-forward-gradient-with-local-losses/)** - 2022/2023. Ren et al. reduce estimator variance with activity perturbations and local greedy losses. Forward-mode autodiff supplies the directional derivatives; ImageNet performance improves over tested alternatives but still trails backprop.
- **[Equilibrium propagation](https://arxiv.org/abs/1602.05179)** - 2016/2017. Contrasts free and weakly nudged equilibria in an energy-based network. Under smoothness and fixed-point assumptions, the update approaches the objective gradient as nudging vanishes; settling has a computation cost.

### Comparative evidence

- **[Scaling evidence](https://proceedings.neurips.cc/paper/2018/file/63c3ddcc7b23daa1e42dc41f9a44a873-Paper.pdf)** - 2018. Bartunov et al. find tested target-propagation and feedback-alignment variants far behind backprop on ImageNet. Results concern their architectures and tuning; direct feedback alignment exceeded their 16GB GPU memory.
- **[Noise](https://github.com/qlabs-eng/noise/tree/395b5f609b115a908337208bc35aae1da18f63dc)** - 2026 benchmark. Fixed 37.7M transformer, ~10M tokens, 8 H100s; final one-epoch loss or time to loss 4.2. Rules ban autodiff and chained nonlinear backward credit but allow exact linear credit. Public validation, seed 0; scripts pinned.
- **[Noise snapshot](https://github.com/qlabs-eng/noise/blob/395b5f609b115a908337208bc35aae1da18f63dc/README.md#track-1-validation-loss)** - 2026-10-21. Best validation loss: 3.777; listed Adam: 3.769. A separate author-reported [backprop recipe](https://github.com/spicylemonade/boost-cut/blob/539386e7faf07767f48f8a443a88a3c72b7ef159/results/bp38_s0/result.json) scores 3.708 on the same split. Compute and tuning differ; public scoring permits adaptive overfitting.

## Pruning, quantization, and efficient execution

- **[Optimal Brain Damage](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)** — 1989. Uses second-derivative information to remove weights while balancing network complexity against training error.
- **[Prune and retrain](https://arxiv.org/abs/1506.02626)** — 2015. Trains a dense network, removes unimportant connections, then fine-tunes remaining weights. Reduced parameter counts do not alone establish hardware speedups.
- **[Taylor-based filter pruning](https://arxiv.org/abs/1611.06440)** — 2016. Approximates the loss change from pruning with a Taylor criterion and interleaves removal with fine-tuning. Evaluated for adapted convolutional networks and classification.
- **[L0 regularization](https://arxiv.org/abs/1712.01312)** — 2017. Uses differentiable expectations over stochastic gates to encourage exact zeros during training. The hard-concrete relaxation enables gradient-based structure learning.
- **[Lottery ticket hypothesis](https://arxiv.org/abs/1803.03635)** — 2018. Finds sparse subnetworks whose original initializations permit effective retraining. The experiments support an initialization-sensitive hypothesis, not a universal pruning guarantee.
- **[ReplaceMe](https://arxiv.org/abs/2505.02819)** - 2025. Replaces transformer blocks with linear maps fitted to calibration activations; compatible maps merge into existing projections. Least-squares fitting is closed form; cosine-distance fitting uses optimization.
- **[Binarized networks](https://arxiv.org/abs/1602.02830)** — 2016. Trains networks with binary weights and activations at runtime, replacing many arithmetic operations with bitwise ones. Actual efficiency depends on implementation and hardware.
- **[Compact binary-network training](https://ojs.aaai.org/index.php/AAAI/article/view/10862)** — 2017. Studies low learning rates, PReLU, bipolar regularization, output scaling, and multiple activation binarizations to improve the compression–accuracy trade-off.
- **[Knowledge distillation](https://arxiv.org/abs/1503.02531)** — 2015. Transfers a teacher model or ensemble's predictions into a single model. Separates the cost of producing training targets from the cost of deploying the student.
- **[GPTQ](https://arxiv.org/abs/2210.17323)** — 2022. Uses approximate second-order information for one-shot post-training weight quantization of large Transformers. Low-bit accuracy and speed depend on the model and inference implementation.
- **[AWQ](https://arxiv.org/abs/2306.00978)** — 2023. Uses activation statistics to identify and scale important weight channels before low-bit weight-only quantization. Avoids backpropagation during calibration.
- **[LoRA](https://arxiv.org/abs/2106.09685)** — 2021. Freezes pretrained weights and learns low-rank update matrices for adaptation, reducing trainable parameters and optimizer state.
- **[QLoRA](https://arxiv.org/abs/2305.14314)** — 2023. Backpropagates through a frozen 4-bit model into low-rank adapters. Combines NormalFloat quantization, quantized scaling constants, and paged optimizers to reduce training memory.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)** — 2022. Tiles exact attention to reduce traffic between GPU memory levels. Changes execution and memory use without replacing dense attention by an approximate attention rule.
- **[Weightless neural systems](https://www.esann.org/sites/default/files/proceedings/legacy/es2014-7.pdf)** — 2014. Reviews RAM-based neurons addressed by Boolean inputs, including WiSARD and related architectures. Covers capacity analysis, extensions, and applications.
- **[KD-Lib](https://arxiv.org/abs/2011.14691)** — 2020. A PyTorch library collecting knowledge distillation, pruning, and quantization methods, with support for tuning and experiment logging.
- **[Activation checkpointing](https://arxiv.org/abs/1604.06174)** — 2016. Stores selected forward activations and recomputes intermediate values during backpropagation, trading additional computation for lower training memory.

## Ensembles

- **[Neural-network ensembles](https://machine-learning.martinsewell.com/ensembles/HansenSalamon1990.pdf)** — 1990. Studies cross-validation for architecture selection and consensus predictions from multiple trained networks to reduce residual classification error.
- **[When networks disagree](https://www.researchgate.net/publication/2438296_When_Networks_Disagree_Ensemble_Methods_for_Hybrid_Neural_Networks)** — 1992. Develops a framework for combining regression estimators, with closed-form ensemble parameters under its assumptions. The cited manuscript is the 1992 technical report.
- **[Ensemble ambiguity and active learning](https://papers.nips.cc/paper/1001-neural-network-ensembles-cross-validation-and-active-learning.pdf)** — 1994. Uses disagreement among network outputs with cross-validation to study ensemble error, combination weights, and selection of examples for labeling.
- **[Ensemble size versus model size](https://arxiv.org/abs/2005.00570)** — 2020. Compares small-model ensembles with larger single models on CIFAR-10 and ImageNet. Reported accuracy–FLOP trade-offs do not establish a universal deployment advantage.

## Related Topics

- [Optimization](https://mlpapers.org/optimization/)
- [Ensemble Learning](https://mlpapers.org/ensemble-learning/)
- [Bayesian Inference](https://mlpapers.org/bayesian-inference/)
- [Reinforcement Learning](https://mlpapers.org/reinforcement-learning/)
- [Interpretability](https://mlpapers.org/interpretability/)
- [NLP](https://mlpapers.org/nlp/)
- [Computer Vision](https://mlpapers.org/computer-vision/)
- [Generative Models](https://mlpapers.org/generative-models/)
- [Graph Neural Networks](https://mlpapers.org/graph-neural-networks/)
