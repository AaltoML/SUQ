# SUQ: Streamlined Uncertainty Quantification

![image](suq.png)

This repository contains an open-source library implementation of Streamlined Uncertainty Quantification (SUQ) used in the paper *Streamlining Prediction in Bayesian Deep Learning* accepted at ICLR 2025.

<table>
<tr>
	<td>
   		<strong> Streamlining Prediction in Bayesian Deep Learning</strong><br>
            Rui Li, Marcus Klasson, Arno Solin, Martin Trapp<br>
		<strong>International Conference on Learning Representations (ICLR 2025)</strong><br>
		<a href="https://arxiv.org/abs/2411.18425"><img alt="Paper" src="https://img.shields.io/badge/-Paper-gray"></a>
		<a href="https://github.com/AaltoML/suq"><img alt="Code" src="https://img.shields.io/badge/-Code-gray" ></a>
		</td>
    </tr>
</table>

## SUQ Library
### 📦 Installation
To install SUQ with `pip`, run the following
```bash
pip install suq
```

Alternatively, you can install the latest development version directly from GitHub:
```
git clone https://github.com/AaltoML/SUQ.git
cd SUQ
pip install -e .
```

### 🚀 Simple Usage

```python
from suq import streamline_mlp, streamline_vit

# Load your model and estimated posterior
model = ...
posterior = ...

# Wrap an MLP model with SUQ
suq_model = streamline_mlp(
    model=model,
    posterior=posterior,
    covariance_structure='diag',       # currently only 'diag' is supported
    likelihood='classification'        # or 'regression'
)

# Wrap a Vision Transformer with SUQ
suq_model = streamline_vit(
    model=model,
    posterior=posterior,
    covariance_structure='diag',      # currently only 'diag' is supported
    likelihood='classification',      
    MLP_deterministic=True,
    Attn_deterministic=False,
    attention_diag_cov=False,
    num_det_blocks=10
)

# fit scale factor
suq_model.fit(train_loader, scale_fit_epoch, scale_fit_lr)

# Make a prediction
pred = suq_model(X)
```

📄 See [`examples/mlp_la_example.py`](examples/mlp_la_example.py), [`examples/vit_la_example.py`](examples/vit_la_example.py), [`examples/mlp_vi_example.py`](examples/mlp_vi_example.py), and [`examples/vit_vi_example.py`](examples/vit_vi_example.py) for full, self-contained examples that cover:
- Training the MAP model
- Estimating the posterior with Laplace or IVON (mean field VI)
- Wrapping the model into a streamlined SUQ version


> ⚠️ **Note on Vision Transformer Support**  
Currently, SUQ only supports Vision Transformers implemented in the same style as [`examples/vit_model.py`](examples/vit_model.py). If you're using a different ViT implementation, compatibility is not guaranteed.

### 🛠️ TODO
- Extend support to other Transformer implementations
- Kronecker covariance
- Full covariance


## Citation

```bibtex
@inproceedings{li2025streamlining,
  title = {Streamlining Prediction in Bayesian Deep Learning},
  author = {Rui Li, Marcus Klasson, Arno Solin and Martin Trapp},
  booktitle = {International Conference on Learning Representations ({ICLR})},
  year = {2025}
}
```

## License
This software is provided under the MIT license.
