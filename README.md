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
Install the stable version with `pip`:
```bash
pip install suq
```

Or install the latest development version from source:
```bash
git clone https://github.com/AaltoML/SUQ.git
cd SUQ
pip install -e .
```

### 🚀 Simple Usage
#### Streamline Whole Network
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

# Fit scale factor
suq_model.fit(train_loader, scale_fit_epoch, scale_fit_lr)

# Make a prediction
pred = suq_model(X)
```

📄 See [`examples/mlp_la_example.py`](examples/mlp_la_example.py), [`examples/vit_la_example.py`](examples/vit_la_example.py), [`examples/mlp_vi_example.py`](examples/mlp_vi_example.py), and [`examples/vit_vi_example.py`](examples/vit_vi_example.py) for full, self-contained examples that cover:
- Training the MAP model
- Estimating the posterior with Laplace or IVON (mean field VI)
- Wrapping the model into a streamlined SUQ version


> ❗ **Note on Vision Transformer Support**  
Currently, SUQ only supports Vision Transformers implemented in the same style as [`examples/vit_model.py`](examples/vit_model.py). If you're using a different ViT implementation, compatibility is not guaranteed.

#### Streamline Individual Layers

In addition to wrapping full models like MLPs or ViTs, SUQ allows you to manually wrap individual layers in your own networks.

You can directly import supported modules from `suq.streamline_layer`.

Supported Layers:

| Layer Type         | SUQ Wrapper                   |
|--------------------|-------------------------------|
| `nn.Linear`        | `SUQ_Linear_Diag`             |
| `nn.ReLU`, etc.    | `SUQ_Activation_Diag`         |
| `nn.BatchNorm1d`   | `SUQ_BatchNorm_Diag`          |
| `nn.LayerNorm`     | `SUQ_LayerNorm_Diag`          |
| `MLP (Transformer block)`    | `SUQ_TransformerMLP_Diag`     |
| `Attention`    | `SUQ_Attention_Diag`          |
| `Transformer block`  | `SUQ_Transformer_Block_Diag`  |
| `Final classifier`   | `SUQ_Classifier_Diag`         |

Example:

```python
from suq.streamline_layer import SUQ_Linear_Diag

# Define a standard linear layer
linear_layer = nn.Linear(100, 50)
# Provide posterior variances for weights and biases
w_var = torch.rand(50, 100)
b_var = torch.rand(50)

# Wrap the layer with SUQ's linear module
streamlined_layer = SUQ_Linear_Diag(linear_layer, w_var, b_var)

# Provide input mean and variance (e.g., from a previous layer)
input_mean = torch.randn(32, 100)
input_var = torch.rand(32, 100)

# Forward pass through the streamlined layer
pred_mean, pred_var = streamlined_layer(input_mean, input_var)
```

### 🛠️ TODO
- Extend support to other Transformer implementations
- Add Kronecker covariance
- Add full covariance


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
