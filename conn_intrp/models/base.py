"""
Base model adapter for connector interpretability.

Defines the interface that each VLM adapter must implement. The shared
pipeline code in :mod:`conn_intrp.dm` and :mod:`conn_intrp.ablation`
calls only these methods, keeping pipeline logic model-agnostic.

Example::

    >>> from conn_intrp.models import SmolVLM2Adapter
    >>> adapter = SmolVLM2Adapter("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    >>> inputs = adapter.preprocess(batch, image_base_path)
    >>> coefficients = adapter.compute_coefficients(inputs)

Main Classes:
    ModelAdapter: Abstract base with default SVD operations.
    SVDLayer: Descriptor for one maskable linear layer in the connector.
"""

import torch
from dataclasses import dataclass
from torch.nn import functional as F
from pathlib import Path


@dataclass
class SVDLayer:
    """
    Descriptor for one SVD-decomposed linear layer in the connector.

    :param name: Layer identifier (e.g. ``"proj"``, ``"linear_1"``).
    :type name: str
    :param U: Left singular vectors.
    :type U: torch.Tensor
    :param S: Singular values.
    :type S: torch.Tensor
    :param Vt: Right singular vectors (transposed).
    :type Vt: torch.Tensor
    :param bias: Layer bias.
    :type bias: torch.Tensor
    :param n_dirs: Number of SVD directions (``S.shape[0]``).
    :type n_dirs: int
    """

    name: str
    U: torch.Tensor
    S: torch.Tensor
    Vt: torch.Tensor
    bias: torch.Tensor | None
    n_dirs: int


class ModelAdapter:
    """
    Abstract base adapter for VLM connector interpretability.

    Subclasses implement model-specific operations (vision encoding,
    connector architecture, embedding merging, generation). Three SVD
    helpers have default implementations that work for any connector
    whose last layer is decomposed as ``U @ diag(S) @ Vt``.

    The ``U``, ``S``, ``Vt``, ``proj_bias``, ``n_dirs``, and
    ``component_name`` attributes always refer to the **last** linear
    layer in the connector (used by :mod:`conn_intrp.ablation`).
    For per-layer access (used by :mod:`conn_intrp.dm`), use
    :attr:`svd_layers`.

    :ivar U: Left singular vectors of the last decomposed layer.
    :ivar S: Singular values of the last decomposed layer.
    :ivar Vt: Right singular vectors of the last decomposed layer.
    :ivar proj_bias: Bias of the last decomposed layer.
    :ivar vocab_size: Language model vocabulary size.
    :ivar n_dirs: Number of SVD directions in the last layer.
    :ivar n_patches: Number of spatial patches per image.
    :ivar model_name: Short identifier for logging (e.g. ``"smolvlm2"``).
    :ivar component_name: Last connector layer name (e.g. ``"proj"``, ``"linear_2"``).
    :ivar compute_dtype: Model compute dtype (``torch.float16`` or ``torch.bfloat16``).
    """

    U: torch.Tensor
    S: torch.Tensor
    Vt: torch.Tensor
    proj_bias: torch.Tensor | None
    vocab_size: int
    n_dirs: int
    n_patches: int
    model_name: str
    component_name: str
    compute_dtype: torch.dtype

    # --- Must override --------------------------------------------------------

    def preprocess(self, batch: list[dict], image_base_path: Path) -> dict:
        """
        Format prompts, load images, and tokenize.

        :param batch: List of data dicts with ``"image"`` and ``"question"`` keys.
        :type batch: list[dict]
        :param image_base_path: Root directory for image files.
        :type image_base_path: Path
        :returns: Tokenized inputs on CUDA.
        :rtype: dict
        """
        raise NotImplementedError

    def extract_vision(self, inputs: dict) -> torch.Tensor:
        """
        Run the vision encoder and pixel shuffle.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: Pre-connector vision features.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def pre_svd_forward(self, vision_out: torch.Tensor) -> torch.Tensor:
        """
        Layers between vision output and the SVD-decomposed layer.

        Identity for single-linear connectors (SmolVLM2).
        ``LN → linear_1 → GELU`` for MLP connectors (InternVL).

        :param vision_out: Output of :meth:`extract_vision`.
        :type vision_out: torch.Tensor
        :returns: Input to the SVD-decomposed layer.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def run_connector(self, vision_out: torch.Tensor) -> torch.Tensor:
        """
        Full connector forward pass (unmasked).

        :param vision_out: Output of :meth:`extract_vision`.
        :type vision_out: torch.Tensor
        :returns: Connector output embeddings.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def get_text_embeds(self, inputs: dict) -> torch.Tensor:
        """
        Compute text token embeddings from ``input_ids``.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: Text embeddings of shape ``(B, N, C)``.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def merge_embeds(
        self, inputs: dict, text_embeds: torch.Tensor,
        conn_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge connector output into text embeddings.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :param text_embeds: Text embeddings from :meth:`get_text_embeds`.
        :type text_embeds: torch.Tensor
        :param conn_out: Connector output to insert at image positions.
        :type conn_out: torch.Tensor
        :returns: Merged embeddings of shape ``(B, N, C)``.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def generate(
        self, embeds: torch.Tensor, attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> list[str]:
        """
        Generate text from merged embeddings.

        :param embeds: Merged input embeddings.
        :type embeds: torch.Tensor
        :param attention_mask: Attention mask.
        :type attention_mask: torch.Tensor
        :param max_new_tokens: Maximum tokens to generate.
        :type max_new_tokens: int
        :returns: Decoded prediction strings.
        :rtype: list[str]
        """
        raise NotImplementedError

    def get_logits(
        self, embeds: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        First-token logits from the language model.

        :param embeds: Merged input embeddings.
        :type embeds: torch.Tensor
        :param attention_mask: Attention mask.
        :type attention_mask: torch.Tensor
        :returns: Logits of shape ``(B, vocab_size)``.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @property
    def svd_layers(self) -> list[SVDLayer]:
        """
        All maskable linear layers in the connector, ordered input-to-output.

        Single-linear connectors return one entry; MLP connectors return
        one per linear layer.

        :returns: List of :class:`SVDLayer` descriptors.
        :rtype: list[SVDLayer]
        """
        raise NotImplementedError

    def run_connector_layer_masked(
        self, vision_out: torch.Tensor, layer_name: str,
        W_masked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full connector, replacing one layer's weight with *W_masked*.

        All other layers use their original weights.

        :param vision_out: Output of :meth:`extract_vision`.
        :type vision_out: torch.Tensor
        :param layer_name: Which layer to replace (must match an
            :attr:`SVDLayer.name` from :attr:`svd_layers`).
        :type layer_name: str
        :param W_masked: Reconstructed weight with masked singular values.
        :type W_masked: torch.Tensor
        :returns: Connector output embeddings.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    # --- Default implementations (override if needed) -------------------------

    def run_connector_masked(
        self, vision_out: torch.Tensor, W_masked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Connector forward with a masked weight matrix.

        Default: ``F.linear(pre_svd_forward(vision_out), W_masked, proj_bias)``.

        :param vision_out: Output of :meth:`extract_vision`.
        :type vision_out: torch.Tensor
        :param W_masked: Reconstructed weight with masked singular values.
        :type W_masked: torch.Tensor
        :returns: Connector output embeddings.
        :rtype: torch.Tensor
        """
        hidden = self.pre_svd_forward(vision_out)
        bias = self.proj_bias.to(hidden.dtype) if self.proj_bias is not None else None
        return F.linear(hidden, W_masked, bias)

    def compute_coefficients(self, inputs: dict) -> torch.Tensor:
        """
        Compute SVD coefficients: ``S * (pre_svd_output @ Vt.T)``.

        Default: chains :meth:`extract_vision` and :meth:`pre_svd_forward`.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: Coefficients of shape ``(B, n_patches, n_dirs)``.
        :rtype: torch.Tensor
        """
        vision_out = self.extract_vision(inputs)
        hidden = self.pre_svd_forward(vision_out)
        return self.S.to(hidden.dtype) * (hidden @ self.Vt.T.to(hidden.dtype))

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct connector output from (possibly ablated) coefficients.

        Default: ``coefficients @ U.T + proj_bias``.

        :param coefficients: SVD coefficients, possibly with ablated directions.
        :type coefficients: torch.Tensor
        :returns: Reconstructed connector output.
        :rtype: torch.Tensor
        """
        out = coefficients @ self.U.T.to(coefficients.dtype)
        if self.proj_bias is not None:
            out = out + self.proj_bias.to(coefficients.dtype)
        return out

    def compute_probe_projections(self, inputs: dict) -> dict[str, torch.Tensor]:
        """
        L2-normalised projection of each connector layer's input onto
        the layer's right singular vectors (rows of Vt).

        Returns cosine similarities in ``[-1, 1]``. Each adapter returns
        one entry per linear layer in the connector (e.g. ``{"proj": ...}``
        for SmolVLM2, ``{"linear_1": ..., "linear_2": ...}`` for InternVL).

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: Map of layer name to projection tensor of shape
            ``(B, n_patches, n_dirs_for_layer)``.
        :rtype: dict[str, torch.Tensor]
        """
        raise NotImplementedError
