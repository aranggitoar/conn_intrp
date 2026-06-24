"""
SmolVLM2 model adapter.

Wraps SmolVLM2 (single linear connector via ``modality_projection.proj``)
for use with the shared pipeline in :mod:`conn_intrp.dm` and
:mod:`conn_intrp.ablation`.

Example::

    >>> from conn_intrp.models import SmolVLM2Adapter
    >>> adapter = SmolVLM2Adapter("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    >>> inputs = adapter.preprocess(batch, image_base_path)

Main Classes:
    SmolVLM2Adapter: ModelAdapter for SmolVLM2 single-linear connector.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..data import HARNESS_PROMPT
from .base import ModelAdapter, SVDLayer


class SmolVLM2Adapter(ModelAdapter):
    """
    Adapter for SmolVLM2 with a single linear projection connector.

    SVD is performed on ``connector.modality_projection.proj``.
    :meth:`pre_svd_forward` is identity since there are no preceding layers.

    :param repo_id: HuggingFace model repository ID.
    :type repo_id: str
    :param dtype: Model compute dtype.
    :type dtype: torch.dtype
    :param model_kwargs: Extra keyword arguments forwarded to
        ``AutoModelForImageTextToText.from_pretrained``.
    """

    def __init__(
        self,
        repo_id: str,
        dtype: torch.dtype = torch.float16,
        cache_images: bool = True,
        **model_kwargs,
    ):
        self.repo_id = repo_id
        self.compute_dtype = dtype

        self.processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        self.processor.image_processor.do_image_splitting = False

        self.model = AutoModelForImageTextToText.from_pretrained(
            repo_id, dtype=dtype, **model_kwargs
        ).cuda()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        proj = self.model.model.connector.modality_projection.proj
        self.U, self.S, self.Vt = torch.linalg.svd(proj.weight.float(), full_matrices=False)
        self.proj_bias = proj.bias

        self.model_name = "smolvlm2"
        self.component_name = "proj"
        self.vocab_size = self.model.model.text_model.embed_tokens.num_embeddings
        self.n_dirs = self.S.shape[0]

        vis = self.model.config.vision_config
        self.patch_size = vis.patch_size
        self.scale_factor = self.model.config.scale_factor
        self.n_patches = (vis.image_size // vis.patch_size) ** 2 // self.scale_factor**2

        self.cache_images = cache_images
        self._image_cache = {}
        self._vision_cache = {}

    # --- Model-specific -------------------------------------------------------

    @property
    def svd_layers(self) -> list[SVDLayer]:
        return [SVDLayer("proj", self.U, self.S, self.Vt, self.proj_bias, self.n_dirs)]

    def run_connector_layer_masked(
        self,
        vision_out: torch.Tensor,
        layer_name: str,
        W_masked: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.pre_svd_forward(vision_out).to(W_masked.dtype)
        bias = self.proj_bias.to(W_masked.dtype) if self.proj_bias is not None else None
        return F.linear(hidden, W_masked, bias)

    def _load_image(self, img_path: str) -> Image.Image:
        if self.cache_images:
            if img_path not in self._image_cache:
                self._image_cache[img_path] = Image.open(img_path).convert("RGB")
            return self._image_cache[img_path]
        return Image.open(img_path).convert("RGB")

    def preprocess(self, batch: list[dict], image_base_path: Path) -> dict:
        """
        Tokenize a batch of image–question pairs via ``apply_chat_template``.

        :param batch: Data dicts with ``"image"`` and ``"question"`` keys.
        :type batch: list[dict]
        :param image_base_path: Root directory for image files.
        :type image_base_path: Path
        :returns: Batched, padded inputs on CUDA.
        :rtype: dict
        """
        prompts = []
        for datum in batch:
            img_path = str(image_base_path / datum["image"])
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": HARNESS_PROMPT},
                            {"type": "image", "image": self._load_image(img_path)},
                            {"type": "text", "text": datum["question"]},
                        ],
                    }
                ]
            )
        return self.processor.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            padding=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

    def extract_vision(self, inputs: dict) -> torch.Tensor:
        """
        Run SigLIP vision encoder + pixel shuffle.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: Vision features of shape ``(B, n_patches, C_vision)``.
        :rtype: torch.Tensor
        """
        B, N, C, H, W = inputs["pixel_values"].shape
        pixel_values = inputs["pixel_values"].view(B * N, C, H, W)
        B, N, H, W = inputs["pixel_attention_mask"].shape
        pixel_attention_mask = inputs["pixel_attention_mask"].view(B * N, H, W)
        patches_subgrid = pixel_attention_mask.unfold(
            dimension=1, size=self.patch_size, step=self.patch_size
        ).unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        patch_attn = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        vision_out = self.model.model.vision_model(pixel_values.to(self.compute_dtype), patch_attn)
        return self.model.model.connector.pixel_shuffle(
            vision_out.last_hidden_state, self.scale_factor
        )

    def pre_svd_forward(self, vision_out: torch.Tensor) -> torch.Tensor:
        """
        Identity — single linear connector has no preceding layers.

        :param vision_out: Output of :meth:`extract_vision`.
        :type vision_out: torch.Tensor
        :returns: *vision_out* unchanged.
        :rtype: torch.Tensor
        """
        return vision_out

    def run_connector(self, vision_out: torch.Tensor) -> torch.Tensor:
        """
        Full linear projection via ``modality_projection.proj``.

        :param vision_out: Output of :meth:`extract_vision`.
        :type vision_out: torch.Tensor
        :returns: Connector output in language model embedding space.
        :rtype: torch.Tensor
        """
        return self.model.model.connector.modality_projection.proj(
            vision_out.to(self.compute_dtype)
        )

    def get_text_embeds(self, inputs: dict) -> torch.Tensor:
        """
        Look up text token embeddings.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: Embeddings of shape ``(B, N, C)``.
        :rtype: torch.Tensor
        """
        return self.model.model.text_model.get_input_embeddings()(inputs["input_ids"]).to(
            inputs["input_ids"].device
        )

    def merge_embeds(
        self,
        inputs: dict,
        text_embeds: torch.Tensor,
        conn_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge via SmolVLM2's built-in ``inputs_merger``.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :param text_embeds: Text embeddings from :meth:`get_text_embeds`.
        :type text_embeds: torch.Tensor
        :param conn_out: Connector output to insert at image positions.
        :type conn_out: torch.Tensor
        :returns: Merged embeddings of shape ``(B, N, C)``.
        :rtype: torch.Tensor
        """
        return self.model.model.inputs_merger(
            input_ids=inputs["input_ids"],
            inputs_embeds=text_embeds,
            image_hidden_states=conn_out.to(
                dtype=self.model.model.dtype, device=text_embeds.device
            ),
        )

    def generate(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> list[str]:
        """
        Generate and decode text.

        :param embeds: Merged input embeddings.
        :type embeds: torch.Tensor
        :param attention_mask: Attention mask.
        :type attention_mask: torch.Tensor
        :param max_new_tokens: Maximum tokens to generate.
        :type max_new_tokens: int
        :returns: Decoded prediction strings.
        :rtype: list[str]
        """
        out = self.model.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        return self.processor.batch_decode(out, skip_special_tokens=True)

    def generate_with_logits(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> tuple[list[str], torch.Tensor]:
        gen_out = self.model.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
        preds = self.processor.batch_decode(gen_out.sequences, skip_special_tokens=True)
        return preds, gen_out.scores[0]

    def get_logits(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Last-position logits, handling both left and right padding.

        :param embeds: Merged input embeddings.
        :type embeds: torch.Tensor
        :param attention_mask: Attention mask.
        :type attention_mask: torch.Tensor
        :returns: Logits of shape ``(B, vocab_size)``.
        :rtype: torch.Tensor
        """
        text_out = self.model.model.text_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if self.processor.tokenizer.padding_side == "right":
            last_pos = attention_mask.sum(dim=1) - 1
            B = embeds.shape[0]
            return torch.stack(
                [self.model.lm_head(text_out[0][b, last_pos[b], :]) for b in range(B)]
            )
        return self.model.lm_head(text_out[0][:, -1, :])

    def compute_coefficients_per_layer(self, inputs: dict) -> dict[str, torch.Tensor]:
        return {"proj": self.compute_coefficients(inputs)}

    def forward_connector_from(self, layer_name: str, layer_output: torch.Tensor) -> torch.Tensor:
        return layer_output

    def compute_probe_projections(self, inputs: dict) -> dict[str, torch.Tensor]:
        """
        Cosine similarity of pixel-shuffled vision output with each right
        singular vector of ``proj``.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: ``{"proj": tensor (B, n_patches, n_dirs)}``.
        :rtype: dict[str, torch.Tensor]
        """
        vision_out = self.extract_vision(inputs).float()
        x_norm = vision_out / (vision_out.norm(dim=-1, keepdim=True) + 1e-8)
        return {"proj": x_norm @ self.Vt.T.float()}
