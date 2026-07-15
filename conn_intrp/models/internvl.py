"""
InternVL3.5 model adapter.

Wraps InternVL3.5 (MLP connector: ``layer_norm → linear_1 → GELU →
linear_2``) for use with the shared pipeline. SVD is performed on
``linear_2``; :meth:`~InternVLAdapter.pre_svd_forward` runs the
preceding ``LN → linear_1 → GELU`` layers.

Example::

    >>> from conn_intrp.models import InternVLAdapter
    >>> adapter = InternVLAdapter("OpenGVLab/InternVL3-2B-HF")
    >>> inputs = adapter.preprocess(batch, image_base_path)

Main Classes:
    InternVLAdapter: ModelAdapter for InternVL3.5 MLP connector.

Main Functions:
    load_image: Load and tile an image for InternVL dynamic preprocessing.
"""

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..data import HARNESS_PROMPT
from .base import ModelAdapter, SVDLayer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- InternVL image preprocessing -------------------------------------------


def _build_transform(input_size: int) -> T.Compose:
    """
    Build the InternVL image transform pipeline.

    :param input_size: Target spatial size in pixels.
    :type input_size: int
    :returns: Composed transform (RGB convert → resize → tensor → normalise).
    :rtype: torchvision.transforms.Compose
    """
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """
    Find the target tile ratio closest to *aspect_ratio*.

    :param aspect_ratio: Source image aspect ratio.
    :type aspect_ratio: float
    :param target_ratios: Candidate ``(rows, cols)`` tile configurations.
    :type target_ratios: list[tuple[int, int]]
    :param width: Source image width.
    :type width: int
    :param height: Source image height.
    :type height: int
    :param image_size: Per-tile size in pixels.
    :type image_size: int
    :returns: Best-matching ``(rows, cols)`` ratio.
    :rtype: tuple[int, int]
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    """
    Tile an image into sub-crops following InternVL's dynamic resolution.

    :param image: Input PIL image.
    :type image: Image.Image
    :param min_num: Minimum number of tiles.
    :type min_num: int
    :param max_num: Maximum number of tiles.
    :type max_num: int
    :param image_size: Per-tile spatial size.
    :type image_size: int
    :param use_thumbnail: Whether to append a downscaled thumbnail.
    :type use_thumbnail: bool
    :returns: List of cropped (and optionally thumbnailed) PIL images.
    :rtype: list[Image.Image]
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(
    image_file: str | Path,
    input_size: int = 448,
    max_num: int = 12,
) -> torch.Tensor:
    """
    Load, tile, transform, and stack an image for InternVL.

    :param image_file: Path to the image file.
    :type image_file: str | Path
    :param input_size: Per-tile spatial size in pixels.
    :type input_size: int
    :param max_num: Maximum number of tiles.
    :type max_num: int
    :returns: Stacked pixel values of shape ``(n_tiles, 3, input_size, input_size)``.
    :rtype: torch.Tensor
    """
    image = Image.open(image_file).convert("RGB")
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# --- Adapter -----------------------------------------------------------------

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class InternVLAdapter(ModelAdapter):
    """
    Adapter for InternVL3.5 with an MLP connector.

    The connector is ``layer_norm → linear_1 → GELU → linear_2``.
    SVD is performed on ``linear_2``; :meth:`pre_svd_forward` runs
    the preceding layers.

    :param model_id: HuggingFace repo ID or local path (``-HF`` variant).
    :type model_id: str
    :param dtype: Model compute dtype.
    :type dtype: torch.dtype
    :param prompt: System prompt prepended to each question.
    :type prompt: str
    :param model_kwargs: Extra keyword arguments forwarded to
        ``AutoModelForImageTextToText.from_pretrained``.
    """

    def __init__(
        self,
        model_id: str,
        dtype: torch.dtype = torch.bfloat16,
        cache_images: bool = True,
        prompt: str = HARNESS_PROMPT,
        **model_kwargs,
    ):
        self.model_id = model_id
        self.compute_dtype = dtype
        self.prompt = prompt

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.processor.image_processor.max_patches = 1
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=True, **model_kwargs
        ).cuda()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        self.mmp = self.model.model.multi_modal_projector
        self.U, self.S, self.Vt = torch.linalg.svd(
            self.mmp.linear_2.weight.float(), full_matrices=False
        )
        self.proj_bias = self.mmp.linear_2.bias

        self.U1, self.S1, self.Vt1 = torch.linalg.svd(
            self.mmp.linear_1.weight.float(), full_matrices=False
        )
        self.proj_bias1 = self.mmp.linear_1.bias

        self.model_name = "internvl3_5"
        self.component_name = "linear_2"
        self.vocab_size = self.model.model.language_model.get_input_embeddings().num_embeddings
        self.n_dirs = self.S.shape[0]

        self.downsample_ratio = self.model.config.downsample_ratio
        vis_cfg = self.model.config.vision_config
        self.num_image_token = int(
            (vis_cfg.image_size[0] // vis_cfg.patch_size[0]) ** 2 * (self.downsample_ratio**2)
        )
        self.n_patches = self.num_image_token

        self.cache_images = cache_images
        self._image_cache = {}
        self._vision_cache = {}

    # --- Model-specific -------------------------------------------------------

    @property
    def svd_layers(self) -> list[SVDLayer]:
        return [
            SVDLayer("linear_1", self.U1, self.S1, self.Vt1, self.proj_bias1, self.S1.shape[0]),
            SVDLayer("linear_2", self.U, self.S, self.Vt, self.proj_bias, self.n_dirs),
        ]

    def run_connector_layer_masked(
        self,
        vision_out: torch.Tensor,
        layer_name: str,
        W_masked: torch.Tensor,
    ) -> torch.Tensor:
        if layer_name == "linear_1":
            hidden = self.mmp.layer_norm(vision_out).to(W_masked.dtype)
            bias1 = self.proj_bias1.to(W_masked.dtype) if self.proj_bias1 is not None else None
            hidden = F.linear(hidden, W_masked, bias1)
            hidden = self.mmp.act(hidden)
            return self.mmp.linear_2(hidden.to(self.compute_dtype))
        hidden = self.pre_svd_forward(vision_out).to(W_masked.dtype)
        bias = self.proj_bias.to(W_masked.dtype) if self.proj_bias is not None else None
        return F.linear(hidden, W_masked, bias)

    def preprocess(self, batch: list[dict], image_base_path: Path) -> dict:
        """
        Tokenize a batch of image-question pairs via the HF processor.

        :param batch: Data dicts with ``"image"`` and ``"question"`` keys.
        :type batch: list[dict]
        :param image_base_path: Root directory for image files.
        :type image_base_path: Path
        :returns: Batched, padded inputs on CUDA.
        :rtype: dict
        """
        conversations = []
        for datum in batch:
            img_path = str(image_base_path / datum["image"])
            conversations.append(
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": self.prompt},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": self._load_image(img_path)},
                            {"type": "text", "text": datum["question"]},
                        ],
                    },
                ]
            )
        return self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            padding=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

    def _get_selected_mask(self, inputs: dict) -> torch.Tensor:
        """
        Boolean mask of ``IMG_CONTEXT_TOKEN`` positions in the flattened input.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: Boolean tensor of shape ``(B * N,)``.
        :rtype: torch.Tensor
        """
        B, N = inputs["input_ids"].shape
        input_ids_flat = inputs["input_ids"].reshape(B * N)
        return input_ids_flat == self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def extract_vision(self, inputs: dict) -> torch.Tensor:
        """
        Run InternViT vision tower + pixel shuffle.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: Vision features of shape ``(n_tiles, n_patches, C_vision)``.
        :rtype: torch.Tensor
        """
        pixel_values = inputs["pixel_values"]
        vision_out = self.model.model.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state[:, 1:, :]
        h = w = int(vision_out.shape[1] ** 0.5)
        vision_out = vision_out.reshape(vision_out.shape[0], h, w, -1)
        vision_out = self.model.model.pixel_shuffle(vision_out, scale_factor=self.downsample_ratio)
        return vision_out.reshape(vision_out.shape[0], -1, vision_out.shape[-1])

    def pre_svd_forward(self, vision_out: torch.Tensor) -> torch.Tensor:
        """
        Run ``layer_norm → linear_1 → GELU`` (layers before ``linear_2``).

        :param vision_out: Output of :meth:`extract_vision`.
        :type vision_out: torch.Tensor
        :returns: Input to the SVD-decomposed ``linear_2``.
        :rtype: torch.Tensor
        """
        hidden = self.mmp.layer_norm(vision_out)
        hidden = self.mmp.linear_1(hidden)
        return self.mmp.act(hidden)

    def run_connector(self, vision_out: torch.Tensor) -> torch.Tensor:
        """
        Full MLP connector forward pass.

        :param vision_out: Output of :meth:`extract_vision`.
        :type vision_out: torch.Tensor
        :returns: Connector output in language model embedding space.
        :rtype: torch.Tensor
        """
        return self.mmp(vision_out)

    def get_text_embeds(self, inputs: dict) -> torch.Tensor:
        """
        Look up text token embeddings via the language model.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: Embeddings of shape ``(B, N, C)``.
        :rtype: torch.Tensor
        """
        return self.model.model.language_model.get_input_embeddings()(inputs["input_ids"])

    def merge_embeds(
        self,
        inputs: dict,
        text_embeds: torch.Tensor,
        conn_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace ``IMG_CONTEXT`` token positions with connector output.

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :param text_embeds: Text embeddings from :meth:`get_text_embeds`.
        :type text_embeds: torch.Tensor
        :param conn_out: Connector output to insert at image positions.
        :type conn_out: torch.Tensor
        :returns: Merged embeddings of shape ``(B, N, C)``.
        :rtype: torch.Tensor
        """
        B, N, C = text_embeds.shape
        selected = self._get_selected_mask(inputs)
        merged = text_embeds.reshape(B * N, C).clone()
        merged[selected] = conn_out.to(dtype=merged.dtype, device=merged.device).reshape(-1, C)
        return merged.reshape(B, N, C)

    def generate(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> list[str]:
        """
        Generate and decode text (single-image, greedy).

        :param embeds: Merged input embeddings.
        :type embeds: torch.Tensor
        :param attention_mask: Attention mask.
        :type attention_mask: torch.Tensor
        :param max_new_tokens: Maximum tokens to generate.
        :type max_new_tokens: int
        :returns: Single-element list with decoded prediction.
        :rtype: list[str]
        """
        out = self.model.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
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
            do_sample=False,
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
        Last-position logits from the language model.

        Handles right-padded batches by indexing the true last token
        per sequence rather than the padded ``[:, -1]`` position.

        :param embeds: Merged input embeddings.
        :type embeds: torch.Tensor
        :param attention_mask: Attention mask.
        :type attention_mask: torch.Tensor
        :returns: Logits of shape ``(B, vocab_size)``.
        :rtype: torch.Tensor
        """
        text_out = self.model.model.language_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_pos = attention_mask.sum(dim=1) - 1
        B = embeds.shape[0]
        return torch.stack([self.model.lm_head(text_out[0][b, last_pos[b], :]) for b in range(B)])

    def compute_coefficients_per_layer(self, inputs: dict) -> dict[str, torch.Tensor]:
        vision_out = self.extract_vision(inputs)
        ln_out = self.mmp.layer_norm(vision_out).float()
        gelu_out = self.mmp.act(self.mmp.linear_1(ln_out.to(self.compute_dtype))).float()
        return {
            "linear_1": self.S1.float() * (ln_out @ self.Vt1.T.float()),
            "linear_2": self.S.float() * (gelu_out @ self.Vt.T.float()),
        }

    def forward_connector_from(self, layer_name: str, layer_output: torch.Tensor) -> torch.Tensor:
        if layer_name == "linear_1":
            hidden = self.mmp.act(layer_output)
            return self.mmp.linear_2(hidden.to(self.compute_dtype))
        return layer_output

    def compute_probe_projections(self, inputs: dict) -> dict[str, torch.Tensor]:
        """
        Cosine similarity of each MLP layer's input with the layer's
        right singular vectors.

        Returns two projections:

        - ``"linear_1"``: LN output projected onto Vt of ``linear_1``
        - ``"linear_2"``: GELU output projected onto Vt of ``linear_2``

        :param inputs: Tokenized inputs from :meth:`preprocess`.
        :type inputs: dict
        :returns: ``{"linear_1": (B, n_patches, n_dirs1),
            "linear_2": (B, n_patches, n_dirs2)}``.
        :rtype: dict[str, torch.Tensor]
        """
        vision_out = self.extract_vision(inputs)
        ln_out = self.mmp.layer_norm(vision_out)
        gelu_out = self.mmp.act(self.mmp.linear_1(ln_out))

        ln_f = ln_out.float()
        gelu_f = gelu_out.float()

        return {
            "linear_1": (ln_f / (ln_f.norm(dim=-1, keepdim=True) + 1e-8)) @ self.Vt1.T.float(),
            "linear_2": (gelu_f / (gelu_f.norm(dim=-1, keepdim=True) + 1e-8)) @ self.Vt.T.float(),
        }
