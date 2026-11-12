from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

import torch
from torch import nn

from ced.config.models import EnsembleTextEncodingSettings, TextModelSettings


class TextComponent(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class SapBertTextEncoder(nn.Module):
    """SapBERT's original last-layer CLS representation with length-grouped dynamic padding."""

    def __init__(self, spec: TextModelSettings, settings: EnsembleTextEncodingSettings, device: str) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as exception:
            raise ModuleNotFoundError("install the optional text extra: uv sync --extra text") from exception
        self.device = torch.device(device)
        self.settings = settings
        self.dimension = spec.dimension
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model, revision=spec.revision, local_files_only=settings.local_files_only
        )
        self.model = AutoModel.from_pretrained(
            spec.model,
            revision=spec.revision,
            local_files_only=settings.local_files_only,
            dtype=torch.float32,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        if int(self.model.config.hidden_size) != self.dimension:
            raise ValueError(f"SapBERT dimension {self.model.config.hidden_size} does not match {self.dimension}")

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.dimension), device=self.device)
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        output = torch.empty((len(texts), self.dimension), dtype=torch.float32, device=self.device)
        for start in range(0, len(order), self.settings.batch_size):
            indices = order[start : start + self.settings.batch_size]
            tokens = self.tokenizer(
                [texts[index] for index in indices],
                padding=True,
                truncation=True,
                max_length=self.settings.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                vectors = self.model(**tokens).last_hidden_state[:, 0, :]
            vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
            output[torch.tensor(indices, device=self.device)] = vectors
        return output


class SentenceTransformerComponent(nn.Module):
    def __init__(
        self,
        spec: TextModelSettings,
        settings: EnsembleTextEncodingSettings,
        device: str,
        *,
        gemma: bool = False,
        qwen: bool = False,
    ) -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as exception:
            raise ModuleNotFoundError("install the optional text extra: uv sync --extra text") from exception
        kwargs: dict[str, Any] = {}
        if gemma:
            kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
        if qwen:
            kwargs["tokenizer_kwargs"] = {"padding_side": "left"}
        self.model = SentenceTransformer(
            spec.model,
            revision=spec.revision,
            device=device,
            local_files_only=settings.local_files_only,
            model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
            **kwargs,
        )
        self.model.max_seq_length = settings.max_length
        self.model.eval()
        self.settings = settings
        self.dimension = spec.dimension
        self.prompt_name = "document" if gemma else None
        actual = self.model.get_sentence_embedding_dimension()
        if actual != self.dimension:
            raise ValueError(f"{spec.model} dimension {actual} does not match {self.dimension}")

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.settings.batch_size,
            prompt_name=self.prompt_name,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return torch.nn.functional.normalize(cast(torch.Tensor, vectors).float(), dim=-1)


class ConcatenatedTextEncoder(nn.Module):
    """Normalize every model block independently before preserving model order in the concatenation."""

    def __init__(self, components: Sequence[TextComponent]) -> None:
        super().__init__()
        self.components = nn.ModuleList(components)  # pyright: ignore[reportArgumentType]
        self._encoders = components
        self.dimension = sum(component.dimension for component in components)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        blocks = [
            torch.nn.functional.normalize(component.encode(texts).float(), dim=-1) for component in self._encoders
        ]
        return torch.cat(blocks, dim=-1)


def create_ensemble_encoder(settings: EnsembleTextEncodingSettings, device: str = "cuda") -> ConcatenatedTextEncoder:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the three-model ensemble requires a CUDA device")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    components = [
        SapBertTextEncoder(settings.sapbert, settings, device),
        SentenceTransformerComponent(settings.gemma, settings, device, gemma=True),
        SentenceTransformerComponent(settings.qwen, settings, device, qwen=True),
    ]
    encoder = ConcatenatedTextEncoder(components)
    if encoder.dimension != settings.dimension:
        raise ValueError(f"ensemble dimension {encoder.dimension} does not match {settings.dimension}")
    return encoder
