"""
LLM / VLM interfaces used by TStar.

This module is cleaned for open-source release:
- No hard-coded API keys
- No hard-coded private endpoints
- No sys.path hacks

Configuration (environment variables):
- OPENAI_API_KEY: API key for OpenAI-compatible endpoints (required)
- OPENAI_BASE_URL: base URL, e.g. "https://api.openai.com/v1" or "http://localhost:8000/v1"
- OPENAI_MODEL: model name, e.g. "gpt-4o-mini"

The rest of the codebase depends on the `TStarUniversalGrounder` class, which provides:
- `inference_query_grounding`
- `inference_query_grounding2` (with relations)
- `inference_qa`
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image

from TStar.utilites import encode_image_to_base64, load_video_frames

logger = logging.getLogger(__name__)


def _normalize_openai_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is empty.")
    # Accept both ".../v1" and raw host urls.
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration for an OpenAI-compatible chat completion endpoint."""

    base_url: str
    api_key: str
    model: str
    timeout_s: float = 120.0


class OpenAICompatibleChatClient:
    """
    Minimal OpenAI-compatible client based on HTTP requests.

    Works with OpenAI and many self-hosted endpoints that implement the
    `/v1/chat/completions` API.
    """

    def __init__(self, cfg: OpenAICompatibleConfig):
        self._cfg = cfg
        self._chat_completions_url = f"{_normalize_openai_base_url(cfg.base_url)}/chat/completions"

    def chat_completions(
        self,
        messages: Sequence[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": self._cfg.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg.api_key}",
        }
        resp = requests.post(
            self._chat_completions_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self._cfg.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class TStarUniversalGrounder:
    """
    Ground objects and answer multiple-choice questions using an OpenAI-compatible VLM.

    Notes:
    - This class proposes object vocabularies (key/cue objects) and can answer QA based on frames.
    - It does not run any object detector itself.
    """

    def __init__(
        self,
        backend: str = "openai_compatible",
        model_path: Optional[str] = None,  # kept for backwards compatibility, unused
        gpt4_model_name: str = "gpt-4o-mini",
        gpt4_api_key: Optional[str] = None,
        num_frames: Optional[int] = 8,
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
    ):
        backend_norm = (backend or "").lower().strip()
        # Backwards-compatible aliases used in earlier experiment scripts.
        if backend_norm in {"gpt4", "openai", "openai_compatible", "qwenvl", "llava", "internvl", "internvl1"}:
            backend_norm = "openai_compatible"
        if backend_norm != "openai_compatible":
            raise ValueError(f"Unsupported backend: {backend}. Use 'openai_compatible'.")

        self.num_frames = int(num_frames or 8)

        base_url = openai_base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = openai_api_key or gpt4_api_key or os.getenv("OPENAI_API_KEY", "")
        model = openai_model or os.getenv("OPENAI_MODEL", gpt4_model_name)

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Provide `openai_api_key`/`gpt4_api_key` or set env OPENAI_API_KEY."
            )

        self._client = OpenAICompatibleChatClient(
            OpenAICompatibleConfig(base_url=base_url, api_key=api_key, model=model)
        )

    def inference_query_grounding(
        self,
        video_path: str,
        question: str,
        upload_video: bool = True,
        options: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Tuple[List[str], List[str]]:
        """
        Identify target objects (answer-critical) and cue objects (contextual) from a question.

        Returns:
            (target_objects, cue_objects)
        """
        frames = load_video_frames(video_path=video_path, num_frames=self.num_frames) if upload_video else []
        prompt = self._build_grounding_prompt(question=question, options=options, with_relations=False)
        response = self._chat_with_optional_frames(prompt=prompt, frames=frames, temperature=temperature, max_tokens=max_tokens)
        target_objects, cue_objects, _rels = self._parse_grounding_response(response, expect_relations=False)
        return target_objects, cue_objects

    def inference_query_grounding2(
        self,
        video_path: str,
        question: str,
        upload_video: bool = True,
        options: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
        """
        Identify target objects, cue objects, and optional relations.

        Returns:
            (target_objects, cue_objects, relations)
        """
        frames = load_video_frames(video_path=video_path, num_frames=self.num_frames) if upload_video else []
        prompt = self._build_grounding_prompt(question=question, options=options, with_relations=True)
        response = self._chat_with_optional_frames(prompt=prompt, frames=frames, temperature=temperature, max_tokens=max_tokens)
        target_objects, cue_objects, rels = self._parse_grounding_response(response, expect_relations=True)
        return target_objects, cue_objects, rels

    def inference_qa(
        self,
        frames: List[Image.Image],
        question: str,
        options: str,
        temperature: float = 0.1,
        max_tokens: int = 30,
        **_: Any,
    ) -> str:
        """
        Answer a multiple-choice question based on retrieved frames.

        Returns:
            Model output stripped. When the model follows instructions, this is a single option letter.
        """
        prompt = (
            "Select the best answer to the multiple-choice question based on the video frames.\n"
            f"Question: {question}\n"
            f"Options: {options}\n"
            "Answer with ONLY the option letter (e.g., A, B, C, D, E).\n"
        )
        response = self._chat_with_optional_frames(prompt=prompt, frames=frames, temperature=temperature, max_tokens=max_tokens)
        return response.strip()

    def _chat_with_optional_frames(
        self,
        prompt: str,
        frames: List[Image.Image],
        temperature: float,
        max_tokens: int,
    ) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in frames:
            b64 = encode_image_to_base64(frame)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        messages = [{"role": "user", "content": content}]
        return self._client.chat_completions(messages=messages, temperature=temperature, max_tokens=max_tokens)

    def _build_grounding_prompt(self, question: str, options: Optional[str], with_relations: bool) -> str:
        base = (
            "You will be given video frames and a multiple-choice question.\n"
            "Your job is to propose object vocabularies for downstream object detection.\n\n"
            f"Question: {question}\n"
        )
        if options:
            base += f"Options: {options}\n"

        if not with_relations:
            base += (
                "\nOutput EXACTLY two lines:\n"
                "Key Objects: obj1, obj2, obj3\n"
                "Cue Objects: cue1, cue2, cue3\n"
                "Rules:\n"
                "- Use short YOLO-detectable noun phrases.\n"
                "- No explanations, no markdown.\n"
            )
            return base

        base += (
            "\nOutput EXACTLY three lines:\n"
            "Key Objects: obj1, obj2, obj3\n"
            "Cue Objects: cue1, cue2, cue3\n"
            "Rel: (obj1; spatial; obj2), (obj3; attribute; obj4)\n"
            "Rules:\n"
            "- relation_type must be one of: spatial, attribute, time, causal\n"
            "- Both objects in each relation must appear in Key Objects or Cue Objects.\n"
            "- No explanations, no markdown.\n"
        )
        return base

    def _parse_grounding_response(
        self, response: str, expect_relations: bool
    ) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
        """
        Parse model output into objects and relations.

        The parser is tolerant to extra whitespace or additional lines.
        """
        text = (response or "").strip()
        key_line = self._extract_prefixed_line(text, "Key Objects")
        cue_line = self._extract_prefixed_line(text, "Cue Objects")
        rel_line = self._extract_prefixed_line(text, "Rel") if expect_relations else None

        target_objects = self._parse_object_list(key_line)
        cue_objects = self._parse_object_list(cue_line)
        relations = self._parse_relations(rel_line) if rel_line else []

        if not target_objects and not cue_objects:
            logger.warning("Grounding response parsed empty object lists. Raw response:\n%s", text)
        return target_objects, cue_objects, relations

    @staticmethod
    def _extract_prefixed_line(text: str, prefix: str) -> str:
        for line in text.splitlines():
            if line.strip().lower().startswith(prefix.lower() + ":"):
                return line.strip()
        m = re.search(rf"{re.escape(prefix)}\s*:\s*(.+)", text, flags=re.IGNORECASE)
        return f"{prefix}: {m.group(1).strip()}" if m else f"{prefix}:"

    @staticmethod
    def _parse_object_list(line: str) -> List[str]:
        _, _, rest = line.partition(":")
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        seen = set()
        out: List[str] = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    @staticmethod
    def _parse_relations(line: str) -> List[Tuple[str, str, str]]:
        """
        Parse "Rel: (a; spatial; b), (c; attribute; d)" into triplets.

        Returns:
            List of (obj1, obj2, relation_type).
        """
        if not line:
            return []
        _, _, rest = line.partition(":")
        triplets: List[Tuple[str, str, str]] = []
        for group in re.findall(r"\(([^)]+)\)", rest):
            parts = [p.strip() for p in group.split(";")]
            if len(parts) != 3:
                continue
            obj1, rel_type, obj2 = parts[0], parts[1].lower(), parts[2]
            triplets.append((obj1, obj2, rel_type))
        return triplets


if __name__ == "__main__":
    # Minimal self-check (requires env OPENAI_API_KEY).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    g = TStarUniversalGrounder(backend="openai_compatible")
    objs = g.inference_query_grounding(
        video_path="example.mp4",
        question="What is the person doing?",
        options="A) sleeping\nB) cooking\nC) answering reporters\nD) swimming",
        upload_video=False,
    )
    logger.info("Parsed grounding (text-only): %s", objs)

