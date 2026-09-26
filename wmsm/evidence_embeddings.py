import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


DEFAULT_EVIDENCE_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate_text(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    max_chars = int(max_chars)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars].rstrip()
    return text[: max_chars - 15].rstrip() + " ...[truncated]"


def compact_action_sequence(before_segments: Any, *, max_actions: int = 12) -> str:
    """Return a compact action-only summary from EvidenceCard before_segments."""
    max_actions = max(1, int(max_actions))
    if not isinstance(before_segments, list):
        return ""

    rendered_segments: List[str] = []
    for segment_idx, raw_segment in enumerate(before_segments):
        if isinstance(raw_segment, dict):
            raw_segment = raw_segment.get("steps") or []
        if not isinstance(raw_segment, list):
            continue
        ordered_steps = sorted(
            (step for step in raw_segment if isinstance(step, dict)),
            key=lambda step: int(step.get("step", 0) or 0),
        )
        actions = [_clean_text(step.get("action")) for step in ordered_steps]
        actions = [action for action in actions if action]
        if not actions:
            continue

        compressed: List[str] = []
        truncated = len(actions) > max_actions
        for action in actions[:max_actions]:
            if compressed and compressed[-1].startswith(action + " x"):
                prefix, _, raw_count = compressed[-1].rpartition(" x")
                try:
                    compressed[-1] = f"{prefix} x{int(raw_count) + 1}"
                except ValueError:
                    compressed.append(action)
            elif compressed and compressed[-1] == action:
                compressed[-1] = f"{action} x2"
            else:
                compressed.append(action)
        if truncated:
            compressed.append("[truncated]")

        text = " -> ".join(compressed)
        if len(before_segments) > 1:
            text = f"segment {segment_idx + 1}: {text}"
        rendered_segments.append(text)

    return "\n".join(rendered_segments)


def build_evidence_embedding_text(
    card: Dict[str, Any],
    *,
    max_reason_chars: int = 300,
    max_actions: int = 12,
) -> str:
    """Build the minimal semantic text used for EvidenceCard clustering."""
    proposed_edit = card.get("proposed_edit") if isinstance(card.get("proposed_edit"), dict) else {}
    edit_op = _clean_text(proposed_edit.get("op"))
    edit_target = _clean_text(proposed_edit.get("target"))
    edit_content = _clean_text(proposed_edit.get("content"))
    pattern = _truncate_text(card.get("pattern"), max_reason_chars)

    lines = [
        f"pattern: {pattern}",
        f"edit_op: {edit_op}",
        f"edit_target: {edit_target}",
        f"edit_content: {edit_content}",
    ]
    return "\n".join(line for line in lines if line.split(':', 1)[-1].strip()).strip()


def evidence_embedding_cache_key(*, model_name: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_name}:{digest}"


def _read_embedding_cache(path: Path) -> Dict[str, List[float]]:
    cache: Dict[str, List[float]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("cache_key") or "")
            embedding = row.get("embedding")
            if key and isinstance(embedding, list):
                try:
                    cache[key] = [float(x) for x in embedding]
                except (TypeError, ValueError):
                    continue
    return cache


def _append_embedding_cache(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_vector(vector: Iterable[Any]) -> List[float]:
    values = [float(x) for x in vector]
    norm = sum(x * x for x in values) ** 0.5
    if norm <= 0:
        return values
    return [x / norm for x in values]


def encode_evidence_cards(
    cards: List[Dict[str, Any]],
    *,
    model_name: str = DEFAULT_EVIDENCE_EMBEDDING_MODEL,
    cache_path: Optional[Path] = None,
    device: str = "auto",
    batch_size: int = 32,
    max_reason_chars: int = 300,
    max_actions: int = 12,
    encoder: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Encode EvidenceCards with cache support.

    The optional encoder is used by tests and must return already comparable
    vectors. Production calls use sentence-transformers and normalize vectors.
    """
    texts = [
        build_evidence_embedding_text(
            card,
            max_reason_chars=max_reason_chars,
            max_actions=max_actions,
        )
        for card in cards or []
    ]
    keys = [evidence_embedding_cache_key(model_name=model_name, text=text) for text in texts]
    cache: Dict[str, List[float]] = _read_embedding_cache(cache_path) if cache_path else {}

    embeddings: List[Optional[List[float]]] = [cache.get(key) for key in keys]
    missing_indices = [idx for idx, embedding in enumerate(embeddings) if embedding is None]
    cache_hits = len(cards or []) - len(missing_indices)

    if missing_indices:
        missing_texts = [texts[idx] for idx in missing_indices]
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - exercised in integration environments
                raise RuntimeError(
                    "Semantic evidence clustering requires sentence_transformers. "
                    "Install it in the active environment or use card/round grouping."
                ) from exc

            try:
                kwargs: Dict[str, Any] = {}
                if str(device or "auto").lower() != "auto":
                    kwargs["device"] = device
                model = SentenceTransformer(model_name, **kwargs)
                encoded = model.encode(
                    missing_texts,
                    batch_size=max(1, int(batch_size)),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as exc:  # pragma: no cover - depends on external model state
                raise RuntimeError(
                    f"Failed to load or run evidence embedding model {model_name!r}: {exc}"
                ) from exc
            encoded_vectors = [list(map(float, vector)) for vector in encoded]
        else:
            encoded_vectors = [list(map(float, vector)) for vector in encoder(missing_texts)]
            if len(encoded_vectors) != len(missing_texts):
                raise RuntimeError("Evidence embedding encoder returned the wrong number of vectors.")

        cache_rows = []
        for idx, vector in zip(missing_indices, encoded_vectors):
            normalized = _normalize_vector(vector)
            embeddings[idx] = normalized
            cache_rows.append(
                {
                    "cache_key": keys[idx],
                    "model": model_name,
                    "text_hash": hashlib.sha256(texts[idx].encode("utf-8")).hexdigest(),
                    "embedding": normalized,
                }
            )
        if cache_path:
            _append_embedding_cache(cache_path, cache_rows)

    final_embeddings = [_normalize_vector(vector or []) for vector in embeddings]
    return final_embeddings, {
        "embedding_model": model_name,
        "embedding_cache_hits": cache_hits,
        "embedding_cache_misses": len(missing_indices),
        "embedding_count": len(final_embeddings),
    }
