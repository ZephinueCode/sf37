#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import tarfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCES = {
    "code": "/mnt/share-bos/data/core-data-share/text_sft/coding/OpenCodeReasoning-2_wds",
    "if_en": "/mnt/share-bos/data/core-data-share/text_sft/mimo7bsftr10515v3amthinkingif",
    "zh": "/mnt/share-bos/data/core-data-share/text_sft/chinese-distill-glm",
}


def discover_shards(path: str | Path, limit: int) -> list[Path]:
    p = Path(path)
    if p.is_file() and p.suffix == ".tar":
        return [p]
    shards = sorted(p.rglob("*.tar"))
    if limit > 0:
        shards = shards[:limit]
    if not shards:
        raise RuntimeError(f"no .tar shards under {p}")
    return shards


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for segment in content:
        if isinstance(segment, str):
            parts.append(segment)
        elif isinstance(segment, dict) and segment.get("type", "text") == "text":
            value = segment.get("value", segment.get("text", segment.get("content", "")))
            if value:
                parts.append(str(value))
    return "".join(parts)


def normalize_role(role: Any) -> str | None:
    r = str(role or "").lower()
    if r in {"human", "user"}:
        return "user"
    if r in {"model", "assistant", "gpt"}:
        return "assistant"
    if r == "system":
        return "system"
    return None


def parse_payload(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    raw_turns = payload.get("conversation", payload.get("conversations"))
    turns: list[dict[str, str]] = []
    if isinstance(raw_turns, list):
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            role = normalize_role(turn.get("role", turn.get("from")))
            if role is None:
                continue
            text = extract_text_content(turn.get("content", turn.get("value", ""))).strip()
            if text:
                turns.append({"role": role, "content": text})
    elif payload.get("text"):
        text = str(payload["text"]).strip()
        if text:
            turns.append({"role": "assistant", "content": text})
    if not turns:
        return None
    return {
        "id": str(payload.get("unique_id", payload.get("id", key))),
        "conversations": turns,
    }


def iter_tar_samples(shards: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for shard in shards:
        with tarfile.open(shard, "r") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                try:
                    payload = json.loads(f.read().decode("utf-8"))
                except Exception:
                    continue
                parsed = parse_payload(payload, member.name)
                if parsed is not None:
                    parsed["shard"] = str(shard)
                    yield parsed


def render_chat(turns: list[dict[str, str]], max_chars: int) -> str:
    out: list[str] = []
    used = 0
    for turn in turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        block = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if max_chars > 0 and used + len(block) > max_chars:
            remain = max_chars - used
            if remain <= len(f"<|im_start|>{role}\n<|im_end|>\n") + 32:
                break
            content_cap = remain - len(f"<|im_start|>{role}\n<|im_end|>\n")
            block = f"<|im_start|>{role}\n{content[:content_cap]}<|im_end|>\n"
        out.append(block)
        used += len(block)
        if max_chars > 0 and used >= max_chars:
            break
    return "".join(out).strip()


def estimate_tokens(text: str, tokenizer: Any | None) -> int:
    if tokenizer is None:
        return max(1, len(text.encode("utf-8")) // 4)
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def load_tokenizer(path: str | None) -> Any | None:
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("--tokenizer requires transformers") from exc
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def take_source(name: str,
                path: str,
                quota: int,
                shard_limit: int,
                max_chars: int,
                rng: random.Random) -> list[dict[str, Any]]:
    shards = discover_shards(path, shard_limit)
    samples = list(iter_tar_samples(shards))
    rng.shuffle(samples)
    kept: list[dict[str, Any]] = []
    for sample in samples:
        text = render_chat(sample["conversations"], max_chars)
        if not text:
            continue
        kept.append({
            "id": sample["id"],
            "source": name,
            "shard": sample["shard"],
            "text": text,
        })
        if len(kept) >= quota:
            break
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="gguf-tools/imatrix/dataset")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-prompts", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=800_000)
    ap.add_argument("--max-chars", type=int, default=16_384)
    ap.add_argument("--shards-per-source", type=int, default=1)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--code-path", default=DEFAULT_SOURCES["code"])
    ap.add_argument("--if-en-path", default=DEFAULT_SOURCES["if_en"])
    ap.add_argument("--zh-path", default=DEFAULT_SOURCES["zh"])
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tokenizer = load_tokenizer(args.tokenizer)
    total = max(1, args.max_prompts)
    quotas = {
        "code": max(1, round(total * 0.35)),
        "if_en": max(1, round(total * 0.35)),
        "zh": max(1, total - max(1, round(total * 0.35)) * 2),
    }

    records: list[dict[str, Any]] = []
    records.extend(take_source("code", args.code_path, quotas["code"],
                               args.shards_per_source, args.max_chars, rng))
    records.extend(take_source("if_en", args.if_en_path, quotas["if_en"],
                               args.shards_per_source, args.max_chars, rng))
    records.extend(take_source("zh", args.zh_path, quotas["zh"],
                               args.shards_per_source, args.max_chars, rng))
    rng.shuffle(records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "prompts.jsonl"
    rendered_path = out_dir / "rendered_prompts.txt"
    manifest_path = out_dir / "manifest.json"

    final: list[dict[str, Any]] = []
    token_total = 0
    for rec in records:
        ntok = estimate_tokens(rec["text"], tokenizer)
        if args.max_tokens > 0 and token_total + ntok > args.max_tokens and final:
            break
        rec["tokens_est"] = ntok
        final.append(rec)
        token_total += ntok
        if len(final) >= args.max_prompts:
            break

    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in final:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with rendered_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(final):
            f.write(f"===== SF37_IMATRIX_PROMPT {i:06d} {rec['source']} =====\n")
            f.write(rec["text"].rstrip() + "\n\n")

    counts: dict[str, int] = {}
    for rec in final:
        counts[rec["source"]] = counts.get(rec["source"], 0) + 1
    manifest = {
        "records": len(final),
        "tokens_est": token_total,
        "counts": counts,
        "sources": {
            "code": args.code_path,
            "if_en": args.if_en_path,
            "zh": args.zh_path,
        },
        "shards_per_source": args.shards_per_source,
        "max_chars": args.max_chars,
        "tokenizer": args.tokenizer,
        "rendered_prompts": str(rendered_path),
        "prompts_jsonl": str(jsonl_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
