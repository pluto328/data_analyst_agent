"""改错记录：失败代码经回溯修正成功后落盘，启动加载并注入 few-shot 提示。"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import (
    CORRECTION_ENABLED,
    CORRECTION_MAX_RECORDS,
    CORRECTION_TOP_K,
    ensure_temp_dir,
)
from utils.logger import get_logger

log = get_logger()

CORRECTION_SUBDIR: str = "correction_records"
CORRECTION_FILENAME: str = "corrections.jsonl"


@dataclass
class CorrectionRecord:
    """单次「错误代码 → 修正成功」样本。"""

    user_request: str
    wrong_code: str
    correct_code: str
    error_type: str
    error: str
    schema_fingerprint: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    dataset_keys: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CorrectionRecord:
        return cls(
            id=str(payload.get("id") or uuid.uuid4().hex),
            timestamp=str(payload.get("timestamp") or ""),
            user_request=str(payload.get("user_request") or ""),
            wrong_code=str(payload.get("wrong_code") or ""),
            correct_code=str(payload.get("correct_code") or ""),
            error_type=str(payload.get("error_type") or ""),
            error=str(payload.get("error") or ""),
            schema_fingerprint=str(payload.get("schema_fingerprint") or ""),
            dataset_keys=list(payload.get("dataset_keys") or []),
        )


_CACHED_RECORDS: list[CorrectionRecord] = []
_CACHE_LOADED: bool = False


def _correction_file_path() -> Path:
    try:
        ensure_temp_dir()
        subdir = Path(ensure_temp_dir()) / CORRECTION_SUBDIR
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / CORRECTION_FILENAME
    except OSError as exc:
        raise OSError(f"Cannot prepare correction store directory: {exc}") from exc


def _column_names(data_preview: dict[str, Any]) -> list[str]:
    names: list[str] = []
    if data_preview.get("all_datasets"):
        for ds in data_preview.get("all_datasets") or []:
            cols = ds.get("columns") or []
            for item in cols:
                if isinstance(item, dict):
                    names.append(str(item.get("name", "")))
                else:
                    names.append(str(item))
    else:
        cols = data_preview.get("columns") or []
        for item in cols:
            if isinstance(item, dict):
                names.append(str(item.get("name", "")))
            else:
                names.append(str(item))
    return sorted({name for name in names if name})


def schema_fingerprint(data_preview: dict[str, Any]) -> str:
    """根据列名集合生成 schema 指纹，用于相似样本匹配。"""
    names = _column_names(data_preview)
    keys = data_preview.get("dataset_keys") or []
    payload = json.dumps(
        {"columns": names, "dataset_keys": keys},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _record_dedup_key(record: CorrectionRecord) -> str:
    raw = "|".join(
        [
            record.user_request.strip(),
            record.wrong_code.strip(),
            record.correct_code.strip(),
            record.schema_fingerprint,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_from_disk() -> list[CorrectionRecord]:
    path = _correction_file_path()
    if not path.is_file():
        return []
    records: list[CorrectionRecord] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                records.append(CorrectionRecord.from_dict(payload))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                log.warning("Skip invalid correction record line: {}", exc)
    except OSError as exc:
        log.exception("Failed to read correction records: {}", path)
        return records
    return records[-CORRECTION_MAX_RECORDS:]


def _write_all(records: list[CorrectionRecord]) -> None:
    path = _correction_file_path()
    trimmed = records[-CORRECTION_MAX_RECORDS:]
    lines = [
        json.dumps(asdict(record), ensure_ascii=False) for record in trimmed
    ]
    try:
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    except OSError as exc:
        log.exception("Failed to write correction records: {}", path)
        raise OSError(f"Failed to write correction records: {path}") from exc


def ensure_correction_records_loaded() -> list[CorrectionRecord]:
    """启动或首次使用时加载改错记录到内存缓存。"""
    global _CACHED_RECORDS, _CACHE_LOADED
    if _CACHE_LOADED:
        return _CACHED_RECORDS
    try:
        _CACHED_RECORDS = _load_from_disk()
        _CACHE_LOADED = True
        log.info("Loaded {} correction record(s)", len(_CACHED_RECORDS))
    except Exception as exc:
        log.exception("Failed to load correction records: {}", exc)
        _CACHED_RECORDS = []
        _CACHE_LOADED = True
    return _CACHED_RECORDS


def correction_record_count() -> int:
    """返回已加载的改错记录条数。"""
    return len(ensure_correction_records_loaded())


def _score_record(
    record: CorrectionRecord,
    *,
    user_request: str,
    fingerprint: str,
    error_type: str | None,
) -> float:
    score = 0.0
    if record.schema_fingerprint == fingerprint:
        score += 10.0
    if error_type and record.error_type == error_type:
        score += 5.0
    req = user_request.strip().lower()
    past = record.user_request.strip().lower()
    if req and past:
        req_set = set(req)
        past_set = set(past)
        overlap = len(req_set & past_set) / max(len(req_set | past_set), 1)
        score += overlap * 4.0
    return score


def find_similar_corrections(
    user_request: str,
    data_preview: dict[str, Any],
    *,
    error_type: str | None = None,
    top_k: int | None = None,
    page: int = 0,
) -> list[CorrectionRecord]:
    """按相似度排序后分页取改错样本。

    ``page=0`` 取前 top_k 条；``page=1`` 取接下来 top_k 条；无更多则返回空列表。
    """
    if not CORRECTION_ENABLED:
        return []
    records = ensure_correction_records_loaded()
    if not records:
        return []
    limit = top_k if top_k is not None else CORRECTION_TOP_K
    page_index = max(0, page)
    fingerprint = schema_fingerprint(data_preview)
    ranked = sorted(
        records,
        key=lambda item: _score_record(
            item,
            user_request=user_request,
            fingerprint=fingerprint,
            error_type=error_type,
        ),
        reverse=True,
    )
    positive = [
        item
        for item in ranked
        if _score_record(
            item,
            user_request=user_request,
            fingerprint=fingerprint,
            error_type=error_type,
        )
        > 0
    ]
    start = page_index * limit
    if start >= len(positive):
        return []
    return positive[start : start + limit]


def format_corrections_for_prompt(
    records: list[CorrectionRecord],
) -> str:
    """将改错样本格式化为 few-shot 提示段落。"""
    if not records:
        return ""
    blocks: list[str] = [
        "## 历史改错经验（同类问题可参考修复思路，列名仍须以当前预览为准）"
    ]
    for index, record in enumerate(records, start=1):
        blocks.append(
            f"### 案例 {index}\n"
            f"- 需求：{record.user_request}\n"
            f"- 错误类型：{record.error_type}\n"
            f"- 错误信息：{record.error}\n"
            f"- 错误代码：\n```python\n{record.wrong_code.strip()}\n```\n"
            f"- 正确代码：\n```python\n{record.correct_code.strip()}\n```"
        )
    return "\n\n".join(blocks) + "\n\n"


def format_similar_corrections_for_prompt(
    user_request: str,
    data_preview: dict[str, Any],
    *,
    error_type: str | None = None,
    retry_count: int = 0,
) -> str:
    """检索并格式化相似改错记录；``retry_count`` 决定加载第几页 top_k 参考。"""
    try:
        similar = find_similar_corrections(
            user_request,
            data_preview,
            error_type=error_type,
            page=retry_count,
        )
        return format_corrections_for_prompt(similar)
    except Exception as exc:
        log.exception("Failed to format correction prompt: {}", exc)
        return ""


def save_correction_record(
    *,
    user_request: str,
    data_preview: dict[str, Any],
    wrong_code: str,
    correct_code: str,
    error_type: str,
    error: str,
) -> CorrectionRecord | None:
    """落盘一条改错记录；重复样本跳过。"""
    if not CORRECTION_ENABLED:
        return None
    if not user_request.strip() or not wrong_code.strip() or not correct_code.strip():
        return None
    if wrong_code.strip() == correct_code.strip():
        return None

    record = CorrectionRecord(
        user_request=user_request.strip(),
        wrong_code=wrong_code.strip(),
        correct_code=correct_code.strip(),
        error_type=error_type or "Unknown",
        error=(error or "").strip(),
        schema_fingerprint=schema_fingerprint(data_preview),
        dataset_keys=list(data_preview.get("dataset_keys") or []),
    )
    dedup_key = _record_dedup_key(record)

    try:
        records = ensure_correction_records_loaded()
        existing_keys = {_record_dedup_key(item) for item in records}
        if dedup_key in existing_keys:
            log.info("Skip duplicate correction record")
            return None
        records.append(record)
        _write_all(records)
        global _CACHED_RECORDS
        _CACHED_RECORDS = records[-CORRECTION_MAX_RECORDS:]
        log.info("Saved correction record id={}", record.id)
        return record
    except Exception as exc:
        log.exception("Failed to save correction record: {}", exc)
        return None


def save_correction_from_retry_success(
    *,
    user_request: str,
    data_preview: dict[str, Any],
    retry_history: list[dict[str, Any]],
    correct_code: str,
) -> CorrectionRecord | None:
    """回溯成功后，用最后一次失败样本与最终正确代码落盘。"""
    if not retry_history:
        return None
    last_failure = retry_history[-1]
    return save_correction_record(
        user_request=user_request,
        data_preview=data_preview,
        wrong_code=str(last_failure.get("code") or ""),
        correct_code=correct_code,
        error_type=str(last_failure.get("error_type") or "Unknown"),
        error=str(last_failure.get("error") or ""),
    )


def _run_self_check() -> None:
    print("=== correction_store self-check ===")
    preview = {
        "columns": [{"name": "a", "dtype": "int64"}, {"name": "b", "dtype": "int64"}],
        "dataset_keys": ["df1"],
    }
    fp = schema_fingerprint(preview)
    print(f"schema fingerprint: {fp}")
    loaded = ensure_correction_records_loaded()
    print(f"loaded records: {len(loaded)}")
    section_p0 = format_similar_corrections_for_prompt("删除空值并求和", preview, retry_count=0)
    section_p1 = format_similar_corrections_for_prompt("删除空值并求和", preview, retry_count=1)
    print(f"prompt page0 length: {len(section_p0)}")
    print(f"prompt page1 length: {len(section_p1)}")
    print("=== done ===")


__all__ = [
    "CORRECTION_FILENAME",
    "CORRECTION_SUBDIR",
    "CorrectionRecord",
    "correction_record_count",
    "ensure_correction_records_loaded",
    "find_similar_corrections",
    "format_corrections_for_prompt",
    "format_similar_corrections_for_prompt",
    "save_correction_from_retry_success",
    "save_correction_record",
    "schema_fingerprint",
]


if __name__ == "__main__":
    _run_self_check()
