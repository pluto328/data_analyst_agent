"""根据用户需求与表格预览，生成可在沙箱中执行的 Pandas 代码。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL
from agent.correction_store import (
    ensure_correction_records_loaded,
    format_similar_corrections_for_prompt,
)
from sandbox.code_sandbox import compile_user_code
from sandbox.safe_globals import SecurityError, validate_code_security
from utils.logger import get_logger

log = get_logger()

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

SYSTEM_PROMPT = """你是数据分析代码生成助手，专门为 RestrictedPython 沙箱编写 Pandas 数据处理代码。

## 硬性约束（必须遵守）
1. 只输出可执行的 Python 代码，不要解释文字。
2. 禁止 import / from 语句（沙箱已预置 pd、np）。
3. 禁止 os、subprocess、open、eval、exec、网络与文件读写。
4. 数据表变量：
   - 单表时：使用 df1（df 为 df1 的别名，二者等价）。
   - 多表时：按上传顺序命名为 df1、df2、df3…；df 仍指向 df1。
   - 多表关联使用 pd.merge(df1, df2, …)、pd.concat 等；勿覆盖 df1/df2 等已注入变量。
5. 最终结果必须赋值给 result（可以是 DataFrame、Series、dict、list 或标量）。
6. 如需保留清洗后的表，可更新 df1 或新建变量（如 df_merged）；result 须代表本次分析结论。
7. 仅使用 pandas / numpy 常见 API；列名必须来自预览 JSON，禁止臆造列名。
8. 代码应简洁、可一次执行成功；优先链式中间变量，避免过度嵌套。

## 常见用户需求 → 处理含义（语义对照）
| 用户说法 | 含义 | 推荐写法 |
|---------|------|---------|
| 删除空值/去掉缺失/去除 NaN | 去掉含缺失值的行或列 | `dropna(subset=[...])` 或 `dropna(how='all')` |
| 去重/重复行 | 删除完全重复记录 | `drop_duplicates(subset=[...], keep='first')` |
| 筛选/过滤/只要/保留 | 按条件保留子集 | `df1[df1['列'] == 值]` 或 `df1[df1['列'].isin([...])]` |
| 排序 | 按列升/降序 | `sort_values(by='列', ascending=False)` |
| 分组/按…汇总/统计/求和/平均/计数 | 分组聚合 | `groupby(..., as_index=False).agg(...)` 或 `.sum()/mean()/count()` |
| 透视/交叉表 | 行列维度汇总 | `pivot_table(index=..., columns=..., values=..., aggfunc=...)` |
| 合并/关联/连接/join | 多表按键合并 | `pd.merge(df1, df2, on='键', how='inner/left/outer')` |
| 纵向拼接/上下合并 | 同结构表堆叠 | `pd.concat([df1, df2], ignore_index=True)` |
| 新增列/计算列/衍生 | 由现有列计算 | `df1['新列'] = ...` 或 `df1.assign(新列=...)` |
| 重命名列 | 改列名 | `rename(columns={'旧':'新'})` |
| 类型转换/转数值/转日期 | 修正 dtype | `astype(...)`、`pd.to_numeric(..., errors='coerce')`、`pd.to_datetime(...)` |
| 填充缺失/补全 | 用常数或统计量填 NaN | `fillna(0)` 或 `fillna(df1['列'].mean())` |
| 替换/改成 | 值映射 | `replace({旧: 新})` |
| 取前 N / Top N / 排名 | 前几条或排序截取 | `nlargest(n, '列')` 或 `sort_values(...).head(n)` |
| 描述统计/概况 | 数值列统计 | `describe()` 或 `agg(['mean','sum','count'])` |
| 计数/各…有多少 | 分类频次 | `value_counts()` 或 `groupby(...).size()` |
| 唯一值/ distinct | 去重后的值 | `drop_duplicates()` 或 `unique()` |
| 重置索引 | 恢复默认行号 | `reset_index(drop=True)` |

## 代码书写模板（按需选用，替换为真实列名）
### 模板 A：单表清洗 + 聚合（最常用）
```python
work = df1.copy()
work = work.dropna(subset=['关键列'])
work = work[work['状态列'] == '目标值']
result = (
    work.groupby('分组列', as_index=False)['数值列']
    .sum()
    .sort_values('数值列', ascending=False)
)
```

### 模板 B：单表筛选 + 统计标量
```python
work = df1.dropna(subset=['amount'])
filtered = work[work['category'] == 'A']
result = filtered['amount'].sum()
```

### 模板 C：双表关联 + 汇总
```python
merged = pd.merge(df1, df2, on='id', how='inner')
merged = merged.dropna(subset=['amount'])
result = merged.groupby('category', as_index=False)['amount'].sum()
```

### 模板 D：多表纵向合并
```python
combined = pd.concat([df1, df2], ignore_index=True)
result = combined.drop_duplicates()
```

### 模板 E：透视分析
```python
result = df1.pivot_table(
    index='行维度',
    columns='列维度',
    values='数值',
    aggfunc='sum',
    fill_value=0,
)
```

### 模板 F：新增计算列后输出
```python
work = df1.copy()
work['total'] = work['price'] * work['qty']
result = work.sort_values('total', ascending=False)
```

## 编码规范
1. 先用 `work = df1.copy()`（或多表时 `merged = pd.merge(...)`）再变换，避免误改注入变量。
2. 写 `subset=`、`on=`、`by=` 时列名必须与预览 JSON 中 `columns` 完全一致（区分大小写）。
3. 数值计算前对非数值列用 `pd.to_numeric(..., errors='coerce')`。
4. 日期列用 `pd.to_datetime(..., errors='coerce')` 再比较或提取年月。
5. `groupby` 后若需继续当表用，加 `as_index=False` 或 `.reset_index()`。
6. 最后一行必须是 `result = ...`；不要把最终结果只赋给其它变量。
7. 禁止 `print` 代替 result；禁止返回 None。

## Debug 方式（收到上次失败信息时必须执行）
1. **先读错误类型再改代码**，不要盲目重写：
   - `KeyError`：列名不存在 → 对照预览 JSON 的 `columns` 修正拼写；多表时确认列在哪个表。
   - `TypeError`：类型不匹配 → 先 `astype` / `to_numeric` / `to_datetime`；避免字符串与数字直接运算。
   - `ValueError`：参数非法 → 检查 merge 的 `on` 键是否两表都有；groupby 列是否存在。
   - `AttributeError`：对象无此属性 → 确认上一步返回的是 DataFrame 而非 Series/标量。
   - `IndexError`：索引越界 → 改用条件筛选，避免硬编码行号。
2. **最小改动原则**：只修复报错点，保留正确逻辑，不要换用全新思路。
3. **Merge 失败**：先 `print` 不可用；改用更安全的 `how='left'`，并检查键列 dtype 是否一致（必要时 `.astype(str)` 统一）。
4. **空结果**：检查 `dropna` / 筛选条件是否过严；用更宽条件或分步保留中间变量。
5. **仍须满足沙箱约束**：无 import、无 open、最终赋值 result。

## 输出格式
仅输出一个 ```python 代码块，或纯 Python 代码。"""

_RETRY_DEBUG_HINTS: dict[str, str] = {
    "KeyError": "列名不存在：对照预览 JSON 的 columns 修正；多表时确认列属于 df1 还是 df2。",
    "TypeError": "类型错误：对参与运算的列先用 pd.to_numeric / pd.to_datetime / astype 转换。",
    "ValueError": "参数或数据不合法：检查 merge 的 on 键、groupby 列、dropna subset 是否存在。",
    "AttributeError": "对象类型不对：确认上一步返回的是 DataFrame；Series 需 to_frame() 或换写法。",
    "IndexError": "索引越界：改用布尔筛选 df1[df1['列']==值]，不要硬编码 iloc 行号。",
    "SecurityError": "违反沙箱规则：删除 import/open/eval/exec 等，仅保留 pd/np 数据处理。",
}


class CodeGenerationError(RuntimeError):
    """代码生成或校验失败。"""


@dataclass
class CodeGenerationResult:
    code: str
    raw_response: str
    model: str


def format_data_context(data_preview: dict[str, Any]) -> str:
    """将 file_parser / dataset_registry 预览摘要格式化为 LLM 可读上下文。"""
    if not data_preview:
        raise ValueError("data_preview cannot be empty.")

    if data_preview.get("all_datasets"):
        context: dict[str, Any] = {
            "dataset_count": data_preview.get("dataset_count"),
            "dataset_keys": data_preview.get("dataset_keys"),
            "datasets": data_preview["all_datasets"],
            "primary_table_key": data_preview.get("dataset_keys", ["df1"])[0]
            if data_preview.get("dataset_keys")
            else "df1",
        }
    else:
        context = {
            "filename": data_preview.get("filename"),
            "shape": data_preview.get("shape"),
            "columns": data_preview.get("columns"),
            "dtypes": data_preview.get("dtypes"),
            "null_counts": data_preview.get("null_counts"),
            "head": data_preview.get("head"),
        }
    try:
        return json.dumps(context, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise ValueError("Failed to serialize data preview.") from exc


def _retry_debug_section(
    *,
    previous_code: str,
    previous_error: str,
    previous_error_type: str,
    retry_count: int,
) -> str:
    """组装回溯重试时的 debug 指引。"""
    hint = _RETRY_DEBUG_HINTS.get(
        previous_error_type.strip(),
        "通读报错信息，对照预览 JSON 检查列名、类型与 merge/groupby 参数。",
    )
    return (
        f"## 上次执行失败（第 {retry_count} 次失败后请求修复）\n"
        f"- 错误类型：{previous_error_type or 'Unknown'}\n"
        f"- 错误信息：{previous_error.strip()}\n"
        f"- 修复提示：{hint}\n\n"
        f"失败代码：\n```python\n{previous_code.strip()}\n```\n\n"
        "请按 Debug 方式最小改动修复，仍须满足沙箱约束，并赋值 result。\n\n"
    )


def build_user_message(
    user_request: str,
    data_preview: dict[str, Any],
    *,
    previous_code: str = "",
    previous_error: str = "",
    previous_error_type: str = "",
    retry_count: int = 0,
) -> str:
    """组装用户侧提示词；重试时附带上一轮失败代码与报错。"""
    if not user_request or not user_request.strip():
        raise ValueError("user_request cannot be empty.")
    context = format_data_context(data_preview)
    ensure_correction_records_loaded()
    correction_section = format_similar_corrections_for_prompt(
        user_request.strip(),
        data_preview,
        error_type=previous_error_type.strip() or None,
        retry_count=retry_count,
    )
    message = f"用户需求：{user_request.strip()}\n\n"
    if correction_section:
        message += correction_section
    message += f"数据表预览（JSON）：\n{context}\n\n"
    if previous_code.strip() and previous_error.strip():
        message += _retry_debug_section(
            previous_code=previous_code,
            previous_error=previous_error,
            previous_error_type=previous_error_type,
            retry_count=retry_count,
        )
    message += (
        "请根据用户需求与预览 JSON 中的列名，"
        "选用合适模板生成 Pandas 代码。"
    )
    return message


def extract_python_code(text: str) -> str:
    """从 LLM 回复中提取 Python 代码。"""
    if not text or not str(text).strip():
        raise CodeGenerationError("LLM response is empty.")

    content = str(text).strip()
    match = _CODE_BLOCK_PATTERN.search(content)
    if match:
        return match.group(1).strip()

    if any(keyword in content for keyword in ("df", "pd.", "np.", "result")):
        return content
    raise CodeGenerationError("No Python code found in LLM response.")


def sanitize_generated_code(code: str) -> str:
    """移除 import 语句并清理空白。"""
    cleaned_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        cleaned_lines.append(line.rstrip())
    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned:
        raise CodeGenerationError("Generated code is empty after sanitization.")
    return cleaned


def validate_generated_code(code: str) -> None:
    """静态校验 + RestrictedPython 编译。"""
    try:
        validate_code_security(code)
        compile_user_code(code)
    except SecurityError as exc:
        raise CodeGenerationError(f"Generated code failed security check: {exc}") from exc
    except Exception as exc:
        raise CodeGenerationError(f"Generated code failed compilation: {exc}") from exc


def create_chat_model(
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> ChatOpenAI:
    """创建 LangChain ChatOpenAI 客户端（OpenAI 兼容接口）。"""
    try:
        return ChatOpenAI(
            model=model or OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            temperature=temperature,
        )
    except Exception as exc:
        raise CodeGenerationError(f"Failed to create chat model: {exc}") from exc


def generate_pandas_code(
    user_request: str,
    data_preview: dict[str, Any],
    *,
    llm: ChatOpenAI | None = None,
    model: str | None = None,
    previous_code: str = "",
    previous_error: str = "",
    previous_error_type: str = "",
    retry_count: int = 0,
) -> CodeGenerationResult:
    """调用 LLM 生成并通过沙箱规则校验的 Pandas 代码。"""
    user_message = build_user_message(
        user_request,
        data_preview,
        previous_code=previous_code,
        previous_error=previous_error,
        previous_error_type=previous_error_type,
        retry_count=retry_count,
    )
    chat = llm or create_chat_model(model=model)
    model_name = getattr(chat, "model_name", None) or model or OPENAI_MODEL

    try:
        response = chat.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
    except Exception as exc:
        log.exception("LLM invocation failed")
        raise CodeGenerationError(f"LLM invocation failed: {exc}") from exc

    raw_content = getattr(response, "content", "")
    if isinstance(raw_content, list):
        raw_text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        )
    else:
        raw_text = str(raw_content)

    try:
        code = sanitize_generated_code(extract_python_code(raw_text))
        validate_generated_code(code)
    except CodeGenerationError:
        raise
    except Exception as exc:
        raise CodeGenerationError(f"Failed to process LLM response: {exc}") from exc

    log.info(
        "Generated pandas code ({} chars), model={}, retry_count={}",
        len(code),
        model_name,
        retry_count,
    )
    return CodeGenerationResult(code=code, raw_response=raw_text, model=model_name)


def _run_self_check(*, live: bool = False) -> None:
    """模块内置自检。"""
    print("=== code_generator self-check ===")

    preview = {
        "filename": "demo.csv",
        "shape": [3, 3],
        "columns": [
            {"name": "name", "dtype": "object"},
            {"name": "amount", "dtype": "int64"},
            {"name": "date", "dtype": "object"},
        ],
        "dtypes": {"name": "object", "amount": "int64", "date": "object"},
        "null_counts": {"name": 0, "amount": 1, "date": 0},
        "head": [
            {"name": "Alice", "amount": 100, "date": "2024-01-01"},
            {"name": "Bob", "amount": None, "date": "2024-01-02"},
        ],
    }

    sample_response = (
        "```python\n"
        "work = df1.dropna(subset=['amount'])\n"
        "result = work['amount'].sum()\n"
        "```"
    )
    extracted = sanitize_generated_code(extract_python_code(sample_response))
    print(f"extract/sanitize: {extracted!r}")
    validate_generated_code(extracted)
    print("static validation: ok")

    context = format_data_context(preview)
    print(f"context length: {len(context)} chars")

    if live:
        print("live LLM test...")
        outcome = generate_pandas_code(
            "删除 amount 为空的行，并计算 amount 总和",
            preview,
        )
        print(f"model: {outcome.model}")
        print(f"code:\n{outcome.code}")
    else:
        print("live LLM test skipped (use --live to enable)")

    print("=== done ===")


__all__ = [
    "CodeGenerationError",
    "CodeGenerationResult",
    "build_user_message",
    "create_chat_model",
    "extract_python_code",
    "format_data_context",
    "generate_pandas_code",
    "sanitize_generated_code",
    "validate_generated_code",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="code_generator self-check")
    parser.add_argument(
        "--live",
        action="store_true",
        help="call real LLM API (requires valid .env)",
    )
    args = parser.parse_args()
    _run_self_check(live=args.live)
