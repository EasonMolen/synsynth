"""
通过 Ollama 本地 HTTP API 调用 Gemma-4-26B。

需要事先下载模型：
    ollama pull gemma4:26b

提供与 Unsloth 版本相同的 generate() / generate_structured() 接口，
但实际通过 Ollama REST API（localhost:11434）执行推理。
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Optional

from synsynth_config import (
    MODEL_REPO, MODEL_QUANT, MAX_SEQ_LEN,
    TEMPERATURE, TOP_P, CACHE_DIR, WORKSPACE_ROOT, logger,
)

# ── Ollama 配置 ──────────────────────────────────────────────────────────
OLLAMA_BASE   = "http://127.0.0.1:11434"
OLLAMA_MODEL  = "gemma4:26b"          # ollama list 中的模型名称
_TIMEOUT      = 600                    # 超时时间（秒）；长回答可能耗时较久


def _ollama_chat(
    messages: list[dict],
    *,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = 2048,
    json_format: bool = False,
) -> str:
    """调用 POST /api/chat，使用非流式模式。

    Gemma-4 的推理内容位于 message.thinking，最终答案位于
    message.content，因此令牌预算要覆盖两个阶段。
    """
    # 为 thinking 和 content 保留足够的令牌预算。
    # Gemma-4 的 thinking 常用掉 2000 到 3000 个令牌，
    # 最少预留 4096 个，避免截断。
    effective_tokens = max(max_tokens * 4, 4096)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": effective_tokens,
            "num_ctx": MAX_SEQ_LEN,
        },
    }
    if json_format:
        payload["format"] = "json"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            msg = body.get("message", {})
            content = msg.get("content", "").strip()
            thinking = msg.get("thinking", "").strip()

            # 检测只含空白或括号等无效字符的内容
            if content and json_format:
                import re as _re
                if not _re.search(r'[a-zA-Z0-9]', content):
                    logger.debug("Content corrompu (pas de caractères utiles) — fallback thinking.")
                    content = ""

            # 如果 content 为空，尝试从 thinking 中提取答案
            if not content and thinking:
                logger.debug("Content vide — extraction depuis thinking.")
                if json_format:
                    content = _extract_last_json_from_thinking(thinking)
                else:
                    content = _extract_answer_from_thinking(thinking)

            return content
    except urllib.error.URLError as e:
        logger.error("无法连接 Ollama（%s）。请检查服务是否已启动。", e)
        raise RuntimeError(f"无法连接 Ollama：{e}") from e


def _extract_last_json_from_thinking(thinking: str) -> str:
    """从 thinking 文本中提取最后一个有效的 JSON 对象。

    Gemma-4 可能在 thinking 开头复述模板，真实答案位于末尾。
    因此遍历所有候选 JSON，并取最后一个有效对象。
    """
    import re
    candidates = list(re.finditer(r'\{[^{}]*\}', thinking, re.DOTALL))
    # 逆序遍历，找到最后一个有效 JSON
    for m in reversed(candidates):
        try:
            json.loads(m.group())
            return m.group()
        except (json.JSONDecodeError, ValueError):
            continue
    # 找不到有效 JSON 时退回到文本提取方式
    return _extract_answer_from_thinking(thinking)


def _extract_answer_from_thinking(thinking: str) -> str:
    """从 Gemma-4 的 thinking 文本中提取最终答案。

    搜索 Direct answer、Réponse、Answer 等标签；
    如果都没有，则返回最后一个非空行。
    """
    import re
    # 匹配 Gemma 在 thinking 中输出最终答案时常见的格式
    for pattern in [
        r"(?:direct answer|réponse finale?|answer|réponse)\s*:\s*[\"«]?(.+?)[\"»]?\s*$",
        r"(?:donc|thus|so)\s*[,:]\s*(.+)$",
    ]:
        m = re.search(pattern, thinking, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip().strip('"').strip("«»").strip()

    # 退回到末尾有意义的文本行，跳过单独的项目符号
    lines = [l.strip() for l in thinking.strip().splitlines() if l.strip()]
    if lines:
        # 取最后一个看起来像答案的文本行
        for line in reversed(lines):
            cleaned = re.sub(r"^[\*\-\•\d\.]+\s*", "", line).strip()
            if len(cleaned) > 5:
                return cleaned
        return lines[-1]
    return thinking


def _check_ollama():
    """检查 Ollama 服务与目标模型是否可用。"""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in body.get("models", [])]
            if not any(OLLAMA_MODEL in m for m in models):
                logger.warning(
                    "Modèle '%s' absent d'Ollama. Disponibles : %s",
                    OLLAMA_MODEL, models,
                )
            else:
                logger.info("Ollama OK — modèle '%s' trouvé.", OLLAMA_MODEL)
    except Exception as e:
        logger.error("Impossible de contacter Ollama : %s", e)
        raise


# 不在导入模块时连接 Ollama，以便离线查看命令行帮助和分析代码。


def generate(
    prompt: str,
    *,
    system: str = "",
    max_new_tokens: int = 2048,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    json_format: bool = False,
) -> str:
    """通过 Ollama 根据提示词生成回答。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return _ollama_chat(
        messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        json_format=json_format,
    )


def generate_structured(
    prompt: str,
    *,
    system: str = "",
    json_mode: bool = False,
    max_new_tokens: int = 4096,
) -> str:
    """使用结构化输出提示词生成 JSON 或 Markdown。"""
    if json_mode:
        system = (system + "\n" if system else "") + (
            "Tu dois répondre UNIQUEMENT avec un objet JSON valide, "
            "sans texte avant ni après."
        )
    return generate(
        prompt,
        system=system,
        max_new_tokens=max_new_tokens,
        temperature=0.1,   # 结构化输出使用接近确定性的低温度
        json_format=json_mode,
    )
