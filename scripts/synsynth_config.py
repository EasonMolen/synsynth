"""
SYNSYNTH+ 的集中配置；文件读写限制在项目目录中。
"""
import os
import sys

# ── 工作区绝对路径 ───────────────────────────────────────────────────────
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 路径越界保护 ─────────────────────────────────────────────────────────
def safe_path(*parts: str) -> str:
    """返回位于 WORKSPACE_ROOT 内的绝对路径。
    如果解析后的路径越过工作区边界，则抛出 RuntimeError。"""
    workspace_root = os.path.realpath(WORKSPACE_ROOT)
    candidate = os.path.realpath(os.path.join(WORKSPACE_ROOT, *parts))
    try:
        inside_workspace = os.path.commonpath((workspace_root, candidate)) == workspace_root
    except ValueError:  # Windows 上的路径可能位于不同磁盘
        inside_workspace = False
    if not inside_workspace:
        raise RuntimeError(
            f"禁止访问：{candidate!r} 位于工作区 {WORKSPACE_ROOT!r} 之外"
        )
    return candidate


# ── 工作目录 ────────────────────────────────────────────────────────────
DATA_DIR       = safe_path("data")
RESULTS_DIR    = safe_path("results")
MODELS_DIR     = safe_path("models")
CACHE_DIR      = safe_path("cache")
ARTICLE_DIR    = safe_path("article")
LOGS_DIR       = safe_path("logs")

for d in (DATA_DIR, RESULTS_DIR, MODELS_DIR, CACHE_DIR, ARTICLE_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

# ── 模型参数 ────────────────────────────────────────────────────────────
MODEL_REPO    = "unsloth/gemma-4-26B-A4B-it-GGUF"
MODEL_QUANT   = "UD-Q4_K_XK"
MAX_SEQ_LEN   = 8192
TEMPERATURE   = 0.3
TOP_P         = 0.9

# ── 各任务的最佳模型（2026 年 4 月 5 日基准测试）──────────────────────────
TASK_MODELS = {
    "extraction": "gemma4:26b",        # F1=0.75, P=1.00, R=0.60
    "query":      "qwen3-deep:latest",  # Acc=1.00, Cypher=1.00, 20s
    "multihop":   "phi4:latest",        # EM=0.80, 124s
    "rag":        "mistral-small:latest",# Faith=1.00, Relev=0.92
}
DEFAULT_MODEL = "gemma4:26b"  # 未列出任务时使用的默认模型

# ── 评估目标阈值 ────────────────────────────────────────────────────────
TARGET_F1_EXTRACTION   = 0.85   # DocRED/TACRED 上 F1 达到 85%
TARGET_ACCURACY_QUERY  = 0.90   # WebQuestionsSP 上准确率达到 90%
TARGET_FAITHFULNESS    = 1.00   # RAGAS 忠实度约为 100%

# ── 实验参数 ────────────────────────────────────────────────────────────
RANDOM_SEED        = 42
BATCH_SIZE         = 4
NUM_EVAL_SAMPLES   = 200       # 每项基准测试的样本数
NUM_MULTIHOP_HOPS  = 3         # 多跳推理的最大跳数

# 各实验样本数，用于完整的论文基准测试
NUM_EXTRACTION_SAMPLES = 500   # Re-DocRED 验证集（从 Hugging Face 下载）
NUM_QUERY_SAMPLES      = 200   # WebQuestionsSP 风格（预置与合成样本）
NUM_MULTIHOP_SAMPLES   = 500   # HotpotQA 验证集（从 Hugging Face 下载）
NUM_RAG_SAMPLES        = 50    # RAGAS 风格（预置与合成样本）

# ── 日志配置 ────────────────────────────────────────────────────────────
import logging

LOG_FILE = safe_path("logs", "synsynth.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("SYNSYNTH+")
