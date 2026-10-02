# SYNSYNTH+ 中文运行说明

本项目用本地 Ollama 模型运行四类实验：关系抽取、文本到图查询、多跳推理、RAG 忠实度评估。主入口是 `scripts/run_synsynth.py`；`scripts/` 中的注释和文档字符串已改为中文。原作者的模型提示词、数据格式及结果字段仍保留原文，以免影响实验。

## 在 PyCharm 中查看和运行

1. 将项目根目录作为 PyCharm 项目打开，并选择已安装项目依赖的 Python 解释器。依赖列表见仓库原有的 `requirements.txt`。
2. 右键 `scripts` 文件夹，选择 **Mark Directory as → Sources Root**。这样 PyCharm 能解析 `synsynth_config` 等位于 `scripts` 中的内部模块。
3. 在项目根目录运行 `python scripts/run_synsynth.py --help` 查看中文参数说明。该命令无需启动 Ollama。
4. 运行实际实验前，启动 Ollama 并下载对应任务所需的模型。小规模试运行示例：

   ```powershell
   python scripts/run_synsynth.py --exp extraction --n-samples 20 --skip-article
   ```

完整实验对显存、内存和模型存储空间的要求较高，详见仓库原有的 `README.md`。

## 缺失的可选模块

- `gbnf_patch.py` **不在这个仓库中**。只有传入 `--gbnf` 时才需要它。程序默认在项目同级的 `PJKG5/gbnf_patch.py` 查找，也可以用 `--gbnf-patch-dir 目录` 指定该文件所在目录。只有本仓库时，请省略 `--gbnf`。
- `synsynth_selfimprove.py` 也不在仓库中，因此 `--self-improve` 暂时不可用；程序会给出明确错误。
- `--qlora` 的数据准备还依赖外部 `PJKG5/prepare_qlora_data.py`；缺少该文件时，训练无法进行。

如果你本地另有 `pyproject.toml`，请注意它可能是后来新建的项目文件。原仓库的依赖定义在 `requirements.txt`，其 `README.md` 声明 Python 3.10 或更新版本。仅安装一个空依赖列表的 `pyproject.toml` 无法运行完整实验。
