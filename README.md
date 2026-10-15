# MiniMatch · 基于 MiniLM 的商品检索

本地运行的商品搜索与比较工具。使用商品任务微调后的 **MiniLM** 理解查询，结合 **BM25** 关键词检索，帮助用户查找商品、阅读详情并比较候选。

## 可以做什么

- **智能搜索**：关键词与语义融合，也可切换语义搜索或关键词搜索。
- **商品详情**：查看原始商品记录中的品牌、颜色、产品要点与描述。
- **商品对比**：最多比较三件商品，支持跨搜索保留和刷新恢复。
- **继续浏览**：分批加载搜索结果，保存最近搜索。
- **本地运行**：查询不调用外部大模型 API，无需账号。

当前为 **140,084 件英语历史商品**，使用 Amazon ESCI 公开数据。不提供实时价格、库存、图片和购买入口；部分商品描述不完整。输入英文需求，例如 `wireless mouse`、`comfortable running shoes` 或 `stainless steel water bottle`。

## 运行

本机已有数据和模型时，双击 **start.cmd**，启动完成后打开 **http://127.0.0.1:8899/**。也可使用已安装项目依赖的 Python：

```powershell
python scripts/run_windows.py -m productsearch
```

可用 `--port 8900` 指定其他本机端口。服务默认仅监听 `127.0.0.1`。内部 Python 包保留为 `productsearch`。

**首次克隆注意**：仓库只包含代码、配置和复现说明，不包含数据集、模型权重、索引或运行环境。安装依赖后需要在本机准备训练产物，不能直接获得已经训练好的模型：

```powershell
python -m pip install -e ".[dense,dev]"
python scripts/run_windows.py scripts/bootstrap_application.py --plan
# 确认数据下载和训练开销后执行：
python scripts/run_windows.py scripts/bootstrap_application.py
python scripts/run_windows.py -m productsearch
```

训练需要 CUDA 版 PyTorch 与可用 GPU；本项目在 Windows / RTX 4060 8 GB 上运行。初始数据下载约 1.16 GB，完整准备和训练明显长于普通启动。详细步骤与环境说明见[复现说明](docs/REPRODUCE.md)。已有产物时不会自动重新训练。Windows 可设置 `MINIMATCH_PYTHON`，或在忽略上传的 `runtime/python.path` 中填写已有解释器路径。

## 模型与质量

语义编码器基于 `sentence-transformers/all-MiniLM-L6-v2`，采用多正例对比训练与原模型向量约束。商品向量预计算；智能搜索通过 RRF 融合 BM25 与语义排名。模型、索引及融合权重由应用清单固定，启动时检查文件指纹。

固定模型在同一商品库、1,000 条已使用测试查询上的回归结果：

| 方法 | 已标注 E Recall@100 |
|---|---:|
| BM25 | 65.33% |
| 未微调 MiniLM | 66.92% |
| 微调后的 MiniLM | 68.69% |
| BM25 + 未微调 MiniLM | 72.82% |
| BM25 + 微调后的 MiniLM | 73.47% |

这些查询在项目早期已评测，**不构成新的未查看测试集**。召回只针对已标注的精确商品，不代表所有商品的准确率、点击率或转化率。微调单模型相对未微调的配对区间为正；混合改进的区间包含零。详见[质量说明](docs/QUALITY.md)。

## 开发

```powershell
python scripts/run_windows.py -m pytest -q
python scripts/run_windows.py scripts/benchmark_application.py
```

公开源码包含 32 项当前算法与应用检查；无数据、无模型环境下为 21 项通过、11 项按预期跳过。完整本机环境加上历史回归共 64 项全部通过。核心代码位于 `src/productsearch/`，网页位于其 `web/` 子目录；[系统结构](docs/ARCHITECTURE.md)说明数据、检索与交互之间的关系。

代码采用 [MIT License](LICENSE)。ESCI 数据和 MiniLM 预训练权重分别遵循上游许可，未随仓库再分发。来源：[Amazon ESCI](https://github.com/amazon-science/esci-data)、[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。
