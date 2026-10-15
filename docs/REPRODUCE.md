# 首次准备与复现

代码包不包含数据、权重、索引或依赖运行时。日常启动只需要已有的应用产物；首次克隆需要单独准备。

## 依赖与解释器

Python 3.12。先选择自己的已有 Python 解释器，再安装项目依赖：

```powershell
python -m pip install -e ".[dense,dev]"
```

训练需要 CUDA 版 PyTorch。项目实际使用 PyTorch 2.8.0 / CUDA 12.8；应按自己的 GPU 与驱动选择相容安装包，并先检查 `torch.cuda.is_available()`。日常在线查询使用 CPU。无需云账号或付费模型 API。

`requirements.txt` 记录本机已验证的核心依赖版本，可作为复现约束；PyTorch 的 CPU / CUDA 发行包仍需按运行设备单独选择。

Windows 启动脚本优先使用 `MINIMATCH_PYTHON`，其次使用本地 `runtime/python.path` 中的一行解释器路径，最后使用 `python`。这些配置不修改其他项目的环境。

## 从源文件构建产物

先查看步骤，不执行下载或训练：

```powershell
python scripts/run_windows.py scripts/bootstrap_application.py --plan
```

确定磁盘、下载和 GPU 资源后执行：

```powershell
python scripts/run_windows.py scripts/bootstrap_application.py
```

该入口依次下载固定提交的官方数据、下载固定 revision 的 MiniLM，准备规范化查询划分和商品库，准备 BM25 / 训练缓存与负例，执行当前训练协议，再注册应用清单。原始数据约 1.16 GB，缓存和训练产物还需要额外数 GB 空间。

当前训练包含三档学习率、冻结底层、检索目标、原始向量约束、训练池扩展和最终方案种子确认。最终模型及融合权重只由验证集选择。配置保存于 `configs/dense_v6.json`，旧数字后缀为内部实现与数据兼容名称，页面不会展示这些训练过程。bootstrap 不运行已经替代的历史模型实验。

代码与数据指纹在训练开始时固定，恢复时拒绝不一致的输入。新克隆没有历史产物，创建空的历史保持清单；已有本地保持清单不会覆盖。已有应用清单时，bootstrap 直接返回，不自动重训。

## 启动与验证

```powershell
python scripts/run_windows.py -m productsearch
python scripts/run_windows.py -m pytest -q
python scripts/run_windows.py scripts/benchmark_application.py
```

浏览器打开 `http://127.0.0.1:8899/`。固定模型评估入口为 `scripts/evaluate_application.py`，只读取固定模型与评测集，不训练、不根据分数更换模型。逐查询结果与本机性能存入忽略上传的 `reports/application/`。

随机性、硬件和依赖版本可能影响具体分数。本仓库提供代码与配置，不保证其他环境产生字节完全相同的模型。所有报告都应说明数据划分、商品库范围与评估查询是否曾经查看。
