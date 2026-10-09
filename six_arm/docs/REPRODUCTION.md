# 检查与实验重放

本目录的执行入口调用冻结源码中的实验函数。它不依赖原桌面调度器，但仍遵守原数据、缓存、模型身份及运行时钟约束。它面向这套已有实验工件，不是任意数据集的通用训练流水线。

## 1. 代码检查

以下命令均在 `six_arm/` 下运行：

```bash
python -m pip install -r requirements.txt
bash run_checks.sh
```

`run_checks.sh` 调用 `tools/check_code.py`，包括 AST 解析、内部导入闭包、`engineering_operator_checks("cpu")`、3 个原测试模块中的用例及入口回归测试。测试使用合成输入，不证明真实数据读取、模型下载、GPU 训练或论文成绩可复现。可用 `UICA_PYTHON=/path/to/python bash run_checks.sh` 选择现有环境。

## 2. 私有输入与环境

完整执行需要自行取得使用许可并恢复下列工件，代码包不包含它们：

| 工件 | 源码读取位置或要求 |
|---|---|
| 音频与标签 CSV | `paths.data_root` 下的音频及 `zh-label.csv` |
| 原始来源清单 | `six_arm/research/uica-full-20261004/source_full.csv`；用于核对标签 CSV 身份 |
| 历史样本清单 | `six_arm/uica_exec/inputs/full_manifest.jsonl`；用于历史来源与样本绑定 |
| 历史缓存 | `full.historical_cache`，必须包含匹配的 `preprocess.json`；复用的特征文件需与收据及样本身份匹配 |
| 预训练模型 | 配置中的 XLS-R、SenseVoice、emotion2vec、BERT 本地目录，匹配随包模型锁中的版本与资产 |
| 继续执行所需工件 | 原划分清单、缓存索引、模型检查点、开发集预测、训练状态、加载覆盖记录等，位于对应 `full.output_root`／`feature_store` |
| 最终评估所需工件 | 原 `state.json` 内冻结的 `closeout_clock`、`closeout_contract`，及合同引用的文件、评估／诊断锁和相应预测工件 |

清单应在取得许可后放入对应本地位置，不随本代码仓库提供。私有配置可放在仓库之外。

音频处理与 GPU 训练需另装 [`requirements-audio.txt`](../requirements-audio.txt)，并按硬件安装相互匹配的 PyTorch、torchaudio CUDA 构建。当前依赖文件记录所需版本，不替代 GPU 驱动及完整系统环境配置。预处理使用 `os.getuid()` 和文件属主检查，应使用 Linux／WSL 环境。

[`six_arm.example.json`](../uica_exec/configs/six_arm.example.json) 仅供阅读字段及配置新授权实验时参考，不能通过填写路径来接续原检查点。**接续或重放既有检查点，必须使用该检查点的原始私有科学配置或已验证的迁移配置。** `full_runtime.science_identity` 将完整 `models` 字段（包括本地路径）和 `training` 字段纳入身份校验；示例的 seed 列表为 `[17, 29]`，原私有科学配置保留 `[17, 29, 43]`，因此二者也不能直接互换。应保留源码的身份校验。

模板中的 `started_unix=null` 刻意不提供运行起点。恢复原实验时必须沿用原 T0 和状态；入口会拒绝占位路径或无效 T0，也不会通过重置时钟恢复已经到期的执行权限。

## 3. 本地阶段入口

查看参数不启动训练：

```bash
bash run_replay.sh --help
```

私有输入与匹配协议就绪后，使用以下命令形式。`/private/six_arm.json` 代表自己的私有配置路径。

```bash
bash run_replay.sh --config /private/six_arm.json --stage audit
bash run_replay.sh --config /private/six_arm.json --stage prepare
bash run_replay.sh --config /private/six_arm.json --stage engineering
bash run_replay.sh --config /private/six_arm.json --stage train --arm acoustic --seed 17
```

训练臂可选 `acoustic|ae|pooljoint|ca|linear|log`，seed 可选 `17|29`。每次命令只指定一个臂和一个 seed；已有匹配断点时追加 `--resume`。`engineering` 会使用真实数据、预训练模型与 GPU，和前面的 CPU 检查不同。

原最终阶段在其前置工件与时间约束满足时按以下形式调用：

```bash
bash run_replay.sh --config /private/six_arm.json --stage freeze
bash run_replay.sh --config /private/six_arm.json --stage evaluate
bash run_replay.sh --config /private/six_arm.json --stage statistics
bash run_replay.sh --config /private/six_arm.json --stage donor
bash run_replay.sh --config /private/six_arm.json --stage components
bash run_replay.sh --config /private/six_arm.json --stage analyze
bash run_replay.sh --config /private/six_arm.json --stage summarize
```

这些命令展示接口及阶段关系，不是可以从示例配置直接运行到底的脚本。`freeze` 先冻结诊断样本和映射，再冻结评估模型；其他最终阶段读取相应锁与工件。它们均要求已有私有 `state.json` 中的两 seed 合同和原时钟，入口不会创建替代合同或延长截止时间。已存在运行状态或配置含历史 `closeout_clock` 时，训练入口仅保留原先限定的 `Log / seed 29 / --resume` 路径；其他训练操作会被拒绝，删除配置字段也不会解除原状态中的限制。

## 4. 结果含义

最终科学比较使用六臂 × seed 17、29，共 12 个配置；保留源码中的 seed 43、18-fit 矩阵和小样本计划仅作历史账本。判定某次执行是否完成，应读取该次运行的实际状态、检查点和评估工件，不能从队列长度推断。

`coling_evaluation.py` 对应两 seed 最终评估及 2,000 次家族共享重采样，主比较区间为 98.75%。这与配置模板中仍保留的历史 `bootstrap_samples` 字段应分别理解。donor 与分量干预描述固定模型的行为；受约束文本替换不能单独证明或否定一般语义／情感信息的检测作用。

本代码包提供代码检查和有前置条件的执行入口。未随仓库提供受限数据、训练权重或完整私有状态，因此代码检查通过不构成端到端复现或成绩保证。
