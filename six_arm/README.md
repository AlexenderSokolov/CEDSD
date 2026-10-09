# 六臂共享 XLS-R 对照实验

本目录提供论文补充实验的六臂模型、训练、评估和诊断源码，以及本地检查与执行入口。最终比较使用 **seed 17、29，六臂共 12 个模型配置**。它对应共享 XLS-R 的新增对照实验；仓库根目录保留的原系统实现与本目录具有不同实验配置。

科学源码取自同一冻结交付快照，模型与训练实现保持原样。代码包可用于阅读实现、检查算子和衔接已有私有实验工件；完整重放仍需要受限音频、原始清单、模型文件、历史预处理缓存和冻结状态。数据或训练权重不随仓库提供，也不保证仅凭这些代码文件复现论文成绩。

## 六臂定义

所有分支均使用相同 XLS-R 声学主干及相同训练协议。`AE` 中的额外音频表示是冻结的 **emotion2vec+ large 帧级特征（1024 维）**。

| 论文名称 | `--arm` | 增加的分支或聚合方式 |
|---|---|---|
| Acoustic | `acoustic` | 仅 XLS-R 声学表示 |
| AE | `ae` | emotion2vec 特征均值池化，经容量分支映射后与声学表示拼接 |
| PoolJoint | `pooljoint` | emotion2vec 与 BERT 文本特征分别池化，再联合映射、拼接 |
| CA | `ca` | emotion2vec 查询 BERT 文本的普通跨注意力聚合 |
| Linear | `linear` | 使用低于均匀注意力参照的线性差额作为聚合权重 |
| Log | `log` | 使用对数差额、上限截断及偏离强度缩放的逆聚合 |

Linear 与 Log 是同时保留的实验臂。具体公式与操作顺序见 [`closeout_model.py`](uica_exec/src/uica/closeout_model.py) 和 [`operators.py`](uica_exec/src/uica/operators.py)。

## 固定科学配置

- 音频转单声道、16 kHz，取开头最多 5 秒；短片段保留实际长度，并在组批时使用掩码。
- XLS-R-300m 声学主干仅解冻最后 4 层；掩码均值与标准差池化后映射到 256 维。
- SenseVoiceSmall 产生转录；冻结 emotion2vec+ large 与 `bert-base-chinese` 分别提供 1024 维音频帧和 768 维文本 token 表示。文本最多 128 token，包含特殊 token。
- 关系表示为 256 维、8 个注意力头；Log 的截断上限 `kappa=5`。
- 训练使用类别加权交叉熵、BF16、有效批量 16（微批量 4）；主干与头部学习率分别为 `1e-5`、`1e-3`。最多 40 个 epoch，至少 5 个 epoch，耐心值 7。
- 开发集 EER 选模，并列时使用未加权 log-loss、再选更早 epoch；固定模型评估使用 FP32，关闭 TF32，批量 4。测试汇总按 seed 等权计算。

详细字段见 [`six_arm.example.json`](uica_exec/configs/six_arm.example.json)。这是路径占位、`started_unix=null` 的配置模板，仅供理解字段及新授权实验参考，不能直接作为训练配置。接续既有检查点必须使用其原始私有科学配置或已验证的迁移配置；修改模板路径不足以恢复原实验。模型版本和资产标识见去除本机 `snapshot_path` 的 [`model-lock.json`](uica_exec/inputs/model-lock.json)。

## 开始检查

在 Python 3.12 环境中，从本目录执行：

```bash
python -m pip install -r requirements.txt
bash run_checks.sh
```

没有 Bash 时可执行 `python tools/check_code.py`。检查覆盖 Python AST、内部导入依赖、六臂 CPU 算子检查、原有 3 个测试模块和新增入口回归测试。它不加载真实音频、预训练权重或启动 GPU 训练，因此检查通过仅说明这些工程检查通过。

本次发布检查已在 Windows、Python 3.12.14、PyTorch 2.5.1+cpu 环境通过：106 项六臂工程检查、19 项原测试与 2 项入口回归测试均通过。此次未执行真实音频／预训练模型推理或 GPU 训练。

完整音频环境另见 [`requirements-audio.txt`](requirements-audio.txt)。GPU 训练需匹配的 PyTorch／torchaudio CUDA 环境；预处理保留 Linux 文件属主检查，建议在 Linux 或 WSL 中执行。

## 文件入口

| 文件 | 用途 |
|---|---|
| [`closeout_model.py`](uica_exec/src/uica/closeout_model.py) | 六臂模型与 CPU 工程检查 |
| [`closeout_pretrained.py`](uica_exec/src/uica/closeout_pretrained.py) | 锁定预训练声学模型的加载与初始化 |
| [`full_data.py`](uica_exec/src/uica/full_data.py)、[`full_prepare.py`](uica_exec/src/uica/full_prepare.py) | 数据审计、录音家族划分、缓存构建与复用 |
| [`full_runtime.py`](uica_exec/src/uica/full_runtime.py)、[`full_training.py`](uica_exec/src/uica/full_training.py) | 执行状态、完整训练池、断点续训与 FP32 推理 |
| [`coling_evaluation.py`](uica_exec/src/uica/coling_evaluation.py) | 两 seed 冻结评估、家族配对统计与汇总 |
| [`coling_diagnostics.py`](uica_exec/src/uica/coling_diagnostics.py) | donor 文本替换、关系分量与保存预测分析 |
| [`run_replay.sh`](run_replay.sh)、[`tools/replay.py`](tools/replay.py) | 对接私有工件的本地执行入口 |

执行步骤及所需私有文件见 [REPRODUCTION.md](docs/REPRODUCTION.md)，源码来源与保留范围见 [SOURCE_NOTES.md](docs/SOURCE_NOTES.md)。源码中原三 seed、18-fit 计划账本仍被保留；它不是 seed 43 已完成或 18 次训练均已完成的记录。
