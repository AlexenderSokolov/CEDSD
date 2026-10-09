# 源码来源与发布范围

本目录从冻结交付包 `coling-closeout/delivery_final/code/uica_exec` 提取六臂实验所需源码。配置保留其 `scientific_snapshot=1393768c19406913f178258a1426adc2e7eabecf` 标识。该标识用于追溯原实验快照，不是本次代码仓库提交号。

## 保留的实现

`uica_exec/src/uica/` 中 17 个 Python 模块来自同一冻结快照，已核对与来源逐字节一致，科学实现保持原样：

- 六臂模型与预训练加载：`closeout_model.py`、`closeout_pretrained.py`、`operators.py`。
- 全量数据与训练执行：`full_data.py`、`full_prepare.py`、`full_runtime.py`、`full_training.py`。
- 评估与诊断：`full_evaluation.py`、`coling_evaluation.py`、`coling_diagnostics.py`。
- 共用依赖：`model.py`、`data.py`、`preprocess.py`、`training.py`、`metrics.py`、`common.py`、`__init__.py`。

其中 `model.py` 提供声学主干及池化路径，`data.py` 提供音频窗口与组批，`preprocess.py` 提供冻结编码器及缓存规则，`training.py` 提供共用训练工具。保留这些文件是为了满足六臂实现的真实依赖；文件中包含的旧入口或辅助函数不表示本次发布覆盖了整套早期实验。

原 `test_full_metrics.py`、`test_full_resume.py`、`test_full_epoch_evidence.py` 随包保留。其中 `test_full_resume.py` 的配置 fixture 改为本目录的 `six_arm.example.json`，并兼容模板中不含旧 `closeout` 字段，原测试断言不变。新增 `test_replay_entry.py` 验证模板拒绝与既有时钟限制。这些测试调整不涉及上述 17 个科学模块。`full_sampling.py`、旧 CLI、桌面调度器及服务器存储迁移逻辑不在本次发布范围。

## 本次发布新增的外围文件

| 文件 | 作用 |
|---|---|
| `tools/check_code.py`、`run_checks.sh` | 调用现有 CPU 检查、原测试，并检查解析与内部依赖 |
| `tools/replay.py`、`run_replay.sh` | 将本地参数映射到保留实验函数；验证配置与私有状态，保留原时钟约束 |
| `requirements.txt`、`requirements-audio.txt` | 分开列出核心检查依赖与额外音频依赖 |
| `uica_exec/configs/six_arm.example.json` | 对外路径模板；使用两 seed，清空 T0，不携带私有运行状态 |
| `uica_exec/inputs/model-lock.json` | 保留模型版本、文件名和资产标识，移除本机 `snapshot_path` |
| README 与本目录文档 | 解释六臂、运行入口、前置工件及结果边界 |

这些外围文件让代码包可检查、让持有原工件的使用者找到执行入口；它们不改写六臂模型、损失函数、训练逻辑或推理精度。

## 历史计划与实际比较

`full_runtime.py` 仍声明 seed 17、29、43，并据此生成 18-fit 队列；`full_evaluation.py` 也保留原矩阵及小样本辅助字段。这些是冻结源码的历史计划结构。

论文新增主比较使用 `coling_evaluation.py` 所定义的两 seed（17、29）六臂协议，合计 12 个配置。不要把历史队列解释为 18 次训练已完成，也不要将小样本计划视为已完成结果。单个 fit 的完成、部分可用或未执行状态必须以原运行工件为准。

音频、转录、逐样本清单、特征缓存、训练检查点或完整运行状态不随仓库提供。模型锁中的资产标识用于匹配使用者自行取得的模型，不包含模型权重。配置模板和源文件本身不足以生成原实验的私人合同、冻结时钟或模型成绩。
