# CEDSD: cross model emotion-driven deepfake speech detection

[English README](README.md)

## 新增六臂受控实验

已完成的共享 XLS-R 实验代码见 [six_arm/](six_arm/README.md)，包含
Acoustic、AE、PoolJoint、CA、Linear、Log 六臂，实际完成 seeds 17、29。
目录提供模型、全量训练、评价、诊断、CPU 检查和配置模板；
理解模型先看 [closeout_model.py](six_arm/uica_exec/src/uica/closeout_model.py)。
完整重放所需的受限数据与历史工件见包内复现说明。下文及仓库根目录仍为原 CEDSD 实现。

本仓库包含用于研究的轻量化代码，目标是结合情感线索、声学特征、ASR 文本语义、逆向注意力和 FAPI 语言模型一致性评分来检测伪造语音。


## 结构概览

- `main.py`：训练、测试、预测和逆向注意力检查的命令行入口。
- `config.py`：训练、缓存、不确定性加权、FAPI 和可解释性参数。
- `Dataset_all.py`：统一多模态数据集、音频读取、MFCC/F0 提取和缓存占位。
- `Model_all.py`：整合频谱、情感、文本、逆向注意力、IACA 和 FAPI 的主模型。
- `Audio_united/`：波形与情感分支，包括 MFCASTDA 和 emotion2vec。
- `Text_encoder/`：BERT 文本编码器和 FunASR 本地模型接口。
- `spectrum/`：MFCC/F0 声学编码器。
- `offline_cache.py`：ASR、FAPI、emotion2vec 和声学特征缓存。
- `prewarm_*.py`：离线缓存预热脚本。
- `scripts/README.md`：保留旧入口的脚本命令索引。

## 安装

建议使用 Python 3.10 或更新版本。

```bash
cd Program_lightweight_for_github
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

较重的可选依赖包括 `torch`、`torchaudio`、`funasr`、`modelscope`、`torchcrepe`、`laion-clap` 和 `transformers`。请按本机 CUDA 环境安装对应的 PyTorch 版本。

## 数据格式

CSV 至少需要包含：

- `file`：相对于 split root 的音频文件名或路径。
- `label`：真实语音使用 `bonafide`/`real`，伪造语音使用 `spoof`/`fake`。
- 情感分数列：`angry`、`disgusted`、`fearful`、`happy`、`neutral`、`other`、`sad`、`surprised`、`unknown`。

默认相对路径：

```text
data/
  train/train.csv
  train/<audio files>
  val/val.csv
  val/<audio files>
  test/test.csv
  test/<audio files>
```

所有路径都可以通过命令行参数或环境变量覆盖，例如 `FAPI_TRAIN_CSV`、`FAPI_TRAIN_ROOT`、`FAPI_MODEL_PATH`、`FAPI_ASR_MODEL_DIR`、`FAPI_CLAP_CKPT`。

## 常用命令

查看主入口参数：

```bash
python main.py --help
```

训练：

```bash
python main.py --mode train ^
  --train-csv data/train/train.csv --train-root data/train ^
  --val-csv data/val/val.csv --val-root data/val ^
  --output-dir outputs_fapi_viz
```

测试：

```bash
python main.py --mode test ^
  --test-csv data/test/test.csv --test-root data/test ^
  --model-path outputs_fapi_viz/run_default/best_model.pth
```

单文件或多文件预测：

```bash
python main.py --mode predict ^
  --model-path outputs_fapi_viz/run_default/best_model.pth ^
  --audio path\to\sample.wav
```

缓存预热：

```bash
python prewarm_offline_cache.py --mode all
python prewarm_e2v_cache.py
python prewarm_acoustic_cache.py
```

下载本地 ASR 模型：

```bash
python Text_encoder/FunASR/sense_download.py --help
python Text_encoder/FunASR/nano_download.py --help
```

## 输出

默认输出目录是 `outputs_fapi_viz/`，通常包含日志、最佳权重、收敛摘要、可解释性报告和缓存特征。这些生成文件已加入 `.gitignore`。

## 引用

论文投稿或录用后会更新正式引用。

## 许可与负责任使用

代码公开后使用 MIT License。数据集、预训练权重、第三方模型、本地缓存和生成报告不随代码自动发布，除非另有独立许可。

请负责任地使用本项目。语音伪造检测具有双重用途，不应被用于缺乏证据的个人判断、隐私侵入式监控或冒充行为。

## 如何重建环境

在项目根目录运行：

```bash
pip install -r requirements.txt
# （若需要 CLAP 功能）
pip install laion-clap
```

注意：若需安装带 CUDA 支持的 `torch` / `torchaudio` 轮子，请根据你的 CUDA 版本选择合适的预编译包或遵循 PyTorch 官方安装说明。
