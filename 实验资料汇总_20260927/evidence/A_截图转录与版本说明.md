# 本次保留版本：浅层引导三尺度权重选择

归档日期：2026-09-10。按用户要求保留本次版本，不修改模型实现。

模型：EffB1_ZoomNeXt_WeightGuided。
本次运行选择：--variant guidance。
配置：configs/wg_guidance.py。
有效开关：use_weight_guidance=True，use_frequency=False，use_edge=False。
浅层编码特征引导第3、2层的三尺度融合权重；新增权重偏移分支末层零初始化。
此实验不是旧SG、FEDER＋SG或SG＋FSP版本。目录保留完整依赖及其他消融配置，本次结果只对应guidance配置。

## 用户服务器截图中的结果

| 数据集 | Sm ↑ | wFm ↑ | MAE ↓ |
| --- | ---: | ---: | ---: |
| CHAMELEON | 0.907 | 0.854 | 0.022 |
| CAMO | 0.831 | 0.773 | 0.061 |
| COD10K | 0.838 | 0.731 | 0.029 |
| NC4K | 0.838 | 0.760 | 0.046 |

截图显示End training及state_final.pth保存成功。数值按截图记录，未在本机重新训练或评估。

服务器对应实验目录：
`/home/gyz/ZoomNeXt/weight_guided_runs/20260909_210557_guidance_zbw88wai/`

该服务器目录中的源码快照、配置、日志和权重是追溯本次正式运行的依据，建议完整保留。

## 本地文件来源

本目录复制自 E:/ZoomNeXt/WeightGuided-20260909，排除Python缓存；48个原文件复制后逐一SHA-256比对通过，清单见source_files.sha256.json。
此处是本地交付代码的归档，尚未与服务器源码快照逐字节核对；不包含服务器训练权重和完整训练日志。
validation.txt为已有本地功能检查记录，不是本次新增训练结果。

## 后续运行

在服务器上使用此版本时，保持目录结构，并提供服务器数据配置dataset.yaml：

```bash
/home/gyz/anaconda3/envs/zoomnext/bin/python -u run_weight_guided.py --variant guidance --gpu 1 --data-cfg dataset.yaml
```

此命令会开始新的训练；仅查看或保存本次结果时无需执行。
