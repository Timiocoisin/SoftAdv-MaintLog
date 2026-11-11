# 统一口径（只保留一个 OrdProg）

## 方法定义（唯一）
**OrdProg-Net** = `ordprog_full`（ResNet-18 + 亮金属通道 + CORAL + 进度头），报告 **3-seed 预测平均**。  
不再把 “full⊕appearance 融合” 当作第二个 Ours。

## 评价指标（加多了）
进度回归：`MAE_stage`、`RMSE`、`Spearman`、`Within-5`、`Within-10`  
阶段分类（由 \(\hat g\) 映射）：`Acc3`、`F1_3`、`Acc5`、`F1_5`

## 消融（每次去掉一个模块）
| Variant | Appearance | CORAL |
|---|---|---|
| OrdProg-Net (full) | ✓ | ✓ |
| w/o appearance | ✗ | ✓ |
| w/o CORAL | ✓ | ✗ |
| w/o appearance & CORAL | ✗ | ✗ |

文件：
- `Paper_Table2_unified.csv`
- `Paper_Table6_leave_one_out.csv`
- `Paper_Table3_spatial_unified.csv`
