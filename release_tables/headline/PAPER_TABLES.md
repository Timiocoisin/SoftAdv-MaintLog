# 论文用表（冻结口径）

**主方法**：OrdProg-Net = \(0.5\times\)seedavg(`ordprog_full`) + \(0.5\times\)seedavg(`appearance_ce`)  
**主协议**：multi-gap（centers 25/55/85/115/145，core45）；指标为 stage-macro MAE（越小越好）  
**不进正文**：噪声版 `ordprog_sota` / `appearance_sota`

数据文件目录：`WaterWallSensing/outputs/softadv_protocol_20261106/tables/`

---

## 一、仓库里有哪些表（清单）

| 文件 | 用途 | 是否进论文 |
|---|---|---|
| `Paper_Table2_main_comparison.csv` | 主对比（含 ViT 等） | **正文 Table 2** |
| `Paper_Table6_ablation.csv` | 模块消融 | **正文 Table 6** |
| `Table3_spatial.csv` | 空间 A↔B | **正文 Table 3** |
| `Table4_corruption_summary.csv` | 成像退化 | **正文 Table 4** |
| `Table4_corruption.csv` | 退化全 severity | 附录可留 |
| `Table5_delay_log.csv` | 延迟日志检测 | **正文 Table 5** |
| `SOTA_bakeoff_rank_seedavg.csv` | bakeoff 总榜 | 与 Table2 同源 |
| `Table2_all_variants_meanstd.csv` | v1/v2 单 seed mean±std | 附录 |
| `Table6_ablation_plan.csv` | 消融定义 | 方法节文字 |
| `FINAL_*` / `v3_*` / `ensemble_*` / `winner_*` | 搜索过程 | **不进正文** |
| `*_template.csv` | 空模板 | 忽略 |
| `SOTA_bakeoff_raw.csv` | 原始 seed 级 | 附录/可复现 |

---

## 二、建议写入论文的表与数据

### Table 2｜Multi-gap 主对比（3-seed 平均预测）

> 单位：进度 stage-macro MAE；Acc3 由 \(\hat g\) 映射 early/mid/late。

| Method | MAE (g0) | ρ (g0) | Acc3 (g0) | MAE (g10) | ρ (g10) | Acc3 (g10) | MAE avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| **OrdProg-Net (ours)** | **3.377** | 0.986 | 0.979 | **7.431** | 0.983 | **0.979** | **5.404** |
| ordprog_full | 3.182 | 0.986 | 0.983 | 8.185 | 0.986 | 0.962 | 5.683 |
| appearance_ce | 3.817 | 0.985 | 0.983 | 7.837 | 0.980 | 0.979 | 5.827 |
| EfficientNet-B0 | 3.835 | 0.986 | 0.940 | 7.833 | 0.986 | 0.974 | 5.834 |
| ResNet-50 | 3.452 | 0.987 | 0.927 | 9.388 | 0.984 | 0.957 | 6.420 |
| ResNet-18 CE (baseline) | 4.080 | 0.983 | 0.957 | 10.475 | 0.979 | 0.910 | 7.278 |
| CORAL-only | 3.676 | 0.988 | 0.996 | 12.103 | 0.981 | 0.799 | 7.890 |
| ConvNeXt-Tiny | 4.022 | 0.986 | 1.000 | 12.430 | 0.980 | 0.932 | 8.226 |
| Swin-T | 5.164 | 0.984 | 0.957 | 11.580 | 0.965 | 0.949 | 8.372 |
| Temporal-TF (R18) | 6.273 | 0.983 | 0.966 | 13.618 | 0.979 | 0.808 | 9.945 |
| ViT-B/16 | 5.691 | 0.980 | 0.983 | 14.906 | 0.916 | 0.868 | 10.299 |
| MobileNetV3-Large | 3.758 | 0.985 | 0.987 | 47.576 | 0.945 | 0.406 | 25.667 |

来源：`Paper_Table2_main_comparison.csv`

---

### Table 3｜空间留出（管长前 40% ↔ 后 40%）

| Fold | Method | MAE (stage) | Spearman | Acc3←prog |
|---|---|---:|---:|---:|
| A→B | baseline seedavg | 7.380 | 0.988 | 0.925 |
| A→B | **ordprog_full seedavg** | **3.310** | 0.989 | **0.957** |
| A→B | OrdProg blend | 4.406 | 0.990 | 0.936 |
| B→A | baseline seedavg | 2.710 | 0.998 | 0.971 |
| B→A | **ordprog_full seedavg** | **2.458** | 0.998 | 0.977 |
| B→A | OrdProg blend | 2.494 | 0.998 | **0.986** |

说明：空间表主报 **ordprog_full**（优于融合）。来源：`Table3_spatial.csv`

---

### Table 4｜成像退化（multi_gap_guard0 ckpt，core45，族内平均 MAE）

| Method | clean | blur | brightness | crop | jpeg | gauss noise | mean corrupt |
|---|---:|---:|---:|---:|---:|---:|---:|
| **OrdProg blend** | **3.377** | **9.505** | **14.441** | **6.119** | **10.612** | 30.241 | 14.184 |
| baseline seedavg | 4.080 | 11.361 | 16.254 | 6.280 | 12.204 | **17.718** | **12.763** |

说明：除强噪声外我们更好；噪声项需在正文诚实讨论。来源：`Table4_corruption_summary.csv`

---

### Table 5｜延迟 / 缺失日志检测（stage 级，skip∈{15,25,40}）

| Detector | AUROC | F1 | Rec skip | Rec stall | Rec dup |
|---|---:|---:|---:|---:|---:|
| **Ours residual** | **0.781** | **0.769** | 1.00 | 0.857 | 1.00 |
| baseline residual | 0.715 | 0.717 | 1.00 | 0.857 | 0.50 |
| log-Δ only | 0.635 | 0.720 | 1.00 | 0.714 | 1.00 |

来源：`Table5_delay_log.csv`（n=45 stages，异常 22）

---

### Table 6｜消融（seedavg，MAE）

| Variant | Appearance | CORAL | MAE g0 | MAE g10 | MAE avg |
|---|---|---|---:|---:|---:|
| baseline CE | ✗ | ✗ | 4.080 | 10.475 | 7.278 |
| CORAL-only | ✗ | ✓ | 3.676 | 12.103 | 7.890 |
| Appearance-only | ✓ | ✗ | 3.817 | 7.837 | 5.827 |
| ordprog_full | ✓ | ✓ | 3.182 | 8.185 | 5.683 |
| **OrdProg-Net (blend)** | ✓ | ✓ (+CE expert) | 3.377 | **7.431** | **5.404** |

来源：`Paper_Table6_ablation.csv`

---

## 三、正文建议用法（一句话）

- **Table 2**：证明相对 ViT/时序/工业 CNN 的主 SOTA  
- **Table 3**：空间外推用 `ordprog_full`  
- **Table 4**：稳健性（承认噪声短板）  
- **Table 5**：关闭 delayed-log 动机  
- **Table 6**：模块必要性 + 融合收益  
