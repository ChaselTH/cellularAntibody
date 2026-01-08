# cellularAntibody

一个基于 Tkinter 的细胞/病毒/抗体模拟小程序，展示随机运动、感染爆发与细胞分裂成长的动态。

## 运行

```bash
python main.py
```

## 可调参数（`main.py` 顶部）

细胞相关（分裂与成长）：
- `CELL_DIVIDE_TIME_MIN` / `CELL_DIVIDE_TIME_MAX`：大细胞分裂的随机时间范围（秒）。
- `CELL_GROW_TIME`：小细胞成长为大细胞的时间（秒）。
- `CELL_R_SMALL` / `CELL_R_LARGE`：小/大细胞半径。

感染爆发：
- `INFECTION_PADDING`：病毒贴到细胞的判定补偿（越大越容易感染）。
- `BURST_VIRUS_COUNT_SMALL` / `BURST_VIRUS_COUNT_LARGE`：小/大细胞爆发释放的病毒数量。

运动与数量：
- `CELL_SPEED` / `VIRUS_SPEED` / `AB_SPEED`：运动速度。
- `N_CELLS` / `N_VIRUSES` / `N_ANTIBODIES`：初始数量。
