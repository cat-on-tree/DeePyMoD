import os
import json
from datetime import datetime


def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def init_report_run(project_root="..", run_name=None):
    """
    初始化一次报告运行目录。
    返回:
        dict with keys:
          run_id, run_dir, figures_dir, tables_dir, metrics_dir, artifacts_dir, report_md, manifest_json
    """
    project_root = os.path.abspath(project_root)
    base_dir = os.path.join(project_root, "reports")
    os.makedirs(base_dir, exist_ok=True)

    run_id = run_name if run_name else f"run_{_now_tag()}"
    run_dir = os.path.join(base_dir, run_id)

    figures_dir = os.path.join(run_dir, "figures")
    tables_dir = os.path.join(run_dir, "tables")
    metrics_dir = os.path.join(run_dir, "metrics")
    artifacts_dir = os.path.join(run_dir, "artifacts")

    for d in [run_dir, figures_dir, tables_dir, metrics_dir, artifacts_dir]:
        os.makedirs(d, exist_ok=True)

    report_md = os.path.join(run_dir, "report.md")
    manifest_json = os.path.join(run_dir, "manifest.json")

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "project_root": project_root,
        "paths": {
            "run_dir": run_dir,
            "figures_dir": figures_dir,
            "tables_dir": tables_dir,
            "metrics_dir": metrics_dir,
            "artifacts_dir": artifacts_dir,
            "report_md": report_md,
            "manifest_json": manifest_json,
        },
        "status": {
            "step1_initialized": True
        }
    }

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def write_report_skeleton(manifest: dict):
    """
    写入报告 Markdown 骨架（后续步骤会逐段填充）
    """
    report_md = manifest["paths"]["report_md"]
    run_id = manifest["run_id"]
    created_at = manifest["created_at"]

    md = f"""# PK/PD 全流程自动化报告

- **Run ID**: `{run_id}`
- **生成时间**: `{created_at}`

---

## 1. 研究目标与已知机制

### 1.1 研究目标
（待填充）

### 1.2 已知 PK 方程
（待填充）

### 1.3 已知 PD 方程
（待填充）

---

## 2. 数据概览与可视化

### 2.1 数据来源与统计
（待填充）

### 2.2 PK/PD 散点图
（待填充图：`figures/pkpd_scatter.png`）

---

## 3. 结构发现（DeepMoD）

### 3.1 候选项库
（待填充）

### 3.2 剪枝与 Top-K 结构
（待填充）

---

## 4. 多起点精调

### 4.1 多起点配置
（待填充）

### 4.2 各候选最优参数与初值
（待填充表：`tables/multistart_summary.csv`）

---

## 5. 群体验证与模型比较

### 5.1 拟合优度指标
（待填充表：`tables/model_compare.csv`）

### 5.2 模型排序结论
（待填充）

---

## 6. 诊断图

### 6.1 GOF
（待填充图）

### 6.2 残差/RSE
（待填充图）

### 6.3 Bootstrap
（待填充图）

### 6.4 VPC
（待填充图）

---

## 7. 方程复原与可解释性

### 7.1 候选模型方程复原
（待填充）

### 7.2 最优模型解释
（待填充）

---

## 8. 结论与下一步

（待填充）

---

## 附录：运行清单

- 目录: `./`
- 图目录: `./figures`
- 表目录: `./tables`
- 指标目录: `./metrics`
- 清单文件: `./manifest.json`
"""

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(md)

    return report_md


def init_and_create_skeleton(project_root="..", run_name=None):
    manifest = init_report_run(project_root=project_root, run_name=run_name)
    report_md = write_report_skeleton(manifest)
    return manifest, report_md