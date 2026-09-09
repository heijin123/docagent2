"""M6 评估体系（需求 F7）：golden 集 + recall@5/MRR + 引用可回查率。

模块划分：
- golden.py   golden 加载/校验 + 锚句归一化子串定位期望块（零人工标注块 id）
- metrics.py  检索层 recall@5 / MRR；答案层引用可回查率
- runner.py   编排：加载→定位→检索/答案→指标→门槛判定→报告落盘
"""
from .golden import GoldenCase, load_golden, locate_expected_chunks, normalize
from .metrics import compute_answer_metrics, compute_retrieval_metrics
from .runner import EvalRunner, run_eval

__all__ = [
    "GoldenCase", "load_golden", "locate_expected_chunks",
    "compute_retrieval_metrics", "compute_answer_metrics", "normalize",
    "EvalRunner", "run_eval",
]
