"""CLI ingest 路径验证（纯桩，不碰真实库）：

覆盖 `cmd_ingest` 的完整执行路径——表格渲染 / 附注 / provider 打印 /
锚点词表重建 / 返回码，确保 `_rebuild_anchors` 与 `cmd_ingest` 的作用域互不串味。

回归背景：曾出现 `_rebuild_anchors` 体内混入 `cmd_ingest` 尾部代码（引用
`reports`/`ok`）→ 运行 `cmd_ingest` 必抛 NameError。仅跑 `--help` 测不出，
必须真正调用 `cmd_ingest`。
"""
from __future__ import annotations

import argparse
import io
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cli import cmd_ingest  # noqa: E402
from app.agent import anchors  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


def fake_report(**over) -> dict:
    r = {
        "filename": "policy_demo.md", "status": "ok", "format": "md",
        "version": 1, "chunks_created": 3, "stored": 3, "skipped": 0,
        "red_pages": [], "yellow_pages": [], "elapsed_s": 0.12,
        "degraded": False, "provider": "dashscope",
        "error": None, "vlm_note": None, "warnings": [],
    }
    r.update(over)
    return r


def run_cmd(reports, rebuild=None, exc=None) -> tuple[int, str, list]:
    """用桩替换 run_ingest / build_vocab，执行 cmd_ingest，返回 (码, 输出, 重建调用)。"""
    calls: list = []
    orig_run, orig_build = __import__("app.cli", fromlist=["x"]).run_ingest, anchors.build_vocab
    import app.cli as cli
    cli.run_ingest = lambda paths, rebuild=False: reports
    if exc is None:
        anchors.build_vocab = lambda db, out: (calls.append((str(db), str(out))) or
                                               {"counts": {"union": 1580}, "docs": 38})
    else:
        def _boom(db, out):
            calls.append((str(db), str(out)))
            raise RuntimeError(exc)
        anchors.build_vocab = _boom
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            code = cmd_ingest(argparse.Namespace(paths=["x"], rebuild=False, list_models=False))
    finally:
        cli.run_ingest = orig_run
        anchors.build_vocab = orig_build
    return code, buf.getvalue(), calls


def main() -> int:
    print("=" * 72)
    print("CLI ingest 路径验证（桩，无网络/无落盘）")
    print("=" * 72)

    print("\n[1] 全成功：返回码 0，provider 行与词表重建都在")
    code, out, calls = run_cmd([fake_report()])
    check("返回码 0", code == 0, f"got={code}")
    check("打印 Embedding provider 行", "Embedding provider=dashscope" in out, out[-200:])
    check("调用锚点词表重建恰 1 次", len(calls) == 1, f"calls={calls}")
    check("重建输出路径为 data/kb_anchors.json", calls and calls[0][1].endswith("kb_anchors.json"))
    check("打印重建结果", "锚点词表已重建" in out)
    check("无 NameError 输出", "NameError" not in out and "Traceback" not in out)

    print("\n[2] 有失败文档：返回码 1（失败不中断，仍重建词表）")
    code, out, calls = run_cmd([fake_report(), fake_report(status="failed", error="坏文件")])
    check("返回码 1", code == 1, f"got={code}")
    check("列出失败原因", "坏文件" in out)
    check("仍调用词表重建", len(calls) == 1)

    print("\n[3] degraded：provider 行带降级说明")
    code, out, _ = run_cmd([fake_report(degraded=True, provider="mock")])
    check("提示 degraded", "degraded" in out)

    print("\n[4] 词表重建抛异常：不影响 ingest 返回码（软失败）")
    code, out, _ = run_cmd([fake_report()], exc="磁盘只读")
    check("返回码仍 0", code == 0, f"got={code}")
    check("打印告警且给出后果", "锚点词表重建失败" in out and "一律改写" in out)

    print("\n[5] 空报告列表：不崩，返回码 0")
    code, out, _ = run_cmd([])
    check("返回码 0", code == 0, f"got={code}")
    check("provider 占位为 ?", "Embedding provider=?" in out)

    print("\n" + "=" * 72)
    print(f"结果: {PASS} pass / {FAIL} fail")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
