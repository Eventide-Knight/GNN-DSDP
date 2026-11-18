#!/usr/bin/env python
"""paths.py -- 唯一的路径配置入口。

本仓库不写死任何人的绝对路径：所有脚本都从这里取路径。
克隆到任意机器后，默认即可运行；若数据放在别处，用环境变量覆盖（见下）。

环境变量（都可选，不设就用默认值）
------------------------------------
GNNSITE_HOME   代码与数据根目录。默认 = 本文件所在目录
DSDP_HOME      DSDP 仓库根（含 test_dataset/ 与两个 GPU 二进制）。
               默认 = 与仓库同级的 ../DSDP
SPLIT_DIR      EquiBind time-split 编号文件所在目录（timesplit_no_lig_overlap_train 等）。
               默认 = $GNNSITE_HOME/splits
PDBBIND_DIR    PDBBind v2020 解包目录（应含 refined-set/ 与 v2020-other-PL/）。
               默认 = $GNNSITE_HOME/pdbbind_v2020
NODECODER_DIR  NodeCoder 源码目录（仅作参考，不参与运行）。默认 = 同级 ../Nodecoder
MKDSSP         mkdssp 可执行文件。默认从 PATH 中查找
REDOCK_BIN     DSDP 常规对接二进制。默认 = $DSDP_HOME/DSDP_redocking/DSDP
BLIND_BIN      DSDP 盲对接二进制。默认 = $DSDP_HOME/DSDP_blind_docking/DSDP

示例
----
    export DSDP_HOME=$HOME/code/DSDP
    export PDBBIND_DIR=/data/PDBBind_v2020
    export SPLIT_DIR=/data/equibind_split
    python paths.py            # 打印当前解析结果与存在性
"""
import os
import shutil

HOME_DIR = os.path.dirname(os.path.abspath(__file__))


def _p(env_name, default):
    """环境变量优先，其次默认值；统一展开 ~ 和相对路径。"""
    val = os.environ.get(env_name)
    if val:
        return os.path.abspath(os.path.expanduser(val))
    return os.path.abspath(os.path.expanduser(default))


GNNSITE_HOME = _p("GNNSITE_HOME", HOME_DIR)
DSDP_HOME = _p("DSDP_HOME", os.path.join(os.path.dirname(HOME_DIR), "DSDP"))
SPLIT_DIR = _p("SPLIT_DIR", os.path.join(GNNSITE_HOME, "splits"))
PDBBIND_DIR = _p("PDBBIND_DIR", os.path.join(GNNSITE_HOME, "pdbbind_v2020"))
NODECODER_DIR = _p("NODECODER_DIR", os.path.join(os.path.dirname(HOME_DIR), "Nodecoder"))

# 派生路径
DATA_DIR = os.path.join(GNNSITE_HOME, "data")
INDEX_JSON = os.path.join(GNNSITE_HOME, "structure_index.json")
SKIPPED = os.path.join(GNNSITE_HOME, "skipped.txt")
REPORTS_DIR = os.path.join(GNNSITE_HOME, "reports")
RESULTS_DIR = os.path.join(GNNSITE_HOME, "results")
RUNS_DIR = os.path.join(GNNSITE_HOME, "runs")

TEST_DIR = os.path.join(DSDP_HOME, "test_dataset")
TEST_PDBBIND_DIR = os.path.join(TEST_DIR, "PDBBind_time_split_dataset")
DSDP_DATASET_DIR = os.path.join(TEST_DIR, "DSDP_dataset")
DUD_E_DIR = os.path.join(TEST_DIR, "DUD_E_dataset")

REDOCK_BIN = _p("REDOCK_BIN", os.path.join(DSDP_HOME, "DSDP_redocking", "DSDP"))
BLIND_BIN = _p("BLIND_BIN", os.path.join(DSDP_HOME, "DSDP_blind_docking", "DSDP"))
MKDSSP = os.environ.get("MKDSSP") or shutil.which("mkdssp") or "mkdssp"

# split 编号文件的固定文件名（来自 EquiBind 仓库 data/）
SPLIT_FILES = {
    "train": "timesplit_no_lig_overlap_train",
    "val": "timesplit_no_lig_overlap_val",
    "test": "timesplit_test",
}


def split_file(split):
    """返回某个 split 的编号文件绝对路径。"""
    if split not in SPLIT_FILES:
        raise KeyError("split must be one of %s" % sorted(SPLIT_FILES))
    return os.path.join(SPLIT_DIR, SPLIT_FILES[split])


def summary():
    rows = [
        ("GNNSITE_HOME", GNNSITE_HOME),
        ("DSDP_HOME", DSDP_HOME),
        ("SPLIT_DIR", SPLIT_DIR),
        ("PDBBIND_DIR", PDBBIND_DIR),
        ("DATA_DIR", DATA_DIR),
        ("TEST_PDBBIND_DIR", TEST_PDBBIND_DIR),
        ("DUD_E_DIR", DUD_E_DIR),
        ("REDOCK_BIN", REDOCK_BIN),
        ("BLIND_BIN", BLIND_BIN),
        ("MKDSSP", MKDSSP),
    ]
    print("%-18s %-6s %s" % ("name", "exists", "path"))
    for name, p in rows:
        exists = os.path.exists(p) if os.path.isabs(p) else bool(shutil.which(p))
        print("%-18s %-6s %s" % (name, "YES" if exists else "NO", p))
    for split in ("train", "val", "test"):
        f = split_file(split)
        n = sum(1 for _ in open(f)) if os.path.exists(f) else -1
        print("%-18s %-6s %s (%d lines)" % ("split:" + split, "YES" if n >= 0 else "NO", f, n))

    if not (os.path.isabs(MKDSSP) and os.path.exists(MKDSSP)) and not shutil.which(MKDSSP):
        print("\n[HINT] mkdssp 没找到：DSSP 相关 15 维特征会变成全 0，预测结果会明显变差。")
        print("       修复：conda install -c conda-forge dssp   或   export MKDSSP=/path/to/mkdssp")
    if not os.path.exists(PDBBIND_DIR):
        print("\n[HINT] PDBBIND_DIR 不存在：请按 README 第 3.1 节下载 PDBBind v2020 并解包。")
    if not os.path.exists(split_file("train")):
        print("[HINT] SPLIT_DIR 里缺切分文件：请按 README 第 3.2 节从 EquiBind 仓库拷三个文件。")
    if not os.path.isdir(DSDP_HOME):
        print("[HINT] DSDP_HOME 不存在：请按 README 第 3.3 节克隆并编译 DSDP。")


if __name__ == "__main__":
    summary()
