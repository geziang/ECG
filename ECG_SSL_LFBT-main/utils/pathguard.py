# -*- coding: utf-8 -*-
"""输出路径守卫(预提交安全扫描整改, 2026-09-23)。

多机流水线允许操作者用 CLI 显式指定任意盘符的输出目录(主机A F:\\ / 主机B E:\\),
因此不限制基目录本身; 守卫保证:
1) 基目录显式 expanduser+resolve, 杜绝相对路径歧义;
2) 追加的文件名必须是纯文件名(不含路径分隔符/..), 拼接后不逃逸基目录;
3) 操作者直接指定的完整输出文件 resolve 后不得是已存在的目录。
"""
from pathlib import Path, PurePath


def _plain_name(n) -> str:
    s = str(n)
    if s in ("", ".", "..") or "/" in s or "\\" in s or ".." in s:
        raise ValueError(f"不安全的输出文件名: {n!r}")
    return s


def safe_out_path(base, *names) -> Path:
    """校验 base+固定文件名 的输出路径, 返回 resolve 后的 Path。"""
    p = Path(base).expanduser().resolve()
    for n in names:
        p = p / _plain_name(n)
    return p


def open_out(base, *names, mode="w", **kw):
    """在守卫路径上打开输出文件, 参数透传 open()。"""
    return open(safe_out_path(base, *names), mode, **kw)


def operator_out_file(path) -> Path:
    """校验操作者 CLI 直接指定的完整输出文件路径。"""
    if not isinstance(path, PurePath) and not isinstance(path, str):
        raise TypeError(f"输出路径必须是 str/Path: {type(path)}")
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        raise ValueError(f"输出路径是已存在目录: {p}")
    return p


def open_out_file(path, mode="w", **kw):
    """在守卫的完整输出文件路径上打开文件, 参数透传 open()。"""
    return open(operator_out_file(path), mode, **kw)
