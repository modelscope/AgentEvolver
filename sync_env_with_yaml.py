#!/usr/bin/env python3
"""
同步 conda 环境与 seeupo_env.yaml 中的包版本

功能：
1. 比较 seeupo_git 环境与 yaml 中指定的包版本
2. 列出版本不一致的包（仅针对 yaml 中显式指定的包）
3. 按 yaml 中的版本进行安装/更新

注意：
- seeupo_git 中包的数目 > yaml 中的（因依赖会自动安装）
- 本脚本只针对 yaml 中显式列出的包进行比对和更新
- 其他未在 yaml 中的包保持不动，避免破坏依赖

Conda 内置指令参考：
- conda compare -n seeupo_git /path/to/seeupo_env.yaml   # 比较环境与 yaml
- conda env update -n seeupo_git -f seeupo_env.yaml     # 用 yaml 更新环境（不删包，不加 --prune）
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Default yaml next to this script; legacy internal example (placeholder): `<INTERNAL_REPO_PATH>/SeeUPOGit/seeupo_env.yaml`
_DEFAULT_SEEUPO_YAML = str(Path(__file__).resolve().parent / "seeupo_env.yaml")

# Conda 内置指令
CONDA_COMPARE = "conda compare"
CONDA_INSTALL = "conda install"
PIP_INSTALL = "pip install"


def run_cmd(cmd: list[str], capture: bool = True) -> tuple[int, str]:
    """执行命令，返回 (exit_code, output)"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=False,
        )
        if capture:
            out = (result.stdout or "") + (result.stderr or "")
        else:
            out = ""
        return result.returncode, out
    except Exception as e:
        return -1, str(e)


def parse_yaml_packages(yaml_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    解析 yaml 中的 conda 和 pip 包。
    返回 (conda_pkgs, pip_pkgs)，每个是 {包名: 版本规范}
    conda: 如 "numpy=1.26.4" 或 "numpy=1.26.4=h1234567_1"
    pip: 如 "numpy==1.26.4"
    """
    conda_pkgs = {}
    pip_pkgs = {}
    in_pip = False
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML 文件不存在: {yaml_path}")

    with open(yaml_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "pip:" or stripped == "- pip:":
                in_pip = True
                continue
            if in_pip:
                if stripped.startswith("- "):
                    stripped = stripped[2:]
                # pip 格式: package==version
                if "==" in stripped:
                    pkg, ver = stripped.split("==", 1)
                    pip_pkgs[pkg.strip().lower()] = ver.strip()
                elif stripped and not stripped.startswith("#"):
                    # 兼容其他格式
                    m = re.match(r"([a-zA-Z0-9_-]+)\s*[=<>!]+\s*(.+)", stripped)
                    if m:
                        pip_pkgs[m.group(1).lower()] = m.group(2).strip()
            else:
                # conda 格式: - package=version 或 - package=version=build
                if stripped.startswith("- ") and "=" in stripped:
                    spec = stripped[2:].strip()
                    # 取第一个 = 前的作为包名（conda 包名不含=）
                    parts = spec.split("=")
                    if len(parts) >= 2:
                        pkg = parts[0]
                        # 跳过纯构建/渠道的条目（如 main, gnu）
                        if pkg and not pkg.startswith("_") or pkg.startswith("_"):
                            conda_pkgs[pkg] = "=".join(parts[1:])

    return conda_pkgs, pip_pkgs


def get_installed_pip(env_name: str) -> dict[str, str]:
    """获取环境中已安装的 pip 包及版本"""
    code, out = run_cmd(
        ["conda", "run", "-n", env_name, "pip", "list", "--format=json"]
    )
    if code != 0:
        # 尝试直接 pip list
        code2, out2 = run_cmd(
            [sys.executable, "-m", "pip", "list", "--format=json"]
        )
        if code2 == 0:
            out = out2
        else:
            return {}
    try:
        data = json.loads(out)
        return {p["name"].lower(): p["version"] for p in data}
    except Exception:
        return {}


def compare_and_report(
    env_name: str,
    yaml_path: str,
    conda_pkgs: dict,
    pip_pkgs: dict,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """
    比较环境与 yaml，返回需要更新的包。
    conda_mismatches: [(pkg, yaml_ver, installed_ver), ...]
    pip_mismatches: [(pkg, yaml_ver, installed_ver), ...]
    """
    conda_mismatches = []
    pip_mismatches = []

    # 1. Conda 比较：使用 conda compare
    code, out = run_cmd(
        ["conda", "compare", "-n", env_name, yaml_path, "--json"]
    )
    if code == 0:
        # 完全一致，无 conda 不匹配
        pass
    else:
        try:
            data = json.loads(out)
            # 文档中 mismatch 的结构可能不同，根据实际解析
            for item in data.get("packages", []):
                if "mismatch" in str(item).lower() or "version" in str(item):
                    # 解析 conda compare 的 json 输出
                    pass
        except Exception:
            pass
        # conda compare 的非 0 退出表示有不匹配，但 json 结构可能因版本而异
        # 我们同时用显式比对作为补充
        code2, out2 = run_cmd(
            ["conda", "list", "-n", env_name, "--json"]
        )
        if code2 == 0:
            try:
                conda_list = json.loads(out2)
                for p in conda_list:
                    name = p.get("name", "")
                    if name in conda_pkgs:
                        installed = p.get("version", "") + (
                            "=" + p.get("build", "") if p.get("build") else ""
                        )
                        yaml_spec = conda_pkgs[name]
                        # 简化比较：去掉 build 部分
                        yaml_ver = yaml_spec.split("=")[0] if "=" in yaml_spec else yaml_spec
                        inst_ver = p.get("version", "")
                        if yaml_ver != inst_ver:
                            conda_mismatches.append((name, yaml_spec, installed))
            except Exception:
                pass

    # 2. Pip 比较
    installed_pip = get_installed_pip(env_name)
    for pkg, yaml_ver in pip_pkgs.items():
        inst = installed_pip.get(pkg)
        if inst is None:
            pip_mismatches.append((pkg, yaml_ver, "(未安装)"))
        elif inst != yaml_ver:
            pip_mismatches.append((pkg, yaml_ver, inst))

    return conda_mismatches, pip_mismatches


def main():
    parser = argparse.ArgumentParser(
        description="比较并同步 conda 环境与 seeupo_env.yaml 中的包版本"
    )
    parser.add_argument(
        "yaml_path",
        nargs="?",
        default=_DEFAULT_SEEUPO_YAML,
        help="seeupo_env.yaml 路径（默认: 与本脚本同目录；旧内部路径占位见脚本顶部注释）",
    )
    parser.add_argument(
        "-n", "--env",
        default="seeupo_git",
        help="conda 环境名",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="仅比较并报告，不执行安装",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="对版本不一致的包执行安装（按 yaml 版本）",
    )
    parser.add_argument(
        "--pip-only",
        action="store_true",
        help="仅处理 pip 包（conda compare 可能因平台/build 差异报错，pip 更可控）",
    )
    parser.add_argument(
        "--conda-only",
        action="store_true",
        help="仅处理 conda 包",
    )
    parser.add_argument(
        "--env-update",
        action="store_true",
        help="使用 conda env update 整体更新（会更新 yaml 中所有包，可能触发依赖重算）",
    )

    args = parser.parse_args()
    yaml_path = Path(args.yaml_path).expanduser().resolve()
    if not yaml_path.exists():
        # 尝试相对路径
        rel = Path(__file__).parent / "seeupo_env.yaml"
        if rel.exists():
            yaml_path = rel
        else:
            print(f"错误: 找不到 YAML 文件: {args.yaml_path}")
            sys.exit(1)

    conda_pkgs, pip_pkgs = parse_yaml_packages(str(yaml_path))
    print(f"YAML 中 conda 包: {len(conda_pkgs)} 个, pip 包: {len(pip_pkgs)} 个")
    print()

    # 可选：使用 conda env update 整体更新
    if args.env_update:
        print(">>> conda env update -n {} -f {}".format(args.env, yaml_path))
        code, out = run_cmd(
            ["conda", "env", "update", "-n", args.env, "-f", str(yaml_path)],
            capture=False,
        )
        sys.exit(0 if code == 0 else 1)

    # 先尝试 conda compare 总览
    if not args.pip_only:
        code, out = run_cmd(
            ["conda", "compare", "-n", args.env, str(yaml_path)]
        )
        print("=== conda compare 输出 ===")
        print(out)
        print()

    conda_mismatches, pip_mismatches = compare_and_report(
        args.env, str(yaml_path), conda_pkgs, pip_pkgs
    )

    print("=== 版本不一致的包 ===")
    if not args.pip_only and conda_mismatches:
        print("\n【Conda 包】")
        for pkg, yaml_ver, inst_ver in conda_mismatches:
            print(f"  {pkg}: yaml={yaml_ver}  vs  已安装={inst_ver}")
    if not args.conda_only and pip_mismatches:
        print("\n【Pip 包】")
        for pkg, yaml_ver, inst_ver in pip_mismatches:
            print(f"  {pkg}: yaml={yaml_ver}  vs  已安装={inst_ver}")

    if not conda_mismatches and not pip_mismatches:
        print("所有 yaml 中指定的包版本已一致。")
        return

    if args.compare_only:
        print("\n(使用 --install 可执行安装)")
        return

    if not args.install:
        print("\n(使用 --install 可执行安装)")
        return

    # 执行安装
    failed = []
    if not args.pip_only and conda_mismatches:
        for pkg, yaml_spec, _ in conda_mismatches:
            spec = f"{pkg}={yaml_spec}"
            print(f"\n>>> conda install -n {args.env} -y {spec}")
            code, out = run_cmd(
                ["conda", "install", "-n", args.env, "-y", spec],
                capture=False,
            )
            if code != 0:
                failed.append(("conda", pkg, out))

    if not args.conda_only and pip_mismatches:
        specs = [f"{pkg}=={ver}" for pkg, ver, _ in pip_mismatches]
        cmd = ["conda", "run", "-n", args.env, "pip", "install"] + specs
        print(f"\n>>> {' '.join(cmd)}")
        code, out = run_cmd(cmd, capture=False)
        if code != 0:
            # 尝试逐个安装，便于定位问题包
            print("\n批量安装失败，尝试逐个安装...")
            for pkg, ver, _ in pip_mismatches:
                code2, out2 = run_cmd(
                    ["conda", "run", "-n", args.env, "pip", "install", f"{pkg}=={ver}"],
                    capture=False,
                )
                if code2 != 0:
                    failed.append(("pip", pkg, out2))

    if failed:
        print("\n安装失败的包:")
        for pkg_type, pkg, err in failed:
            print(f"  [{pkg_type}] {pkg}: {err[:200]}...")
        sys.exit(1)


if __name__ == "__main__":
    main()
