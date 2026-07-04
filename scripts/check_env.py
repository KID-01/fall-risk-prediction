"""
环境检查脚本
运行方式: python scripts/check_env.py
检查所有核心依赖是否正确安装，输出详细报告
"""
import sys


def check_python_version() -> bool:
    """检查Python版本 >= 3.10"""
    version = sys.version_info
    ok = version >= (3, 10)
    status = "✓" if ok else "✗"
    print(f"  [{status}] Python版本: {version.major}.{version.minor}.{version.micro} (需要 >= 3.10)")
    return ok


def check_package(name: str, import_name: str | None = None) -> bool:
    """检查单个包是否可导入"""
    if import_name is None:
        import_name = name
    try:
        pkg = __import__(import_name)
        version = getattr(pkg, "__version__", "已安装")
        print(f"  [✓] {name} ({version})")
        return True
    except ImportError:
        print(f"  [✗] {name} — 未安装！请运行: pip install {name}")
        return False


def check_cuda() -> bool:
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  [✓] CUDA可用: {device_count}个GPU — {device_name}")
            return True
        else:
            print(f"  [⚠] CUDA不可用，将使用CPU模式（推理速度会较慢）")
            return False
    except ImportError:
        print(f"  [✗] PyTorch未安装，无法检查CUDA")
        return False


def main():
    print("=" * 50)
    print("  跌倒风险预测系统 — 环境检查")
    print("=" * 50)
    print()

    results = []

    # 1. Python版本
    print("[1/5] Python版本")
    results.append(check_python_version())
    print()

    # 2. 深度学习核心
    print("[2/5] 深度学习核心")
    results.append(check_package("torch"))
    results.append(check_package("torchvision"))
    results.append(check_package("onnx"))
    results.append(check_package("onnxruntime"))
    print()

    # 3. 计算机视觉
    print("[3/5] 计算机视觉")
    results.append(check_package("cv2", "cv2"))
    results.append(check_package("numpy"))
    results.append(check_package("scipy"))
    print()

    # 4. 数据处理与后端
    print("[4/5] 数据处理 & 后端")
    results.append(check_package("pandas"))
    results.append(check_package("sklearn", "sklearn"))
    results.append(check_package("fastapi"))
    results.append(check_package("uvicorn"))
    results.append(check_package("omegaconf"))
    results.append(check_package("loguru"))
    print()

    # 5. CUDA
    print("[5/5] GPU环境")
    cuda_ok = check_cuda()
    results.append(cuda_ok)  # CUDA不可用不算失败
    print()

    # 总结
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"  ✓ 全部通过！({passed}/{total}) 环境就绪，可以开始开发")
    else:
        failed = total - passed
        print(f"  ⚠ {passed}/{total} 通过，{failed} 项需要修复")
        print(f"  运行 pip install -e . 安装缺失的依赖")
    print("=" * 50)


if __name__ == "__main__":
    main()