"""
Build pinned Taichi for Linux ARM64 and register a verified wheel in the runtime cache.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

COMMIT = "b4b956fdaa67fd3e198527fd607053ecb9bf4aa2"
REPO = "https://github.com/taichi-dev/taichi.git"
BUILD_REQUIREMENTS = (
    "setuptools==80.10.2", "wheel==0.48.0", "scikit-build==0.18.1",
    "cmake==3.31.10", "ninja==1.13.2", "pybind11==2.13.6", "numpy==2.5.4",
)


def run(command: Sequence[str | Path], *, cwd: Path | None = None,
        env: dict[str, str] | None = None, log: Path | None = None,
        timeout: int = 600) -> None:
    """
    Run one bounded build command and optionally retain combined diagnostic output.
    """
    print("Running:", " ".join(map(str, command)), flush=True)
    if log:
        print(f"Log: {log}", flush=True)
        with log.open("w") as stream:
            subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
    else:
        subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True, timeout=timeout)


def llvm_environment(work: Path, env: dict[str, str],
                     prefix: Path | None = None) -> dict[str, str]:
    """
    Use matching LLVM/Clang 15; Ubuntu can extract its native packages without sudo.
    """
    if prefix is None and Path("/usr/lib/llvm-15/bin/clang").is_file():
        prefix = Path("/usr/lib/llvm-15")
    if prefix is None:
        release = platform.freedesktop_os_release()
        if (release.get("ID"), release.get("VERSION_ID")) != ("ubuntu", "24.04"):
            raise RuntimeError("Install LLVM/Clang 15 or pass --llvm-dir to its installation prefix")
        packages = (
            "llvm-15-dev", "llvm-15", "llvm-15-runtime", "llvm-15-linker-tools", "llvm-15-tools",
            "libllvm15t64", "clang-15", "libclang-cpp15t64", "libclang1-15t64",
            "libclang-common-15-dev",
        )
        debs = work / "llvm-debs"
        root = work / "llvm-root"
        debs.mkdir(exist_ok=True)
        root.mkdir(exist_ok=True)
        versions = []
        for name in packages:
            policy = subprocess.check_output(["apt-cache", "policy", name], text=True,
                                             env={**env, "LC_ALL": "C"})
            version = next(line.split(":", 1)[1].strip() for line in policy.splitlines()
                           if line.strip().startswith("Candidate:"))
            if version == "(none)":
                raise RuntimeError(f"No APT candidate for {name}; supply --llvm-dir")
            versions.append(f"{name}={version}")
        (work / "llvm-packages.txt").write_text("\n".join(versions) + "\n")
        run(["apt-get", "download", *versions], cwd=debs, log=work / "llvm-download.log")
        for deb in sorted(debs.glob("*.deb")):
            run(["dpkg-deb", "-x", deb, root])
        prefix = root / "usr/lib/llvm-15"
    prefix = prefix.resolve()
    libraries = [str(prefix / "lib"), str(prefix.parent / "aarch64-linux-gnu")]
    if env.get("LD_LIBRARY_PATH"):
        libraries.append(env["LD_LIBRARY_PATH"])
    env = {**env, "LD_LIBRARY_PATH": os.pathsep.join(libraries),
           "CC": str(prefix / "bin/clang"), "CXX": str(prefix / "bin/clang++"),
           "LLVM_DIR": str(prefix / "lib/cmake/llvm")}
    version = subprocess.check_output([str(prefix / "bin/llvm-config"), "--version"],
                                      env=env, text=True).strip()
    if not version.startswith("15."):
        raise RuntimeError(f"Taichi 1.7.4 needs LLVM 15, found {version}")
    run([env["CC"], "--version"], env=env)
    return env


def patch_source(source: Path) -> str:
    """
    Apply only the recorded AArch64 runtime-bitcode fix, permitting clean incremental rebuilds.
    """
    patch = Path(__file__).resolve().parents[1] / "assets/arm64-no-outline-atomics.patch"
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", ".", ":(exclude)external"],
                                   cwd=source, text=True)
    # Hash abbreviation depends on the caller's Git configuration; compare patch content.
    def content(text: str) -> list[str]:
        """
        Ignore Git's configurable object-name abbreviation when comparing patch text.
        """
        lines = [line for line in text.splitlines() if not line.startswith("index ")]
        return lines

    if content(diff) != content(patch.read_text()):
        if diff.strip():
            raise RuntimeError("Source has unrelated tracked edits; use a clean checkout")
        run(["git", "apply", "--check", patch], cwd=source)
        run(["git", "apply", patch], cwd=source)
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    return digest


def verification_ok(path: Path) -> bool:
    """
    Treat missing or malformed replies as failed verification.
    """
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    verified = isinstance(result, dict) and result.get("ok") is True
    return verified


def verify_backends(python: Path, work: Path) -> tuple[list[str], str | None]:
    """
    Require CPU correctness; a missing or crashing GPU must not prevent CPU registration.
    """
    verifier = Path(__file__).with_name("verify_taichi.py").absolute()
    # No build-time loader overrides in runtime verification: prove the wheel stands alone.
    output = work / "verify-cpu.json"
    output.unlink(missing_ok=True)
    run([python, verifier, "--arch", "cpu", "--output", output],
        log=work / "verify-cpu.log", timeout=120)
    if not verification_ok(output):
        raise RuntimeError("CPU fallback verification failed")
    verified = ["cpu"]
    output = work / "verify-vulkan.json"
    output.unlink(missing_ok=True)
    vulkan_completed = False
    try:
        run([python, verifier, "--arch", "vulkan", "--output", output],
            log=work / "verify-vulkan.log", timeout=120)
        vulkan_completed = True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print("Vulkan unavailable; retaining the verified CPU runtime. See verify-vulkan.log.")
    log = (work / "verify-vulkan.log").read_text()
    devices = re.findall(r'Vulkan Device "([^"]+)" supports Vulkan', log)
    hardware = bool(devices) and not any(word in devices[-1].lower()
                                        for word in ("llvmpipe", "lavapipe", "swiftshader", "software"))
    if vulkan_completed and verification_ok(output) and hardware:
        verified.append("vulkan")
    device = devices[-1] if "vulkan" in verified else None
    result = verified, device
    return result


def main() -> None:
    """
    Build the pinned native wheel, require CPU verification and optionally register it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, help="Target venv Python; defaults to PROJECT/.venv/bin/python")
    parser.add_argument("--build-dir", type=Path, help="Defaults to PROJECT/diagnostics/taichi-cpu-vulkan-build")
    parser.add_argument("--llvm-dir", type=Path, help="LLVM/Clang 15 prefix containing bin and lib/cmake/llvm")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-install", action="store_true", help="Build and verify without registering a runtime")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        parser.error("This workflow targets native Linux ARM64 builds")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    project = args.project.resolve()
    # absolute() preserves venv identity; resolve() would follow its symlink to system Python.
    python = (args.python or project / ".venv/bin/python").absolute()
    facts = json.loads(subprocess.check_output([str(python), "-c",
        ("import json,sys,platform; print(json.dumps(dict(version=list(sys.version_info[:2]),"
         "prefix=sys.prefix,base=sys.base_prefix,machine=platform.machine(),"
         "impl=platform.python_implementation())))")], text=True))
    if (facts["version"] not in ([3, 12], [3, 13]) or facts["prefix"] == facts["base"]
            or facts["impl"] != "CPython" or facts["machine"] not in ("aarch64", "arm64")):
        parser.error("Target must be a native ARM64 CPython 3.12/3.13 virtual environment")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv must be installed and on PATH")
    work = (args.build_dir or project / "diagnostics/taichi-cpu-vulkan-build").resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = llvm_environment(work, dict(os.environ), args.llvm_dir)
    source = work / "source"
    if not source.exists():
        run(["git", "clone", "--branch", "v1.7.4", "--depth", "1", REPO, source])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != COMMIT:
        raise RuntimeError(f"Source must be {COMMIT}; found {head}. Use a fresh build directory.")
    patch_hash = patch_source(source)
    # Official historical Eigen mirror contains the same pinned commit; GitLab git can stall.
    run(["git", "config", "submodule.external/eigen.url",
         "https://github.com/eigenteam/eigen-git-mirror.git"], cwd=source)
    run(["git", "-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=30",
         "submodule", "update", "--init", "--recursive", "--depth", "1", "--jobs", "8"],
        cwd=source, log=work / "submodules.log", timeout=1800)
    status = subprocess.check_output(["git", "submodule", "status", "--recursive"], cwd=source, text=True)
    if any(line[0] != " " for line in status.splitlines()):
        raise RuntimeError("Submodules do not match pinned commits")
    (work / "submodules.txt").write_text(status)
    build_python = work / "build-env/bin/python"
    if not build_python.exists():
        run([uv, "venv", "--python", python, work / "build-env"])
    run([uv, "pip", "install", "--python", build_python, *BUILD_REQUIREMENTS])
    env["PATH"] = str(build_python.parent) + os.pathsep + env.get("PATH", "")
    flags = ["-DTI_WITH_VULKAN=ON", "-DTI_WITH_OPENGL=OFF", "-DTI_WITH_METAL=OFF",
             "-DTI_WITH_C_API=OFF", "-DTI_BUILD_TESTS=OFF",
             "-DTI_WITH_LLVM=ON",
             "-DTI_WITH_CUDA=OFF"]
    env.update(BUILD_NUM_THREADS=str(args.jobs), TAICHI_CMAKE_ARGS=" ".join(flags))
    # Do not reuse CMake objects configured with another ABI/toolchain/backend set.
    identity = {"python": facts, "flags": flags, "cc": env["CC"], "cxx": env["CXX"],
                "source_patch_sha256": patch_hash}
    identity_path = work / "build-identity.json"
    if ((source / "_skbuild").exists() and identity_path.exists()
            and json.loads(identity_path.read_text()) != identity):
        raise RuntimeError("Build configuration changed; use a fresh --build-dir")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    run([build_python, "setup.py", "bdist_wheel"], cwd=source, env=env,
        log=work / "build.log", timeout=7200)
    wheels = list((source / "dist").glob("taichi-1.7.4-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one wheel, found {wheels}")
    wheel = wheels[0]
    stage = work / "verify-env"
    run([uv, "venv", "--clear", "--python", python, stage])
    run([uv, "pip", "install", "--python", stage / "bin/python", wheel])
    verified, device = verify_backends(stage / "bin/python", work)
    manifest: dict[str, Any] = {"source": REPO, "commit": COMMIT, "wheel": str(wheel),
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "host": platform.platform(), "verified_arches": verified,
                "vulkan_device": device,
                "build": identity, "installed": False}
    if not args.no_install:
        response = work / "installed-runtime.json"
        response.unlink(missing_ok=True)
        run([python, "-m", "based_thermals.runtime_setup.taichi", "--wheel", wheel,
             "--arch", "cpu", "--output", response],
            cwd=project, log=work / "installed-runtime.log", timeout=1500)
        selected = json.loads(response.read_text())
        if not selected.get("available") or not selected.get("path"):
            raise RuntimeError("Compiled wheel was not activated in the application runtime cache")
        manifest["runtime_path"] = selected["path"]
        manifest["installed"] = True
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified wheel: {wheel}\nSHA256: {manifest['sha256']}", flush=True)
    print("Registered runtimes live outside .venv and survive ordinary uv sync.", flush=True)


if __name__ == "__main__":
    main()
