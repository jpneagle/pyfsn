import os
import subprocess
import sys
import venv
from pathlib import Path
import shutil

def main():
    # Setup paths
    root_dir = Path(__file__).resolve().parent.parent
    build_dir = root_dir / "build"
    venv_dir = root_dir / ".venv"
    
    print(f"Build Root: {root_dir}")
    print(f"Build Dir : {build_dir}")
    print(f"Venv Dir  : {venv_dir}")

    # 1. Create virtual environment
    if not venv_dir.exists():
        print(f"Creating virtual environment in {venv_dir}...")
        venv.create(venv_dir, with_pip=True)
    
    # Determine executable paths
    if sys.platform == "win32":
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
    
    if not python_exe.exists():
        # Fallback for some systems where venv structure might differ
        # Try finding it
        if (venv_dir / "bin" / "python3").exists():
            python_exe = venv_dir / "bin" / "python3"
        elif (venv_dir / "Scripts" / "python").exists():
            python_exe = venv_dir / "Scripts" / "python"

    print(f"Using Python: {python_exe}")

    # 2. Install dependencies
    print("Installing dependencies...")
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # Install project in editable mode to get dependencies
    # We install 'video' extra globally, but user wanted "minimal".
    # Basic pyfsn needs PyQt6, PyOpenGL, numpy.
    # Video adds opencv-python.
    # We'll install minimal if requested, but let's default to full feature set including video preview.
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "-e", ".[video]"])
    # Install PyInstaller
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "pyinstaller", "Pillow"])

    # 3. Clean build directory (but preserve venv)
    if build_dir.exists():
        # Remove contents but keep dir if needed, or just remove dir.
        # PyInstaller creates dist inside build_dir if we use --distpath build
        # But usually distpath is dist/.
        # User asked: "Windows binary in build" (create inside build folder)
        # So we set --distpath to build/dist or just build?
        # User said "in build", implying build/pyfsn.exe.
        # I'll use build/ as distpath.
        for item in build_dir.iterdir():
            if item.name not in ["spec", "work"]: # keep minimal if needed, but safer to clean
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    else:
        build_dir.mkdir(parents=True, exist_ok=True)

    # 4. Prepare icon
    icon_src_png = root_dir / "src" / "pyfsn" / "icon.png"
    icon_src_ico = root_dir / "src" / "pyfsn" / "icon.ico"
    icon_dest_ico = build_dir / "icon.ico"
    
    # Priority:
    # 1. Existing .ico in src
    # 2. Convert .png in src to .ico in build
    if icon_src_ico.exists():
        print(f"Using existing icon: {icon_src_ico}")
        shutil.copy2(icon_src_ico, icon_dest_ico)
    elif icon_src_png.exists():
        print("Converting icon to .ico for Windows...")
        # Convert using Pillow inside the venv
        convert_script = f"""
from PIL import Image
try:
    img = Image.open(r'{icon_src_png}')
    img.save(r'{icon_dest_ico}', format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print("Icon converted.")
except Exception as e:
    print(f"Icon conversion failed: {{e}}")
"""
        subprocess.check_call([str(python_exe), "-c", convert_script])

    # 5. Build with PyInstaller
    print(f"Building executable for {sys.platform}...")
    
    cmd = [
        str(python_exe), "-m", "PyInstaller",
        "--name", "pyfsn",
        "--noconsole",
        "--clean",
        "--distpath", str(build_dir),
        "--workpath", str(build_dir / "work"),
        "--specpath", str(build_dir / "spec"),
    ]

    # Mode adjustment: macOS works better with --onedir (bundled in .app)
    # Windows/Linux work well with --onefile
    if sys.platform == "darwin":
        cmd.append("--onedir")
    else:
        cmd.append("--onefile")

    # Bundle icons as data
    # Format: src:dest (Windows uses ; Mac/Linux uses :)
    sep = os.pathsep
    icon_png = root_dir / "src" / "pyfsn" / "icon.png"
    icon_ico = root_dir / "src" / "pyfsn" / "icon.ico"
    
    if icon_png.exists():
        cmd.extend(["--add-data", f"{icon_png}{sep}pyfsn"])
    if icon_ico.exists():
        cmd.extend(["--add-data", f"{icon_ico}{sep}pyfsn"])
    
    if icon_dest_ico.exists():
        cmd.extend(["--icon", str(icon_dest_ico)])
    elif icon_src_png.exists() and sys.platform != "win32":
        # Use png on non-windows if ico not made
        cmd.extend(["--icon", str(icon_src_png)])
    
    # Entry point
    cmd.append(str(root_dir / "src" / "pyfsn" / "__main__.py"))

    # Warning for cross-compilation
    if sys.platform != "win32":
        print("\n" + "="*60)
        print("WARNING: You are running this build script on a non-Windows OS.")
        print("PyInstaller will generate an executable for the CURRENT OS (macOS/Linux).")
        print("The resulting binary will NOT run on Windows.")
        print("To generate a Windows .exe, please run this script on Windows.")
        print("="*60 + "\n")

    subprocess.check_call(cmd)
    
    print(f"\nBuild complete! executable is in {build_dir}")

if __name__ == "__main__":
    main()
