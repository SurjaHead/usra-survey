# ---
# deploy: true
# lambda-test: false
# ---
import os
import pathlib
import shutil
import sys
import threading
import time

import modal

# Set up Modal volume and image
volume = modal.Volume.from_name("imagenet-100-dataset", create_if_missing=True)
image = modal.Image.debian_slim().apt_install("tree")
app = modal.App("import-imagenet-100", image=image)

def start_monitoring_disk_space(interval: int = 30) -> None:
    """Start monitoring the disk space in a separate thread."""
    task_id = os.environ.get("MODAL_TASK_ID", "local")

    def log_disk_space(interval: int) -> None:
        while True:
            statvfs = os.statvfs("/")
            free_space = statvfs.f_frsize * statvfs.f_bavail
            print(
                f"{task_id} free disk space: {free_space / (1024**3):.2f} GB",
                file=sys.stderr,
            )
            time.sleep(interval)

    monitoring_thread = threading.Thread(target=log_disk_space, args=(interval,))
    monitoring_thread.daemon = True
    monitoring_thread.start()

@app.function(
    mounts=[modal.Mount.from_local_dir("/Users/suryasure/projects/USRA-Survey/data/imagenet-100", "/mnt/host/imagenet-100")],
    volumes={"/mnt/imagenet-100": volume},
    timeout=60 * 60 * 8,  # 8 hours
    ephemeral_disk=1000 * 1024,  # 1TB
)
def import_local_imagenet100():
    start_monitoring_disk_space()
    src = pathlib.Path("/mnt/host/imagenet-100")
    dst = pathlib.Path("/mnt/imagenet-100")
    print(f"Copying from {src} to {dst} ...")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print("Copy complete.")
    os.system("tree -L 3 /mnt/imagenet-100")

# To run this, you would use:
# modal run import_imagenet100_to_volume.py::import_local_imagenet100
# (where /mnt/host/imagenet-100 is your local path, mounted into the container)