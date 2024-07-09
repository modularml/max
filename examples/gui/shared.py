# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import os
import subprocess
import time
from pathlib import Path
from typing import List

import psutil
import requests
import streamlit as st


def menu():
    st.sidebar.page_link("home.py", label="️Home", icon="⚡️")
    st.sidebar.page_link("pages/llama3.py", label="Llama3", icon="🦙")
    st.sidebar.page_link("pages/bert.py", label="Bert", icon="👓")
    st.sidebar.page_link("pages/yolo.py", label="YOLO", icon="🔍")
    st.sidebar.page_link(
        "pages/stable-diffusion.py", label="Stable Diffusion 1.5", icon="🎨"
    )


def cache_dir() -> Path:
    cache_folder = os.getenv("XDG_CACHE_PATH")
    if not cache_folder:
        return Path.home() / ".cache"
    return Path(cache_folder)


def modular_cache_dir() -> Path:
    modular_dir = cache_dir() / "modular"
    os.makedirs(modular_dir, exist_ok=True)
    return modular_dir


def kill_process(port: int, model_state=None):
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            for conn in proc.connections(kind="inet"):
                if conn.laddr.port == port:
                    text = (
                        f"Killing existing {proc.info['name']} with PID"
                        f" {proc.info['pid']} on port {port}"
                    )
                    if model_state:
                        model_state.info(text, icon="️🔫")
                    else:
                        st.write()
                    proc.kill()
                    return True
        except psutil.AccessDenied:
            continue
        except psutil.NoSuchProcess:
            continue
    return False


def format_time(seconds):
    """Return a pretty format based on how much time is left."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"


def run_subprocess_monitor_download(
    command: List[str],
    folder_path: str,
    total_size_mb: int,
    post_process_msg: str = "",
):
    """Run `command` in a subprocess and monitor `folder_path` will a progress bar
    for how much of the download has completed.
    """
    process = subprocess.Popen(command)

    # All in MB
    downloaded = 0
    compiling_models = st.empty()
    progress_bar = st.progress(
        0.0,
        f"(MB) Size: {total_size_mb} Downloaded: 0000 Speed: 0.0 MB/s",
    )
    download_start = time.time()
    resuming_from = -1
    while process.poll() is None:
        if downloaded > total_size_mb - 50:
            progress_bar.progress(1.0, "Downloaded!")
            if post_process_msg:
                compiling_models.info(post_process_msg)
        else:
            downloaded = 0
            for dirpath, _, filenames in os.walk(folder_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    downloaded += os.path.getsize(file_path)
            # Convert bytes to MB
            downloaded //= 1024**2
            if resuming_from == -1:
                resuming_from = downloaded
            speed = (downloaded - resuming_from) / (
                time.time() - download_start
            )
            progress_bar.progress(
                downloaded / total_size_mb,
                (
                    f"Size: {total_size_mb} MB Downloaded: {downloaded} MB"
                    f" Speed: {speed:.2f} MB/s"
                ),
            )
        time.sleep(1)
    compiling_models.empty()


def format_speed(bytes_per_second):
    if bytes_per_second >= 1024**3:
        return f"{bytes_per_second / (1024**3):.2f} GB/s"
    elif bytes_per_second >= 1024**2:
        return f"{bytes_per_second / (1024**2):.2f} MB/s"
    elif bytes_per_second >= 1024:
        return f"{bytes_per_second / 1024:.2f} KB/s"
    else:
        return f"{bytes_per_second:.2f} B/s"


def download_file(model_url: str, model_path: str, model_state=None):
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not model_url and not os.path.isfile(model_path):
            st.error(
                "`Model URL` not provided and `Model Path` is not a file",
                icon="🚨",
            )
            file_size = os.path.getsize(model_path)
        if not model_url and model_path:
            if model_state:
                model_state.success("Using local model.", icon="✅")
            else:
                st.success("Using local model.", icon="✅")
        if model_url:
            response = requests.get(model_url, stream=True)
            download_size = int(response.headers.get("content-length", 0))
            # If the file exists and has the right amount of data, don't download again.
            if os.path.isfile(model_path):
                file_size = os.path.getsize(model_path)
                if file_size + 1024 > download_size:
                    if model_state:
                        model_state.success(
                            "Model Previously Downloaded.", icon="✅"
                        )
                    else:
                        st.success("Model Already Downloaded.", icon="✅")
                    return
            start_time = time.time()
            with open(model_path, "wb") as file:
                download_progress = st.progress(0)
                status_text = st.empty()
                downloaded_size = 0

                for data in response.iter_content(1024):
                    file.write(data)
                    downloaded_size += len(data)
                    elapsed_time = time.time() - start_time
                    download_speed = downloaded_size / elapsed_time
                    remaining_time = (
                        download_size - downloaded_size
                    ) / download_speed

                    # Update progress bar and status text
                    download_progress.progress(downloaded_size / download_size)
                    download_status = (
                        f"size: {download_size / (1024 * 1024):.2f} MB\n"
                    )
                    download_status += (
                        f"Downloaded: {downloaded_size / (1024 * 1024):.2f} MB "
                    )
                    download_status += f"({format_speed(download_speed)})\n"
                    download_status += (
                        f"Estimated time left: {format_time(remaining_time)}\n"
                    )

                    if model_state:
                        model_state.text(download_status)
                    else:
                        status_text.text(download_status)
            st.success("Model downloaded successfully!", icon="🎉")
    except Exception as e:
        st.error(f"Couldn't download file: {e}", icon="🚨")
        return 1
