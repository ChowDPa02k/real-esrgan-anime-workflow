#!/usr/bin/env python
# coding: utf-8

# # An Real-ESRGAN animevideov3 Workflow for Anime Upscaling
# 
# > Powered by [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
# 
# This Python notebook will utilize r-esrgan-ncnn-vulkan, ffmpeg, and mkvtoolnix to automate the process of super resolution, re-encoding, and remuxing of MKV files. 
# 
# Processing frames using NCNN executable, so this notebook does not need any CUDA/ROCm python environment.
# 
# For a 23.976fps, 24-minute TV anime, running this workflow requires **at least 500GB of hard drive space** and a graphics card with **more than 4GB of VRAM**.
# 
# The workflow provides preset x264 and x265 encoding parameters that ensure the encoding quality is adequate for TV (rather than Bluray) sources. If you have more stringent standards or plan to encode Bluray sources, set the `enable_encoding` parameter to `False` and seek out a higher-quality encoding solution.
# 
# *Note: HEVC x265 encoding for 4K is **very very very slow**. On my AMD Ryzen 9950X machine, the average speed is between 0.6 and 1.2fps.*
# 
# ### Steps:
# 
# 1. Put mkv files into `input`
# 2. Change the configurations in the first cell
# 3. Run all cells
# 4. Wait till the world ends

# In[1]:

import argparse
import platform
import os
import sys
import warnings

system_type = platform.system()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autodl',
                        action='store_true',
                        default=False,
                        help='Help configure Vulkan environment when running on AutoDL (default: %(default)s)')
    parser.add_argument('--default-fps',
                        type=str,
                        default='24000/1001',
                        help='Default output fps when input fps undetected (default: %(default)s)')
    # parser.add_argument('--upscale-temp-path', type=str, default='upscale_frames',
    #                     help='Temporary folder for upscaled PNG series (default: %(default)s)')
    parser.add_argument('--disable-encoding',
                        action='store_false',
                        dest='enable_encoding',
                        help='Disable video encoding, move png series to output folder')
    parser.set_defaults(enable_encoding=True)
    parser.add_argument('--enable-avx512',
                        action='store_true',
                        dest='enable_avx512',
                        help='Disable AVX-512 acceleration while encoding (default: %(default)s)')
    parser.set_defaults(enable_avx512=False)
    parser.add_argument('--gpus',
                        type=str,
                        default='0',
                        help='Specify GPU devices (default: %(default)s)')
    parser.add_argument('--threads',
                        type=str,
                        default='12:20:16',
                        help='Thread assignment: <input>:<process>:<output> (default: %(default)s)')
    parser.add_argument('--proxy',
                        type=str,
                        default='',
                        help='Proxy server address (default: empty)')
    parser.add_argument('--proxy-tls',
                        type=str,
                        default=None,
                        help='TLS proxy address (default: same as --proxy)')

    args = parser.parse_args()
    args.proxy_tls = args.proxy if args.proxy_tls is None else args.proxy_tls
    return args




# ## Pre-request for OS

# In[2]:


import subprocess
import requests


def download_file(url, out):
    proxies = {
        'http': os.environ.get('http_proxy'),
        'https': os.environ.get('https_proxy')
    }
    response = requests.get(url, proxies=proxies)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            f.write(response.content)
        print(f'[OK] Files saved to {out}')
    else:
        print(f'[ERROR] Failed to download {url}')

def prepare_env():
    _ffmpeg, _ffprobe, _mkvtool, _resr = '', '', '', ''

    if system_type == "Windows":
        if not os.path.exists('7zr.exe'):
            download_file('https://www.7-zip.org/a/7zr.exe', out='7zr.exe')
        if not os.path.exists('mkvtoolnix'):
            download_file('https://mkvtoolnix.download/windows/releases/90.0/mkvtoolnix-64-bit-90.0.7z', out='mkvtoolnix.7z')
        if not os.path.exists('ffmpeg'):
            download_file('https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z', out='ffmpeg.7z')
        if not os.path.exists('realesrgan-ncnn-vulkan'):
            download_file('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip', out='r-esrgan.zip')
        
        if os.path.exists('ffmpeg.7z'):
            subprocess.run(['7zr.exe', 'x', 'ffmpeg.7z'])
            os.remove('ffmpeg.7z')
            ffmpeg_folder = [i for i in os.listdir() if i.startswith('ffmpeg-')]
            if ffmpeg_folder:
                os.rename(ffmpeg_folder[0], 'ffmpeg')
                os.environ['PATH'] += os.pathsep + os.path.abspath('.\\ffmpeg\\bin')

        if os.path.exists('mkvtoolnix.7z'):
            subprocess.run(['7zr.exe', 'x', 'mkvtoolnix.7z'])
            os.remove('mkvtoolnix.7z')

        if os.path.exists('r-esrgan.zip'):
            subprocess.run(['powershell', '-Command', "Expand-Archive .\\r-esrgan.zip -DestinationPath realesrgan-ncnn-vulkan"])
            os.remove('r-esrgan.zip')

        _ffmpeg = '.\\ffmpeg\\bin\\ffmpeg.exe'
        _ffprobe = '.\\ffmpeg\\bin\\ffprobe.exe'
        _mkvtool = '.\\mkvtoolnix\\mkvmerge.exe'
        _resr = '.\\realesrgan-ncnn-vulkan\\realesrgan-ncnn-vulkan.exe'

    if system_type == "Linux":
        # Ubunu only
        # Need root to perform installation
        # ---------------------------------
        # download_file('https://mkvtoolnix.download/gpg-pub-moritzbunkus.gpg', out='/etc/apt/keyrings/gpg-pub-moritzbunkus.gpg')
        # with open('/etc/apt/sources.list.d/mkvtoolnix.list', 'w') as f:
        #     uversion = subprocess.run(['grep', '-oE', 'noble|jammy|focal|bionic', '/etc/apt/sources.list'], capture_output=True, text=True).stdout.splitlines()[0]
        #     if uversion:
        #         f.write(f'deb [arch=amd64 signed-by=/etc/apt/keyrings/gpg-pub-moritzbunkus.gpg] https://mkvtoolnix.download/ubuntu/ {uversion} main\n')
        #         f.write(f'deb-src [arch=amd64 signed-by=/etc/apt/keyrings/gpg-pub-moritzbunkus.gpg] https://mkvtoolnix.download/ubuntu/ {uversion} main\n')
        #         download_file('https://packages.lunarg.com/lunarg-signing-key-pub.asc', out='/etc/apt/trusted.gpg.d/lunarg.asc')
        #         download_file(f'https://packages.lunarg.com/vulkan/1.4.304/lunarg-vulkan-1.4.304-{uversion}.list', out='/etc/apt/sources.list.d/lunarg-vulkan.list')
        #         subprocess.run(['apt-get', 'update'])
        #         subprocess.run(['apt-get', 'install', '-y', 'mkvtoolnix', 'ffmpeg']) 
        #         subprocess.run(['apt-get', 'install', '-y', 'vulkan-sdk', 'vulkan-tools', 'libvulkan1', 'libsm6', 'libegl1'])
        # ---------------------------------
        # If you are running in non-root user, comment the above code
        # and run the bash script in the next cell

        download_file('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip', out='r-esrgan.zip')
        os.makedirs('realesrgan-ncnn-vulkan', exist_ok=True)
        subprocess.run(['unzip', 'r-esrgan.zip', '-d', 'realesrgan-ncnn-vulkan'])
        subprocess.run(['chmod', '+x', 'realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan'])
        os.remove('r-esrgan.zip')

        _ffmpeg = 'ffmpeg'
        _ffprobe = 'ffprobe'
        _mkvtool = 'mkvmerge'
        _resr = './realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan'

        if autodl == True:
            with open('/etc/vulkan/icd.d/my_nvidia_icd.json', 'w') as f:
                f.write('{"file_format_version":"1.0.0","ICD":{"library_path":"/lib/x86_64-linux-gnu/libEGL_nvidia.so.0","api_version":"1.3.277"}}')
            os.environ.update(VK_ICD_FILENAMES='/etc/vulkan/icd.d/my_nvidia_icd.json')

    if system_type == "Darwin":
        raise Exception("macOS is not supported yet")

    print(f'FFmpeg      : {_ffmpeg}')
    print(f'MKVToolNix  : {_mkvtool}')
    print(f'Real-ESRGAN : {_resr}')
    print(f'FFprobe     : {_ffprobe}')

    return _ffmpeg, _ffprobe, _mkvtool, _resr

# # Prepare Functions for Workflow

# ## Pre-request for Python

# In[3]:


# get_ipython().system('pip install better-ffmpeg-progress==3.1.0 tqdm')


# ## Modify better-ffmpeg-progress to use tqdm
# 
# Just make it looks prettier

# In[4]:


from enum import Enum
from pathlib import Path
import psutil
from queue import Empty, Queue
import subprocess
from threading import Thread
from typing import List, Optional, Union
from ffmpeg import probe
from better_ffmpeg_progress import FfmpegProcess
from tqdm.auto import tqdm
from tqdm import TqdmWarning

warnings.filterwarnings("ignore", category=TqdmWarning)

class FfmpegLogLevel(Enum):
    QUIET = "quiet"
    PANIC = "panic"
    FATAL = "fatal"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    VERBOSE = "verbose"
    DEBUG = "debug"
    TRACE = "trace"

class TqdmFfmpegProcess(FfmpegProcess):
    def __init__(
        self,
        command: List[str],
        ffmpeg_log_level: Optional[FfmpegLogLevel] = None,
        ffmpeg_log_file: Optional[Union[str, Path]] = None,
        print_detected_duration: bool = False,
        print_stderr_new_line: bool = False,
        raw_file: str = ''
    ):
        super().__init__(command, ffmpeg_log_level, ffmpeg_log_file, print_detected_duration, print_stderr_new_line)
        self.raw_file = raw_file
        if self.raw_file:
            try:
                probe_data = probe(self.raw_file)
                self._duration_secs = float(probe_data["format"]["duration"])
                if self._print_detected_duration:
                    print(f"Detected duration: {self._duration_secs:.2f} seconds")
            except Exception:
                self._duration_secs = None

    @staticmethod
    def _read_pipe(pipe: subprocess.PIPE, queue: Queue, stdout: bool = False) -> None:
        """Read from pipe and put lines into queue."""
        try:
            with pipe:
                for line in iter(pipe.readline, ""):
                    line = line.strip()
                    if line:
                        if stdout:
                            if any((
                                line.startswith("out_time_ms="),
                                line.startswith("bitrate="),
                                line.startswith("speed="),
                                line.startswith("fps=")
                            )):
                                queue.put(line)
                        else:
                            queue.put(line)
        except (IOError, ValueError) as e:
            print(f"Error reading pipe: {e}")
        finally:
            pipe.close()

    @classmethod
    def _validate_command(cls, command: List[str]) -> bool:
        if not shutil.which("ffmpeg"):
            print("Error: FFmpeg executable not found in PATH")
            return False

        if "-i" not in command:
            print("Error: FFmpeg command must include '-i'")
            return False

        input_idx = command.index("-i") + 1
        if input_idx >= len(command):
            print("Error: No input file specified after -i")
            return False

        input_file = Path(command[input_idx])
        if not input_file.exists() and '%' not in str(input_file):
            print(f"Error: Input file not found: {input_file}")
            return False

        if input_idx + 1 >= len(command):
            print("Error: No output file specified")
            return False

        return True

    def run(self, prefix: str = 'Encoding', title: str = '', print_command: bool = False) -> int:
        if hasattr(self, "return_code") and self.return_code != 0:
            return 1

        if self._output_filepath.exists() and "-y" not in self._ffmpeg_command:
            if (
                input(
                    f"{self._output_filepath} already exists. Overwrite? [Y/N]: "
                ).lower()
                != "y"
            ):
                print(
                    "FFmpeg process cancelled. Output file exists and overwrite declined."
                )
                return 1
            self._ffmpeg_command.insert(1, "-y")

        if print_command:
            print(f"Executing: {' '.join(self._ffmpeg_command)}")

        def _contains_shell_operators(command: List[str]) -> bool:
            shell_operators = {"|", ">", "<", ">>", "&&", "||"}
            return any(op in command for op in shell_operators)

        # If command contains shell operators, turn the list into a string.
        if _contains_shell_operators(self._ffmpeg_command):
            self._ffmpeg_command = " ".join(self._ffmpeg_command)

        try:
            process = subprocess.Popen(
                self._ffmpeg_command,
                shell=isinstance(self._ffmpeg_command, str),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0,
            )
        except Exception as e:
            print(f"Error starting FFmpeg process: {e}")
            return 1

        stdout_queue, stderr_queue = Queue(), Queue()
        # Start stdout and stderr pipe readers
        Thread(
            target=self._read_pipe,
            args=(process.stdout, stdout_queue, True),
            daemon=True,
        ).start()
        Thread(
            target=self._read_pipe,
            args=(process.stderr, stderr_queue, False),
            daemon=True,
        ).start()
        
        try:
            with tqdm(total=1, 
                      desc=prefix, 
                      bar_format='{desc} {elapsed} {percentage:3.2f}%|{bar}| ETA={remaining} {postfix}',
                      position=2, leave=False
                      ) as progress_bar:
                
                if self._duration_secs:
                    progress_bar.reset(total=self._duration_secs)

                info = {}
                # if title:
                #     init, current = 10, 10
                #     pos = 0
                #     window = ''
                #     backgroud = ' ' * 16 + title + ' ' * 16
                #     roller_length = len(backgroud)
                while process.poll() is None:
                    if not stdout_queue.empty():
                        try:
                            line = stdout_queue.get()
                            if line.startswith(self.TIME_MS_PREFIX):
                                try:
                                    value = int(line.split("=")[1]) / 1_000_000
                                    if value <= self._duration_secs:
                                        progress_bar.n = value
                                except ValueError:
                                    pass

                            if line.startswith('bitrate='):
                                try:
                                    value = line.split("=")[1]
                                    info['bitrate'] = value
                                except ValueError:
                                    pass
                            
                            if line.startswith('fps='):
                                try:
                                    value = line.split("=")[1]
                                    info['fps'] = value
                                except ValueError:
                                    pass

                            if line.startswith('speed='):
                                try:
                                    value = line.split("=")[1]
                                    info['speed'] = value
                                except ValueError:
                                    pass

                            progress_bar.set_postfix(info)
                            # if title:
                            #     if current == init:
                            #         if pos >= roller_length - 16:
                            #             pos = 0
                            #         window = backgroud[pos:pos+16]
                            #         progress_bar.set_description(f'{prefix} [{window}]')
                            #         pos += 1
                            #     current = init if current == 0 else current - 1

                        except Empty:
                            pass

                    # stderr
                    if not stderr_queue.empty():
                        try:
                            self._process_stderr(stderr_queue.get_nowait())
                        except Empty:
                            pass
                
                # Process remaining stderr
                while not stderr_queue.empty():
                    try:
                        self._process_stderr(stderr_queue.get_nowait())
                    except Empty:
                        break
                
                if process.returncode != 0:
                    progress_bar.write(
                        f"The FFmpeg process did not complete successfully. Check out {self._ffmpeg_log_file} for details.",
                    )
                    # progress_bar.container.children[0].bar_style = "danger"
                    return 1

                progress_bar.n = progress_bar.total
                progress_bar.set_description(prefix)
                # progress_bar.container.children[0].bar_style = "success"

        except KeyboardInterrupt:
            # progress_bar.container.children[0].bar_style = "danger"
            try:
                psutil.Process(process.pid).terminate()
            except psutil.NoSuchProcess:
                pass
            return 1
        
        return 0 if process.returncode == 0 else 1


# ## Extract Frame Series
# 
# Frames will be saved as RGB PNG.
# 
# For 1080P 24min@23.976fps video, this step will generate ~60GB of images.

# In[5]:


def extract_frames(input_file: str):
    process = TqdmFfmpegProcess([
        _ffmpeg, '-i', input_file, '-qscale:v', '1', '-qmin', '1', '-qmax', '1', '-vsync', '0', 'tmp_frames/frame_%08d.png'
    ], ffmpeg_log_file=f'logs/{src_file}_extract.log')
    return process.run(prefix='Extracting frames')

def count_frames(path: str = 'tmp_frames'):
    return len(os.listdir(path))


# ## Upscale with NCNN Vulkan Executable

# In[6]:


from time import sleep

def upscale_frames(input_folder: str = 'tmp_frames', output_folder: str = 'upscale_frames'):
    cmd = [_resr, '-i', input_folder, '-o', output_folder, '-n', 'realesr-animevideov3', '-s', '2', '-f', 'png', '-g', gpus, '-j', threads]
    total_frames = count_frames(input_folder)

    try:
        with open(f'logs/{src_file}_upscale.log', 'w') as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            # p = subprocess.Popen(cmd)
            with tqdm(total=total_frames, desc='Upscaling', unit='frame', position=2, leave=False,
                    bar_format='{desc}: {elapsed} {percentage:3.2f}%|{bar}| {n_fmt}/{total_fmt}, {rate_fmt}, ETA {remaining}'
                    ) as pbar:
                
                current = 0
                while p.poll() is None:
                    new = count_frames(output_folder)
                    pbar.update(new - current)
                    current = new
                    # print(pbar.n)
                    sleep(0.5)
                sleep(0.5)
                if p.returncode == 0:
                    pbar.n = count_frames(output_folder)
                    pbar.refresh()
                
            return p.returncode

    except KeyboardInterrupt:
        p.terminate()
        p.wait()
        return 1


# ## Re-encode Upscaled Frames
# 
# Encode params are tuned for 4K animation

# In[7]:


x264_params = ':'.join([
    "deblock=-1,-1", 
    "keyint=600", 
    "min-keyint=1", 
    "bframes=8", 
    "ref=5", 
    "qcomp=0.6", 
    "no-mbtree=1", 
    "rc-lookahead=60", 
    "aq-strength=0.8", 
    "me=tesa", 
    "psy-rd=0,0", 
    "chroma-qp-offset=-1", 
    "no-fast-pskip=1", 
    "aq-mode=2", 
    "colorprim=bt709", 
    "transfer=bt709", 
    "colormatrix=bt709", 
    "chromaloc=0", 
    "fullrange=off"
])

x265_params = ':'.join([
    "deblock=-1,-1", 
    "ctu=64", 
    "qg-size=8", 
    "crqpoffs=-2", 
    "cbqpoffs=-2", 
    "me=4", 
    "subme=6", 
    "merange=64", 
    "b-intra=1", 
    "limit-tu=4", 
    "no-amp=1", 
    "ref=6", 
    "weightb=1", 
    "keyint=360", 
    "min-keyint=1", 
    "bframes=6", 
    "aq-mode=1", 
    "aq-strength=0.8", 
    "rd=5", 
    "psy-rd=2.0", 
    "psy-rdoq=1.0", 
    "rdoq-level=2", 
    "no-open-gop=1", 
    "rc-lookahead=80", 
    "scenecut=40", 
    "qcomp=0.65", 
    "no-strong-intra-smoothing=1",
    "nr-intra=0",
    "nr-inter=0"
])


def get_framerate(input_file: str):
    cmd = f"{_ffprobe} -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate".split() + [input_file]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return default_fps

def encode_frames(output_file: str, input_folder: str = 'upscale_frames', raw_file: str = '', engine: str = 'libx264', crf: int = 18):
    output_file = os.path.join('output', output_file)

    # fps = get_framerate(raw_file)
    if engine == 'libx264':
        cmd = [_ffmpeg, '-framerate', src_fps, '-i', f'{input_folder}/frame_%08d.png', '-c:v', 'libx264', '-preset', 'veryslow', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-x264-params', x264_params, '-r', src_fps, output_file]
    elif engine == 'libx265':
        cmd = [_ffmpeg, '-framerate', src_fps, '-i', f'{input_folder}/frame_%08d.png', '-c:v', 'libx265', '-preset', 'slower', '-crf', str(crf), '-pix_fmt', 'yuv420p10le', '-x265-params', x265_params, '-r', src_fps, output_file]
    else:
        tqdm.write(f'Unsupported engine: {engine}')
        return 1
    
    process = TqdmFfmpegProcess(cmd, raw_file=raw_file, ffmpeg_log_file=f'logs/{src_file}_encode.log')
    return process.run(prefix=f'Encoding {"AVC" if engine == "libx264" else "HEVC"}')


# ## Remux Encoded Stream to Input file

# In[8]:


def remux_video_stream(input_file: str, stream_file: str, output_file: str):
    # print(f'Remuxing {stream_file} with {input_file} to {output_file}')
    cmd = [_mkvtool, '-o', output_file, '-D', input_file, '-A', '-S', '-T', '-M', '-B', '--no-chapters', '--default-duration', f'0:{src_fps}p', stream_file]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.stdout:
        tqdm.write(result.stdout)
    if result.returncode != 0:
        tqdm.write(f'{result.stderr}')
        return result.returncode
    return 0


# # Everything is ready, Let's begin!

# In[9]:


import shutil

def clean_folder(folder: str):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


# In[ ]:

if __name__ == '__main__':
    global src_file

    args = parse_args()
    autodl = args.autodl
    default_fps = args.default_fps
    enable_encoding = args.enable_encoding
    enable_avx512 = args.enable_avx512
    gpus = args.gpus
    threads = args.threads
    proxy = args.proxy
    proxy_tls = args.proxy_tls
    # upscaled_folder = args.upscale_temp_path

    if enable_avx512:
        x265_params += ':asm=avx512'

    # Environment Magic
    os.environ.update(http_proxy=proxy)
    os.environ.update(https_proxy=proxy_tls)
    os.environ.update(HTTP_PROXY=proxy)
    os.environ.update(HTTPS_PROXY=proxy_tls)
    print(os.getcwd())
    os.makedirs('input', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('tmp_frames', exist_ok=True)
    os.makedirs('upscale_frames', exist_ok=True)

    _ffmpeg, _ffprobe, _mkvtool, _resr = prepare_env()

    input_files = [f for f in os.listdir('input') if f.endswith('.mkv')]
    for input_file in tqdm(input_files, desc='Processing files', unit='file', bar_format='{desc} {elapsed}|{bar}| {n_fmt}/{total_fmt} files, {rate_inv_fmt}', position=1):
        src_file, ext = os.path.splitext(input_file)
        input_path = os.path.join('input', input_file)
        transcode_file = 'output.hevc'
        output_mkv = f'[Upscaled]{src_file}.mkv'
        tqdm.write(f'\nProcessing {src_file}...')
        
        try:
            # Clean temporary folders
            clean_folder('tmp_frames')
            clean_folder('upscale_frames')

            global src_fps
            # Get source fps
            src_fps = get_framerate(input_path)

            # Extract frames
            if extract_frames(input_path) != 0:
                tqdm.write(f"Failed to extract frames for {input_file}")
                continue
            else:
                tqdm.write(f'Extract: finished {count_frames('tmp_frames')} frames')
            
            # Upscale frames
            if upscale_frames() != 0:
                tqdm.write(f"Failed to upscale frames for {input_file}")
                continue
            else:
                tqdm.write(f'Upscale: finished {count_frames('upscale_frames')} frames')
            
            if enable_encoding:
                # Encode frames
                if encode_frames(transcode_file, input_folder=os.path.abspath('upscale_frames'), raw_file=input_path, engine='libx265') != 0:
                    tqdm.write(f"Failed to encode frames for {input_file}")
                    continue
                else:
                    size = os.path.getsize('output/output.hevc') / 1014 / 1024
                    fps_f = (int(src_fps.split('/')[0]) / int(src_fps.split('/')[1])) if '/' in src_fps else int(src_fps)
                    frames = count_frames('upscale_frames')
                    duration_s = frames / fps_f
                    bitrate = size * 8 * 1024 / duration_s
                    tqdm.write(f"Encode : Average {bitrate:.0f} Kbps, {size:.2f} MB finished")
                
                # Remux video stream
                if remux_video_stream(input_path, os.path.join('output', transcode_file), os.path.join('output', output_mkv)) != 0:
                    tqdm.write(f"Failed to remux video stream for {input_file}")
                    continue
                else:
                    # Clean output.hevc if successful
                    if os.path.exists(os.path.join('output', transcode_file)):
                        os.remove(os.path.join('output', transcode_file))
            else:
                os.rename('upscale_frames', f'[Upscaled]{src_file}')
                shutil.move(f'[Upscaled]{src_file}', 'output')
                os.makedirs('upscale_frames', exist_ok=True)
        
        except Exception as e:
            tqdm.write(f"An error occurred while processing {input_file}: {e}")
            continue
        
        finally:
            # Clean temporary folders
            if os.path.exists(os.path.join('output', transcode_file)):
                os.remove(os.path.join('output', transcode_file))
            clean_folder('tmp_frames')
            clean_folder('upscale_frames')
