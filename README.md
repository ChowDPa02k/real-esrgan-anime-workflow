# An Real-ESRGAN animevideov3 Workflow for Anime Upscaling

> Powered by [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

This Python notebook will utilize r-esrgan-ncnn-vulkan, ffmpeg, and mkvtoolnix to automate the process of super resolution, re-encoding, and remuxing of MKV files. 

Processing frames using NCNN executable, so this notebook does not need any CUDA/ROCm python environment.

For a 23.976fps, 24-minute TV anime, running this workflow requires **at least 500GB of hard drive space** and a graphics card with **more than 4GB of VRAM**.

The workflow provides preset x264 and x265 encoding parameters that ensure the encoding quality is adequate for TV (rather than Bluray) sources. If you have more stringent standards or plan to encode Bluray sources, set the `enable_encoding` parameter to `False` and seek out a higher-quality encoding solution.

*Note: HEVC x265 encoding for 4K is **very very very slow**. On my AMD Ryzen 9950X machine, the average speed is between 0.6 and 1.8fps.*

### Steps:

1. Put mkv files into `input`
2. Change the configurations in the first cell
3. Run all cells
4. Wait till the world ends

## Python version

While running this workflow, I found that the progress bar in Jupyter Notebook will lost after disconnected from webpage, then although the workflow is still running, the progress will become untraceable.

So I've made an .py version of workflow. You can still get console output after reconnecting. The params are almost the same as notebook, and just put mkv files into `./input` and run:

```sh
pip install better-ffmpeg-progress==3.1.0 tqdm
python ./main.py
```

Available arguments:

```
usage: main.py [-h] [--autodl] [--default-fps DEFAULT_FPS] [--disable-encoding] [--enable-avx512] [--gpus GPUS] [--threads THREADS] [--proxy PROXY] [--proxy-tls PROXY_TLS]

options:
  -h, --help            show this help message and exit
  --autodl              Help configure Vulkan environment when running on AutoDL (default: False)
  --default-fps DEFAULT_FPS
                        Default output fps when input fps undetected (default: 24000/1001)
  --disable-encoding    Disable video encoding, move png series to output folder
  --enable-avx512       Disable AVX-512 acceleration while encoding (default: False)
  --gpus GPUS           Specify GPU devices (default: 0)
  --threads THREADS     Thread assignment: <input>:<process>:<output> (default: 12:20:16)
  --proxy PROXY         Proxy server address (default: empty)
  --proxy-tls PROXY_TLS
                        TLS proxy address (default: same as --proxy)
```

## Ubuntu Pre-configure

Run the following command before get started, remove `sudo` if you're using root.

```bash
curl -fsSL https://mkvtoolnix.download/gpg-pub-moritzbunkus.gpg -o /tmp/gpg-pub-moritzbunkus.gpg
sudo mv /tmp/gpg-pub-moritzbunkus.gpg /etc/apt/keyrings/gpg-pub-moritzbunkus.gpg

uversion=$(grep -oE 'noble|jammy|focal|bionic' /etc/apt/sources.list | head -n 1)

if [ -n "$uversion" ]; then
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/gpg-pub-moritzbunkus.gpg] https://mkvtoolnix.download/ubuntu/ $uversion main" | sudo tee /etc/apt/sources.list.d/mkvtoolnix.list > /dev/null
    echo "deb-src [arch=amd64 signed-by=/etc/apt/keyrings/gpg-pub-moritzbunkus.gpg] https://mkvtoolnix.download/ubuntu/ $uversion main" | sudo tee -a /etc/apt/sources.list.d/mkvtoolnix.list > /dev/null

    curl -fsSL https://packages.lunarg.com/lunarg-signing-key-pub.asc -o /tmp/lunarg-signing-key-pub.asc
    sudo mv /tmp/lunarg-signing-key-pub.asc /etc/apt/trusted.gpg.d/lunarg.asc

    curl -fsSL "https://packages.lunarg.com/vulkan/1.4.304/lunarg-vulkan-1.4.304-$uversion.list" -o /tmp/lunarg-vulkan.list
    sudo mv /tmp/lunarg-vulkan.list /etc/apt/sources.list.d/lunarg-vulkan.list

    sudo apt-get update

    sudo apt-get install -y mkvtoolnix ffmpeg
    sudo apt-get install -y vulkan-sdk vulkan-tools libvulkan1 libsm6 libegl1
else
    echo "Distribuion is unsupported, the script is for Ubuntu 20.04 / 22.04"
fi
```

This command is suitable for Ubuntu. For other Distributions, just remember to install:

- ffmpeg (https://www.ffmpeg.org/download.html#build-linux)
- mkvtoolnix (https://mkvtoolnix.download/downloads.html)
- vulkan-sdk (https://vulkan.lunarg.com/sdk/home#linux)

## Benchmark

After running this workflow on several machines with different configurations, I believe its characteristics - leveraging ncnn while disregarding dedicated GPU computation frameworks, handling high-pressure HEVC encoding loads, fixed input resolution and total frame count - might make it particularly suitable as a benchmarking tool.

The Upscaling examines the GPU's sustained general-purpose computing capabilities, while the Encoding constitutes genuine 4K video processing. Any 1-5% theoretical performance gap would manifest as significant disparities during these dozen-hour full-load operations. 

Such discrepancies could serve as valuable references for video compression researchers - by setting aside hardware acceleration tricks and purely testing raw CPU computational power, we can clearly identify which solutions demonstrate truly robust performance through sheer processing might.

| CPU            | GPU          | OS      | Extracting | Upscaling         | Encoding |
| -------------- | ------------ | ------- | ---------- | ----------------- | -------- |
| Ryzen 9950x    | RTX 4070 TiS | Windows | 8.7x       | 58:32 at -j 12:24:16 | 10h35m   |
| Core i9-13900K | RTX 4090     | Windows | 7.3x       | 41:41 at -j 12:24:16 | 17h55m   |
|                |              |         |            |                   |          |

