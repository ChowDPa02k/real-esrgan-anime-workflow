# An Real-ESRGAN animevideov3 Workflow for Anime Upscaling

> Powered by [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

This Python notebook will utilize r-esrgan-ncnn-vulkan, ffmpeg, and mkvtoolnix to automate the process of super resolution, re-encoding, and remuxing of MKV files. 

For a 23.976fps, 24-minute TV anime, running this workflow requires **at least 500GB of hard drive space** and a graphics card with **more than 4GB of VRAM**.

The workflow provides preset x264 and x265 encoding parameters that ensure the encoding quality is adequate for TV (rather than Bluray) sources. If you have more stringent standards or plan to encode Bluray sources, set the `enable_encoding` parameter to `False` and seek out a higher-quality encoding solution.

*Note: HEVC x265 encoding for 4K is **very very very slow**. On my AMD Ryzen 9950X machine, the average speed is between 0.6 and 1.2fps.*

### Steps:

1. For the first time, run the `Environment Magic` cell and Pre-request chapter for your OS
2. Run `Env Definition` cell
3. Put all of the mkv files into `input`
4. Run `Pre-request for Python` and **all of the cells below**
5. Wait till the world ends
