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

## Benchmark

After running this workflow on several machines with different configurations, I believe its characteristics - leveraging ncnn while disregarding dedicated GPU computation frameworks, handling high-pressure HEVC encoding loads, fixed input resolution and total frame count - might make it particularly suitable as a benchmarking tool.

The Upscaling examines the GPU's sustained general-purpose computing capabilities, while the Encoding constitutes genuine 4K video processing. Any 1-5% theoretical performance gap would manifest as significant disparities during these dozen-hour full-load operations. 

Such discrepancies could serve as valuable references for video compression researchers - by setting aside hardware acceleration tricks and purely testing raw CPU computational power, we can clearly identify which solutions demonstrate truly robust performance through sheer processing might.

| CPU            | GPU          | OS      | Extracting | Upscaling         | Encoding |
| -------------- | ------------ | ------- | ---------- | ----------------- | -------- |
| Ryzen 9950x    | RTX 4070 TiS | Windows | 8.7x       | 58:32 at -j 16:16 | 10h35m   |
| Core i9-13900K | RTX 4090     | Windows | 7.3x       | 41:41 at -j 16:16 | 17h55m   |
|                |              |         |            |                   |          |

