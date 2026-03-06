<h1 align="center"> Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data </h1>

<div align="center">

<!-- **Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data** -->

<!-- [[Website]](https://opendrivelab.com/AMS/)
[[Arxiv]](https://arxiv.org/abs/2511.17373) -->
<!-- [[Video]](https://www.youtube.com/watch?v=mzXH4MEypsk) -->
<p align="center">
    <a href="https://arxiv.org/abs/2511.17373">
        <img src='https://img.shields.io/badge/Arxiv-lightgreen?style=for-the-badge&labelColor=006400&color=006400' alt='Paper PDF'></a>
    <a href='https://opendrivelab.com/AMS/'>
        <img src='https://img.shields.io/badge/Website-darkgreen?style=for-the-badge&labelColor=006400' alt='Project Page'></a>
    <a href='https://huggingface.co/datasets/ruoyiqiao/AMS'>
        <img src='https://img.shields.io/badge/Dataset-lightgreen?style=for-the-badge&labelColor=006400&color=006400' alt='Dataset'></a>
    <!-- <a href=""><img alt="youtube views" src="https://img.shields.io/badge/Video-red?style=for-the-badge&logo=youtube&labelColor=ce4630&logoColor=red"/></a> -->
</p>

Yixuan Pan<sup>1*</sup>, Ruoyi Qiao<sup>4*</sup>, L. Chen<sup>1</sup>, K. Chitta<sup>2</sup>, L. Pan<sup>1</sup>, H.Mai<sup>1</sup>, Q. Bu<sup>1</sup>, C. Zheng<sup>4</sup>, H. Zhao<sup>3</sup>, P. Luo<sup>1</sup>, H. Li<sup>1</sup>

*Equal Contribution

<sup>1</sup>The University of Hong Kong, <sup>2</sup>NVIDIA, <sup>3</sup>Tsinghua University, <sup>4</sup>Individual Contributor
<p align="center">
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/master/AMS/HKU_logo_1.png" height="45" style="vertical-align: middle;">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/master/AMS/NVIDIA_logo.png" height="35" style="vertical-align: middle;">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/OpenDriveLab/opendrivelab.github.io/master/AMS/Tsinghua_logo.png" height="45" style="vertical-align: middle;">
</p>

</div>

## News
* **`[2026-03-06]`** Synthetic motion dataset is released on 🤗 Hugging Face.
* **`[2026-03-02]`** Balance motion generation pipeline is released. 🤹🏽
* **`[2026-01-31]`** Accepted to ICRA 2026.

## Usage

Clone the repository:
```bash
git clone https://github.com/OpenDriveLab/AMS.git
cd AMS
```
For detailed instructions of motion generation and the synthetic motion dataset, please refer to the [MotionGen/README.md](MotionGen/README.md) in the MotionGen sub-repositories.


## TODO List
> Note: we re-implement the AMS code for 29dof G1 humanoid.
- [x] Release MotionGen: a sample-based scalable balance motion generator
- [x] Synthetic Balance Motion Datasets
- [ ] Release training and evaluation code
- [ ] Release Sim2Sim code
- [ ] Release deployment code


## Citation

If you find our work useful, please consider citing us!

```bibtex
@article{pan2025ams,
  title={Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data},
  author={Pan, Yixuan and Qiao, Ruoyi and Chen, Li and Chitta, Kashyap and Pan, Liang and Mai, Haoguang and Bu, Qingwen and Zheng, Cunyuan and Zhao, Hao and Luo, Ping and Li, Hongyang},
  journal={arXiv preprint arXiv:2511.17373},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
