
<h2 align="center">Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data</h2>

<div align="center">

[[Website]](https://opendrivelab.com/AMS/) · [[Arxiv]](https://arxiv.org/abs/2511.17373)

</div>

## Motion Generation

### Contents
- [🏠 Description](#description_section)
- [📦 Setup](#setup_section)
- [📚 Usage](#usage_section)
- [🔗 Citation](#citation_section)
- [📄 License](#license_section)
- [👏 Acknowledgements](#acknowledgements_section)

## 🏠 Description
<a name="description_section"></a>

Implementation for **“Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data”**, built on [Pyroki](https://github.com/chungmin99/pyroki). Generates stable balance motions for the **29-DoF Unitree G1**.


https://github.com/user-attachments/assets/851a4a0e-5103-4f3d-bd30-2189a93f917a



## 📦 Setup
<a name="setup_section"></a>

```bash
conda create -n motion_gen python=3.10
conda activate motion_gen

git clone https://github.com/chungmin99/pyroki.git
cd pyroki && pip install -e . && cd ..

pip install -r requirements.txt
```


## 📚 Usage
<a name="usage_section"></a>

The motion generation pipeline consists of three main stages: generation, quality assurance (collision checking), and data filtering.

### 1. Generate motions
Generate stable balance sequences for the Unitree G1.
```bash
python g1_sample_sequences.py \
  --sample.num-samples 4 \
  --sample.side both \
  --seq.num-steps 60
```

Common knobs:
- `--sample.output-path PATH` (default: `motions/g1/sampled_static_poses`)
- `--sample.max-retries INT`
- `--sample.float-x-range xmin xmax`
- `--sample.float-y-abs-range ymin ymax`
- `--sample.float-z-range zmin zmax`
- `--sample.pelvis-height-range zmin zmax`
- `--seq.enable-collision-check BOOL`

Output goes to `--sample.output-path/<timestamp>/*.pkl`.

#### Output format
Each `*.pkl` is a dict with one key; value contains:
- `root_trans_offset`: `(T, 3)` root position
- `root_rot`: `(T, 4)` quaternion **xyzw**
- `dof`: `(T, 29)` joint DoFs
- `pose_aa`: `(T, 30, 3)` axis-angle (root + joints)
- `fps`: int
- `stance_leg`: `"left"` / `"right"`


### 2. Motion Validation Check
Detect self-collision and ground penetration using the MuJoCo engine.

```bash
python scripts/auto_check.py \
  --motion_folder motions/g1/sampled_static_poses/<timestamp> \
  --robot_type unitree_g1 \
  --batch_size 100
```


Results are written under `annotations/<motion_folder_name>/` (progress + abnormal-only reports/annotations).



### 3. Filter and Merge
Finalize the dataset by removing "abnormal" motions identified in the motion validation step.

```bash
python scripts/process_annotations_to_pkl.py \
  -a annotations/<folder>/auto_check_annotations.json \
  --motion_source motions/g1/sampled_static_poses/<timestamp> \
  --save_name filtered_motions
```


<!-- we provided a [synthetic balance motion dataset](xxx) containing approximately 7,000 sequences generated using this exact procedure. -->

---

### Motion Visualization
We provide a visualization tool to help inspecting the generated motions.
```bash
python scripts/vis_motion.py path/to/motion.pkl \
  --urdf robot/unitree_description/urdf/g1/g1_sysid_29dof.urdf \
  --fps 30
```




## 🔗 Citation
<a name="citation_section"></a>

```bibtex
@article{pan2025ams,
  title={Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data},
  author={Pan, Yixuan and Qiao, Ruoyi and Chen, Li and Chitta, Kashyap and Pan, Liang and Mai, Haoguang and Bu, Qingwen and Zheng, Cunyuan and Zhao, Hao and Luo, Ping and Li, Hongyang},
  journal={arXiv preprint arXiv:2511.17373},
  year={2025}
}
```

## 📄 License

<a name="license_section"></a>

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## 👏 Acknowledgements

<a name="acknowledgements_section"></a>

- [Pyroki](https://github.com/pyroki/pyroki): We use `pyroki` library to generate the balance motion for the G1 robot.
