# 3DGS Research Topic Recommendations for Students
### Targeting Q1 / Q2 / Q3 Venues | 6–12 Month Completion | Free/Lightweight Compute

> **Student constraints assumed:** 1× consumer GPU (RTX 3080/4090 or equivalent), access to Google Colab / Kaggle free tier, 6–12 months of part-time research. All topics build on open-source codebases and free datasets.

---

## How to Read This Document

Each topic lists:
- **The gap** — what's broken or missing in the current literature
- **Your novelty** — what contribution you'd make
- **Venue target** — realistic publication target
- **Codebase to build on** — don't start from scratch
- **Datasets** — free, large-scale, widely used
- **Compute estimate** — hours per experiment, RAM/VRAM needed
- **Risk** — competition density and difficulty

---

## Free Compute Resources for Students

| Resource | GPU | VRAM | Free? | Notes |
|---|---|---|---|---|
| Google Colab (free) | T4 | 16 GB | ✅ | ~4–6h sessions; limited |
| Kaggle Notebooks | T4 × 2 | 16 GB | ✅ | 30h/week; best free option |
| Google Colab Pro | A100 | 40 GB | 💲 ~$10/mo | Best value for students |
| Paperspace Gradient | A4000 | 24 GB | ✅ (free tier) | Compute units limited |
| University HPC | A100/V100 | 40–80 GB | ✅ | Ask your advisor |
| Lambda Labs | A100 | 40 GB | 💲 ~$1.10/hr | Pay-as-you-go, affordable |

**Most 3DGS topics below require only ~4–8h per experiment on a T4/A100. Kaggle free tier is sufficient for Q2/Q3 topics.**

---

---

# Q1-Tier Topics
### Target venues: CVPR, ICCV, NeurIPS, SIGGRAPH, ICLR
### Acceptance rate: ~25% | Impact: Very high

---

## 🔴 Q1-A: Multi-View Consistent Language-Guided 3DGS Editing

### The Gap
Language-guided 3D scene editing (e.g., "make the chair red", "add snow to the floor") is one of the most demanded applications of 3DGS. Current methods like GaussianEditor and InstructNeRF2NeRF apply 2D diffusion edits view-by-view, causing **critical multi-view inconsistency**: the edited chair looks red from the front but orange from the side. The underlying issue is that 2D diffusion models have no 3D awareness, so edited images from different viewpoints do not agree geometrically or photometrically.

Additionally, editing is stuck at 512–768px resolution (InstructPix2Pix's limit), and global scene edits (change the lighting throughout, repaint all walls) remain completely unsolved.

### Your Novelty
Propose a 3D-aware editing pipeline that **lifts 2D diffusion edits into Gaussian space** with multi-view consistency enforcement. Specifically:
1. Use a multi-view diffusion model (e.g., Zero123++, SyncDreamer) to generate geometrically consistent edits across selected viewpoints simultaneously
2. Distill these multi-view edits back into Gaussian attributes via a consistency-regularized optimization
3. Evaluate on a new benchmark that measures per-edit multi-view consistency (novel metric)

### Venue Target
**CVPR / ICCV** (primary), ECCV (backup)

### Build On
- [GaussianEditor](https://buaacyw.github.io/gaussian-editor/) — open source, PyTorch
- [InstructNeRF2NeRF](https://instruct-nerf2nerf.github.io/) — reference prior work
- [LangSplat](https://langsplat.github.io/) — for semantic grounding
- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus) — multi-view diffusion backbone

### Datasets
| Dataset | Scenes | Size | Access |
|---|---|---|---|
| Mip-NeRF 360 | 9 real scenes | ~5 GB | Free download |
| NeRF Synthetic (Blender) | 8 synthetic objects | ~2 GB | Free download |
| LERF dataset | 15+ scenes | ~3 GB | Free download |
| Instruct-NeRF2NeRF scenes | 10+ scenes | ~2 GB | Free download |

### Compute Estimate
- Per scene training: ~1h on T4 (standard 3DGS), 30min on A100
- Editing run: ~30min per edit on T4
- Full experiment suite: ~40–60 GPU-hours total (manageable on Kaggle free tier)

### Risk Assessment
- **Competition:** High — LangSplatV2 appeared July 2025; the editing space is hot
- **Difficulty:** Medium — building on well-established pipelines
- **Key differentiator:** Multi-view consistency has not been solved; strong clear benchmark contribution

---

## 🔴 Q1-B: Efficient Sparse-View Feed-Forward 3DGS Reconstruction

### The Gap
Standard 3DGS requires 50–200+ calibrated input images and 1–4 hours of per-scene optimization. Feed-forward methods (PixelSplat, GS-LRM) bypass this with a single forward pass through a neural network, but they have critical gaps:
- GS-LRM needs 2–4 posed (calibrated) images — still requires SfM preprocessing
- Quality degrades sharply on complex real-world scenes (outdoor, non-Lambertian materials)
- Models are heavy (transformer-based, >1B parameters), limiting practical use
- No method works well with as few as 1–2 images on in-the-wild content

### Your Novelty
Design a **lightweight, generalizable feed-forward 3DGS model** that:
1. Works from 1–3 images without pre-calibrated poses (using relative pose estimation)
2. Uses an efficient CNN-Transformer hybrid (e.g., EfficientViT backbone) instead of full ViT-L — reducing model size 5×
3. Incorporate depth priors from a monocular depth estimator (Depth Anything v2) to improve geometry

This targets the "lightweight" angle that GS-LRM and PixelSplat don't prioritize.

### Venue Target
**CVPR / ICCV** (primary), ECCV (backup)

### Build On
- [PixelSplat](https://github.com/dcharatan/pixelsplat) — open source, MIT license
- [GS-LRM](https://github.com/google-research/projects/tree/main/gs-lrm) — reference architecture
- [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2) — monocular depth prior
- [gsplat](https://github.com/nerfstudio-project/gsplat) — NVIDIA's efficient 3DGS library

### Datasets
| Dataset | Scale | Notes | Access |
|---|---|---|---|
| RealEstate10K | 10M+ frames, 80K+ videos | Indoor/outdoor scenes from YouTube | Free |
| CO3Dv2 | 1.5M frames, 50 categories | Object-centric, 360° | Free |
| MVImgNet | 220K multi-view images | 238 object categories | Free |
| ScanNet | 2.5M images, 1,513 scans | Indoor reconstruction | Free (academic) |
| DL3DV-10K | 140K scenes | Recent large-scale benchmark | Free |

### Compute Estimate
- Training a lightweight model: ~3–5 days on single A100 (use Colab Pro, ~$50 total)
- Inference/evaluation: <1min per scene
- Ablations: 10–20 GPU-hours

### Risk Assessment
- **Competition:** Very high — hot area, many groups active
- **Difficulty:** High — requires training generalizable model, not just per-scene tuning
- **Key differentiator:** "Lightweight + pose-free" angle not yet fully explored
- **Mitigation:** Focus on the efficiency + pose-free contribution as a clear gap

---

## 🔴 Q1-C: 3DGS for Long-Duration Dynamic Scene Reconstruction

### The Gap
4DGS methods handle clips of seconds to ~1 minute. Longer sequences (sports highlights, surgery videos, concerts) are intractable because Gaussian counts multiply with time. The 2024 Temporal Gaussian Hierarchy paper addresses this but is limited in quality and doesn't generalize to monocular (single camera) input — it assumes multi-camera rigs.

For students: monocular casual video is the sweet spot — everyone has a phone but no multi-camera rig.

### Your Novelty
A **temporally compact dynamic 3DGS** that reconstructs minute-long monocular videos by:
1. Separating static background (static Gaussians, no temporal cost) from dynamic foreground (time-varying)
2. Using a learned **deformation field** shared across a scene rather than per-frame Gaussians
3. Leveraging optical flow and monocular depth priors to initialize and constrain the deformation

### Venue Target
**CVPR / SIGGRAPH** (primary), ECCV / AAAI (backup)

### Build On
- [4DGS](https://github.com/hustvl/4DGaussians) — open source
- [Deformable 3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians) — open source
- [MoDGS](https://github.com/enrollie/MoDGS) — monocular dynamic baseline

### Datasets
| Dataset | Content | Access |
|---|---|---|
| HyperNeRF | Casual dynamic captures | Free |
| D-NeRF Synthetic | Synthetic animated scenes | Free |
| iPhone Dataset (Gao et al.) | Casual monocular videos | Free |
| Nvidia Dynamic Scenes | Multi-view dynamic | Free |
| PointOdyssey | Long dynamic trajectories | Free |

### Compute Estimate
- Training: 2–4h per scene on RTX 3080 (dynamic scenes are longer but the model is compact)
- Full paper experiments: ~80–100 GPU-hours

### Risk Assessment
- **Competition:** Medium — still room for strong monocular contribution
- **Difficulty:** High — dynamic reconstruction from monocular is genuinely hard
- **Key differentiator:** Minute-long + monocular is an underexplored combo

---
---

# Q2-Tier Topics
### Target venues: ECCV, AAAI, ACM MM, 3DV, BMVC, SIGGRAPH Asia
### Acceptance rate: ~27% | Impact: High

---

## 🟡 Q2-A: Perceptually-Guided Compression for Mobile 3DGS

### The Gap
3DGS compression (Compact3DGS, EAGLES, LightGaussian) optimizes for PSNR/SSIM — metrics that correlate poorly with human perception. A 10× compressed model may look visually identical in smooth regions but introduce obvious "floaters" (stray Gaussians in empty space) and loss of fine detail that PSNR misses entirely. No compression method uses a perceptual loss that is specific to Gaussian artifacts.

Additionally, Mobile-GS (March 2026) achieves 4.8 MB + 1098 FPS but was evaluated only on standard benchmarks — no user study, no perceptual metric, no web delivery pipeline assessment.

### Your Novelty
1. **Gaussian-specific perceptual metric**: Train a lightweight metric network that predicts human preference between compressed Gaussian scenes, using a collected comparison dataset (you collect ~500–1000 pairwise comparisons via crowdsourcing — free via Amazon Mechanical Turk student credit or Prolific)
2. **Metric-guided pruning**: Use the trained metric as a differentiable loss to guide which Gaussians to prune (preserve perceptually important ones: object boundaries, sharp edges, foreground)
3. Demonstrate that metric-guided compression achieves better human ratings at same file size

### Venue Target
**ECCV / ACM MM** (primary), SIGGRAPH Asia / BMVC (backup)

### Build On
- [gsplat](https://github.com/nerfstudio-project/gsplat) — modular, easy to plug in custom losses
- [LightGaussian](https://github.com/VITA-Group/LightGaussian) — pruning baseline
- [Compact3DGS](https://github.com/maincold2/Compact-3DGS) — compression baseline

### Datasets
| Dataset | Scenes | Notes | Access |
|---|---|---|---|
| NeRF Synthetic | 8 objects | Standard benchmark | Free |
| Mip-NeRF 360 | 9 real scenes | Standard benchmark | Free |
| Tanks & Temples | 4 scenes | Standard benchmark | Free |
| Deep Blending | 2 scenes | Standard benchmark | Free |

### Compute Estimate
- Training per scene: 30min on T4 (standard 3DGS is the base; compression adds ~10min)
- Human study: ~20h of your time to design + analyze; no GPU needed
- Total GPU: ~20–30 hours (Kaggle free tier easily covers this)

### Risk Assessment
- **Competition:** Medium — perceptual metric for 3DGS is an identified gap
- **Difficulty:** Low-Medium — builds on well-understood baselines
- **Key differentiator:** The human study is the contribution; GPU compute is minimal

---

## 🟡 Q2-B: Semantic 3DGS SLAM with Open-Vocabulary Scene Understanding

### The Gap
Current Gaussian SLAM systems (SplaTAM, MonoGS, GS-SLAM) produce photorealistic maps but have no semantic understanding — they cannot answer "where is the chair?" or "show me all exits." LangSplat adds language to static Gaussian scenes but not to SLAM (online, incremental mapping). The only related work (LEG-SLAM, June 2026) is very recent and preliminary.

This means robots and AR devices using Gaussian SLAM are visually rich but semantically blind.

### Your Novelty
Extend a monocular Gaussian SLAM system with **real-time open-vocabulary querying** by:
1. Incrementally distilling CLIP features into new Gaussians as they are added to the map
2. Using a compressed feature representation (follow LangSplatV2's approach: autoencoder for CLIP embeddings) to keep memory per-Gaussian small
3. Enable natural language spatial queries at 30+ FPS on a consumer GPU

### Venue Target
**ECCV / IROS / ICRA** (primary), 3DV / BMVC (backup)

### Build On
- [MonoGS](https://github.com/muskie82/MonoGS) — monocular RGB SLAM, open source
- [Splat-SLAM](https://github.com/eriksandstroem/Splat-SLAM) — RGB-only SLAM, open source
- [LangSplatV2](https://langsplat-v2.github.io/) — compressed CLIP embedding, open source

### Datasets
| Dataset | Sequences | Type | Access |
|---|---|---|---|
| Replica | 18 scenes | Indoor synthetic, RGBD | Free |
| TUM RGB-D | 39 sequences | Real indoor, Kinect | Free |
| ScanNet | 1,513 scans | Real indoor, RGBD | Free (academic) |
| HM3D-Semantics | 3,000+ scans | Semantic-rich indoor | Free (academic) |

### Compute Estimate
- Per-sequence training: 20–40min on T4
- Full evaluation suite: ~30–40 GPU-hours
- Entirely feasible on Kaggle free tier + occasional Colab Pro session

### Risk Assessment
- **Competition:** Low-Medium — LEG-SLAM appeared June 2026 but is preliminary; open-vocabulary SLAM is underexplored
- **Difficulty:** Medium — integration challenge, but each component (SLAM, CLIP distillation) is well-understood separately
- **Advantage:** Robotics + SLAM community rewards practical system contributions

---

## 🟡 Q2-C: 3DGS for Transparent and Refractive Objects

### The Gap
Standard 3DGS assumes Lambertian or mildly specular surfaces. Transparent objects (glass, water, plastic containers) violate this because they involve **refraction** — light bends as it passes through, creating complex caustics and view-dependent effects that spherical harmonics cannot model. TransparentGS (April 2025) is a first step but does not model refraction physically, only approximates it. No method handles **scenes with mixed transparent and opaque objects** efficiently.

### Your Novelty
A physically-grounded extension of 3DGS for transparent objects:
1. Add a **refraction Gaussian** type that carries an index-of-refraction (IOR) parameter and traces secondary rays through transparent primitives
2. Use a lightweight ray-bending formulation (first-order approximation) that maintains rasterization speed
3. Benchmark on a new dataset of common transparent objects (you capture it — bottles, glasses, vases — with a smartphone)

### Venue Target
**ECCV / ICCV** (primary), SIGGRAPH Asia (backup)

### Build On
- Original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) — modify the renderer
- [TransparentGS](https://arxiv.org/pdf/2504.18768) — most recent baseline
- [3DGUT](https://arxiv.org/abs/2402.11755) — NVIDIA's distorted camera + secondary rays extension

### Datasets
| Dataset | Content | Access |
|---|---|---|
| Shiny dataset (NeRF++) | Reflective/transparent objects | Free |
| Trans10K | 10,428 transparent images | Free |
| You capture | Bottles, glasses, jars | Smartphone, free |
| NeRF Synthetic subset | Glass ball scene | Free |

### Compute Estimate
- Standard 3DGS training: ~1h per scene on T4
- Secondary ray overhead: ~2× compute → ~2h per scene
- Total: ~40–60 GPU-hours for full paper

### Risk Assessment
- **Competition:** Medium — Reflective Gaussian Splatting (ICLR 2025) covers reflections; refraction is less explored
- **Difficulty:** Medium-High — modifying the rasterizer requires CUDA knowledge
- **Key differentiator:** Physical refraction, not just appearance matching; custom dataset is a contribution

---

## 🟡 Q2-D: 3DGS-Based Surgical Scene Reconstruction with Tissue Deformation

### The Gap
SurgicalGS and EndoGaussian reconstruct endoscopic scenes but assume near-rigid tissue. Real surgical scenes involve **breathing-induced non-rigid tissue deformation** (EndoFlow-SLAM 2025 addresses this with optical flow) **and** strong specular highlights from the endoscope light. No single method handles all three simultaneously: deformation + specular suppression + real-time performance.

This is a niche venue (MICCAI) with a very active but small community. A clear contribution on tissue deformation + lighting jointly is very publishable.

### Your Novelty
A **deformable 3DGS framework for endoscopy** that:
1. Models tissue deformation with a lightweight deformation field (MLP or 4DGS-style) conditioned on time
2. Separates specular highlights from Lambertian tissue using a physically-grounded decomposition (inspired by PR-ENDO)
3. Runs in real-time (30+ FPS) for intraoperative use

### Venue Target
**MICCAI** (Q1 in medical imaging — very high impact factor in its domain), Medical Image Analysis journal

### Build On
- [EndoGaussian](https://github.com/yifliu3/EndoGaussian) — open source
- [PR-ENDO](https://arxiv.org/pdf/2411.12510) — relightable endoscopic 3DGS
- [Deformable 3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians) — deformation field

### Datasets
| Dataset | Content | Size | Access |
|---|---|---|---|
| SCARED | Stereo endoscopy, 7 datasets | ~50 GB | Free (Hamlyn Centre, academic) |
| C3VD | Colonoscopy, 22 videos | ~15 GB | Free |
| StereoMIS | Robotic surgery, stereo | ~20 GB | Free (academic) |
| EndoNeRF dataset | Dynamic surgical scenes | ~2 GB | Free |
| SimCol | Simulated colonoscopy | ~5 GB | Free |

### Compute Estimate
- Per-sequence training: 30–60min on T4 (surgical videos are short, 20–300 frames)
- No large-scale pretraining needed — per-scene optimization
- Total: ~20–30 GPU-hours for all experiments

### Risk Assessment
- **Competition:** Medium — MICCAI 2025 had several 3DGS endoscopy papers; but joint deformation + relighting is still open
- **Difficulty:** Medium — medical domain knowledge needed; datasets require registration
- **Big advantage:** Very low compute requirement; MICCAI has a structured, welcoming community for clear incremental contributions

---
---

# Q3-Tier Topics
### Target venues: WACV, IROS, ICRA, Computers & Graphics, IEEE Access
### Acceptance rate: ~40%+ | Best for first paper / quick publication

---

## 🟢 Q3-A: Robust Depth-Prior Integration for 3DGS with Cheap Sensors

### The Gap
iPhone-class LiDAR sensors (2–5 cm accuracy) and stereo depth cameras produce noisy, sparse depth maps. DN-Gaussian and CDGS use monocular depth estimates from neural networks (clean but scale-ambiguous). No method is specifically designed to fuse **real, noisy consumer-grade depth** (not neural depth estimates) to improve 3DGS in challenging scenes (textureless walls, reflective floors) without degrading performance on normal areas.

### Your Novelty
A **noise-robust depth fusion module** for 3DGS:
1. Learn a per-pixel confidence weighting for the depth signal based on local geometric consistency
2. Apply confidence-weighted depth supervision during Gaussian densification (don't trust noisy depths for Gaussian placement)
3. Benchmark on ARKitScenes (which has real iPhone LiDAR depth — inherently noisy)

### Venue Target
**WACV / 3DV** (primary), Computers & Graphics journal (backup)

### Build On
- [Gaussian Opacity Fields](https://github.com/autonomousvision/gaussian-opacity-fields)
- [DN-Gaussian](https://github.com/Fangkang515/DN-Gaussian)
- Original 3DGS codebase

### Datasets
| Dataset | Content | Access |
|---|---|---|
| ARKitScenes | 5,000+ iPhone LiDAR scans | Free (Apple) |
| ScanNet | 2.5M RGBD frames | Free (academic) |
| Mip-NeRF 360 | Standard benchmark | Free |

### Compute Estimate
- **Minimal** — standard 3DGS training + depth loss
- Per scene: ~1h on T4
- Total: ~15–20 GPU-hours
- **Feasible on Kaggle free tier alone**

### Risk Assessment
- **Competition:** Low — this specific angle (real sensor noise, not neural depth) is underexplored
- **Difficulty:** Low — add a confidence module to existing depth-guided 3DGS
- **Fast path:** 4–6 months from idea to submission

---

## 🟢 Q3-B: 3DGS Quality Benchmark with Perceptual Evaluation

### The Gap
Every 3DGS paper reports PSNR, SSIM, and LPIPS. But the 2025 perceptual study from SIGGRAPH found these metrics frequently disagree with human preference. There is **no standardized perceptual benchmark** for 3DGS that measures the artifacts humans actually care about: floaters, view-dependent popping, temporal jitter in dynamic scenes.

A benchmark paper that: (a) collects human preference data, (b) trains a 3DGS-specific quality predictor, and (c) evaluates all major 3DGS methods is a publishable, high-citation contribution.

### Your Novelty
1. Design a subjective study (pairwise comparisons) covering 5 artifact types: floaters, edge sharpness, specular accuracy, temporal consistency (4DGS), and mobile degradation
2. Collect ~2,000 pairwise judgments (Amazon Mechanical Turk student credit: ~$100–200)
3. Train a lightweight perceptual metric network (distillation from LPIPS backbone, fine-tuned on your data)
4. Re-rank 15+ published 3DGS methods under the new metric — results will differ from PSNR rankings, which is your main finding

### Venue Target
**WACV / Computers & Graphics / IEEE TVCG** (primary)

### Build On
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) — backbone to fine-tune
- Existing 3DGS codebases — run all baselines, don't need to implement them

### Datasets
Use outputs from **existing 3DGS methods** run on Mip-NeRF 360 + NeRF Synthetic + Tanks & Temples. You generate the comparison images; no new training data needed.

### Compute Estimate
- Run 15 existing methods: ~150–200 GPU-hours total (staggered over time on Kaggle)
- Training the metric: ~5 GPU-hours
- Human study: $100–200 budget, no GPU

### Risk Assessment
- **Competition:** Very low — benchmark papers for 3DGS are rare
- **Difficulty:** Low — no novel algorithm; contribution is the dataset and analysis
- **Risk:** May be seen as incremental; needs strong analysis and clear findings
- **Advantage:** High citation potential; everyone uses benchmarks

---

## 🟢 Q3-C: Stylized 3DGS for Artistic Scene Rendering

### The Gap
Neural style transfer applies 2D styles to images. For 3DGS, naively applying style per-view causes severe multi-view inconsistency (different style on each viewpoint). StyleGaussian (2024) addresses this but is slow and lacks control over *where* style is applied (e.g., "apply Van Gogh style only to the background, not the person").

### Your Novelty
A **spatially-controlled 3D style transfer** for Gaussian scenes:
1. Use LangSplat-style semantic masks to identify user-specified regions ("background," "walls")
2. Apply style optimization only to Gaussians in those regions via CLIP-based style loss
3. Maintain photorealism in non-stylized regions
4. Fast enough for interactive editing (< 5 minutes per edit)

### Venue Target
**ACM MM / WACV** (primary), Computers & Graphics (backup)

### Build On
- [StyleGaussian](https://github.com/KU-CVLAB/StyleGaussian)
- [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) — for region segmentation
- [LangSplat](https://langsplat.github.io/) — for language-guided masking

### Datasets
| Dataset | Content | Access |
|---|---|---|
| Mip-NeRF 360 | 9 real scenes | Free |
| NeRF Synthetic | 8 objects | Free |
| WikiArt | 80K+ art style images | Free |

### Compute Estimate
- Very low — no large-scale training, just scene-level optimization
- Per scene: ~1h on T4 for training + ~5min for style transfer
- Total: ~10–15 GPU-hours

### Risk Assessment
- **Competition:** Low — spatially controlled 3D style transfer is underexplored
- **Difficulty:** Low — well-understood component combination
- **Fast path:** Could be done in 4–5 months

---

---

# Decision Framework: Which Topic Is Right for You?

```
START: What's your situation?
│
├─► First paper, risk-averse → Q3-A (Depth Sensor Fusion) or Q3-C (Stylized 3DGS)
│   Fast, clear contribution, manageable in 4–6 months
│
├─► Advisor wants Q1/Q2 in top CV venue → Q1-A (Language Editing) or Q2-B (Semantic SLAM)
│   Higher risk but stronger trajectory; needs 10–12 months
│
├─► Interest in medical/biomedical applications → Q2-D (Surgical Endoscopy)
│   MICCAI is Q1 in medical imaging; lower compute; smaller competition pool
│
├─► You have coding + CUDA skills → Q2-C (Transparent Surfaces)
│   Requires modifying the 3DGS CUDA rasterizer; strong technical contribution
│
├─► You have a dataset advantage (medical partner, unique capture setup) → Q2-D or Q2-C
│
└─► You want broad impact + citations for future research → Q3-B (Benchmark)
    Requires budget (~$200) not compute; high citation potential
```

---

# Summary Table

| ID | Topic | Venue | Compute (GPU-hrs) | Timeline | Novelty Type | Risk |
|---|---|---|---|---|---|---|
| **Q1-A** | Multi-view consistent language editing | CVPR/ICCV | ~50 | 10–12 mo | Algorithm + Benchmark | High |
| **Q1-B** | Lightweight feed-forward sparse-view | CVPR/ICCV | ~200 (training) | 12 mo | Architecture | Very High |
| **Q1-C** | Long-duration monocular 4DGS | CVPR/SIGGRAPH | ~100 | 10–12 mo | Algorithm | High |
| **Q2-A** | Perceptual compression for mobile | ECCV/ACM MM | ~30 | 8–10 mo | Metric + Algorithm | Medium |
| **Q2-B** | Semantic SLAM with open vocabulary | ECCV/IROS | ~40 | 8–10 mo | System | Medium |
| **Q2-C** | Transparent/refractive surfaces | ECCV/ICCV | ~60 | 10–12 mo | Algorithm + Dataset | Medium-High |
| **Q2-D** | Surgical endoscopy + deformation | MICCAI | ~25 | 6–9 mo | System + Algorithm | Medium |
| **Q3-A** | Depth sensor noise robustness | WACV/3DV | ~20 | 4–6 mo | Algorithm | Low |
| **Q3-B** | Perceptual benchmark | WACV/TVCG | ~200 (baselines) | 6–8 mo | Benchmark | Low |
| **Q3-C** | Spatially-controlled style transfer | ACM MM/WACV | ~15 | 4–5 mo | Application | Low |

---

# Recommended Starting Point

**For most students: Start with Q2-D (Surgical Endoscopy) or Q2-A (Compression).**

- Q2-D is high-impact in a focused community, has the lowest compute cost, has clear open problems, and MICCAI has a constructive review culture. If you have any access to a medical school or hospital, a dataset collaboration can elevate this to Q1.
- Q2-A is technically straightforward, has a clear novel contribution (perceptual metric), and the human study component differentiates it from pure algorithm papers.

Both can realistically be submitted within 8–10 months on student hardware.

---

## Essential Open-Source Codebases

| Codebase | What it covers | Language | Stars |
|---|---|---|---|
| [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Original INRIA implementation | CUDA/Python | ~15K |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Modular, pip-installable, best for research | CUDA/Python | ~5K |
| [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | Framework with 3DGS support, easy baselines | Python | ~10K |
| [4DGaussians](https://github.com/hustvl/4DGaussians) | Dynamic 4DGS | CUDA/Python | ~2K |
| [LightGaussian](https://github.com/VITA-Group/LightGaussian) | Compression baseline | Python | ~1K |
| [MonoGS](https://github.com/muskie82/MonoGS) | Monocular SLAM | CUDA/Python | ~1K |
| [LangSplat](https://langsplat.github.io/) | Language-embedded 3DGS | Python | ~2K |

---

## Sources

- [Kerbl, "The Impact and Outlook of 3D Gaussian Splatting," arXiv:2510.26694](https://arxiv.org/pdf/2510.26694)
- [Lightweight 3DGS Compression via Video Codec, arXiv:2512.11186](https://arxiv.org/pdf/2512.11186)
- [Mobile-GS: Real-time Gaussian Splatting for Mobile Devices, arXiv:2603.11531](https://arxiv.org/pdf/2603.11531)
- [TransparentGS: Fast Inverse Rendering of Transparent Objects, arXiv:2504.18768](https://arxiv.org/pdf/2504.18768)
- [Reflective Gaussian Splatting, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/abf3682c9cf9245a0294a4bebe4544ff-Paper-Conference.pdf)
- [LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS](https://arxiv.org/html/2507.07136v1)
- [EndoPlanar: Deformable Planar-based GS for Surgical Scene Reconstruction, MICCAI 2025](https://papers.miccai.org/miccai-2025/0295-Paper3722.html)
- [EndoFlow-SLAM, MICCAI 2025](https://papers.miccai.org/miccai-2025/0290-Paper3495.html)
- [SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM](https://spla-tam.github.io/)
- [Sparse-View 3D Reconstruction: Recent Advances and Open Challenges, arXiv:2507.16406](https://arxiv.org/pdf/2507.16406)
- [Awesome-Feed-Forward-3D (GitHub)](https://github.com/ziplab/Awesome-Feed-Forward-3D)
- [Compression in 3DGS: A Survey, arXiv:2502.19457](https://arxiv.org/pdf/2502.19457)
- [CLM: Removing the GPU Memory Barrier for 3DGS, arXiv:2511.04951](https://arxiv.org/pdf/2511.04951)

---

# SOTA Papers and Projects to Start Reading

> Organized by area. Start with the **Foundational** section, then branch into your topic of interest. Links go directly to paper PDFs or project pages (some may be behind paywalls).

---

## Foundational — Read These First

| Paper | Venue | Link |
|---|---|---|
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al.) | SIGGRAPH 2023 | [Project + Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) |
| The Impact and Outlook of 3D Gaussian Splatting (Kerbl) | arXiv 2025 | [PDF](https://arxiv.org/pdf/2510.26694) |
| Mip-Splatting: Alias-free 3D Gaussian Splatting (Yu et al.) | CVPR 2024 Best Student Paper | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf) · [GitHub](https://github.com/autonomousvision/mip-splatting) |
| 2D Gaussian Splatting for Geometrically Accurate Radiance Fields (Huang et al.) | SIGGRAPH 2024 | [PDF](https://arxiv.org/pdf/2403.17888) |
| SuGaR: Surface-Aligned Gaussian Splatting for Mesh Reconstruction (Guédon & Lepetit) | CVPR 2024 | [arXiv](https://arxiv.org/abs/2311.12775) · [GitHub](https://github.com/Anttwo/SuGaR) |
| gsplat: An Open-Source Library for Gaussian Splatting | JMLR 2025 | [PDF](https://arxiv.org/pdf/2409.06765) · [GitHub](https://github.com/nerfstudio-project/gsplat) |

---

## Feed-Forward / Sparse-View Reconstruction (→ Q1-B)

| Paper | Venue | Link |
|---|---|---|
| pixelSplat: 3D Gaussian Splats from Image Pairs (Charatan et al.) | CVPR 2024 | [arXiv](https://arxiv.org/abs/2312.12337) · [GitHub](https://github.com/dcharatan/pixelsplat) |
| GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting (Zhang et al.) | ECCV 2024 | [arXiv](https://arxiv.org/abs/2404.19702) · [Project](https://sai-bi.github.io/project/gs-lrm/) |
| SHARP: Monocular View Synthesis in Less Than a Second (Apple) | arXiv 2025 | [arXiv](https://arxiv.org/abs/2512.10685) · [GitHub](https://github.com/apple/ml-sharp) |
| AnySplat: Feed-forward 3DGS from Unconstrained Views | SIGGRAPH Asia 2025 (ACM TOG) | [arXiv](https://arxiv.org/abs/2505.23716) · [GitHub](https://github.com/InternRobotics/AnySplat) |
| DUSt3R: Geometric 3D Vision Made Easy (Leroy et al.) | CVPR 2024 | [arXiv](https://arxiv.org/abs/2312.14132) |
| MASt3R: Grounding Image Matching in 3D (Leroy et al.) | ECCV 2024 | [arXiv](https://arxiv.org/abs/2406.09756) |
| Depth Anything V2 (Yang et al.) | NeurIPS 2024 | [arXiv](https://arxiv.org/abs/2406.09414) · [GitHub](https://github.com/DepthAnything/Depth-Anything-V2) |
| DNGaussian: Sparse-View 3DGS with Depth Normalization (Li et al.) | CVPR 2024 | [arXiv](https://arxiv.org/abs/2403.06912) · [Project](https://fictionarry.github.io/DNGaussian/) |

---

## Dynamic / 4DGS (→ Q1-C)

| Paper | Venue | Link |
|---|---|---|
| 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering (Wu et al.) | CVPR 2024 | [Project](https://guanjunwu.github.io/4dgs/) · [GitHub](https://github.com/hustvl/4DGaussians) |
| Deformable 3D Gaussians for Monocular Dynamic Scene Reconstruction (Yang et al.) | CVPR 2024 | [arXiv](https://arxiv.org/abs/2309.13101) · [GitHub](https://github.com/ingra14m/Deformable-3D-Gaussians) |
| Anchored 4D Gaussian Splatting for Dynamic Novel View Synthesis | SIGGRAPH Asia 2025 | [ACM DL](https://dl.acm.org/doi/10.1145/3757377.3763898) |
| Prior-Enhanced Gaussian Splatting for Dynamic Scene from Casual Video | SIGGRAPH Asia 2025 | [ACM DL](https://dl.acm.org/doi/10.1145/3757377.3763910) |
| Dynamics-Aware Gaussian Splatting Streaming for On-the-Fly 4D Reconstruction | arXiv 2024 | [PDF](https://arxiv.org/pdf/2411.14847) |

---

## Language / Editing (→ Q1-A)

| Paper                                                                 | Venue      | Link                                                                                              |
| --------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------- |
| LangSplat: 3D Language Gaussian Splatting (Qin et al.)                | CVPR 2024  | [arXiv](https://arxiv.org/abs/2312.16084) · [Project](https://langsplat.github.io/)               |
| LangSplatV2: High-dimensional 3D Language GS with 450+ FPS            | arXiv 2025 | [HTML](https://arxiv.org/html/2507.07136v1)                                                       |
| GaussianEditor: Swift and Controllable 3D Editing (Chen et al.)       | CVPR 2024  | [arXiv](https://arxiv.org/abs/2311.14521) · [Project](https://buaacyw.github.io/gaussian-editor/) |
| Gaussian Grouping: Segment and Edit Anything in 3D Scenes (Ye et al.) | ECCV 2024  | [arXiv](https://arxiv.org/abs/2312.00732) · [GitHub](https://github.com/lkeab/gaussian-grouping)  |

---

## Compression / Mobile (→ Q2-A)

| Paper | Venue | Link |
|---|---|---|
| LightGaussian: Unbounded 3D Gaussian Compression (Fan et al.) | arXiv 2023 | [PDF](https://arxiv.org/pdf/2311.17245) · [GitHub](https://github.com/VITA-Group/LightGaussian) |
| Compact 3D Gaussian Representation for Radiance Field (Lee et al.) | CVPR 2024 | [arXiv](https://arxiv.org/abs/2311.13681) · [GitHub](https://github.com/maincold2/Compact-3DGS) |
| EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS | ECCV 2024 | [arXiv](https://arxiv.org/abs/2312.04564) · [Project](https://efficientgaussian.github.io/) |
| Compression in 3DGS: A Survey (2025) | arXiv 2025 | [PDF](https://arxiv.org/pdf/2502.19457) |
| 3DGS.zip: Survey on 3DGS Compression Methods (2025) | Computer Graphics Forum | [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70078) |
| Lightweight 3DGS Compression via Video Codec | arXiv 2025 | [PDF](https://arxiv.org/pdf/2512.11186) |

---

## SLAM (→ Q2-B)

| Paper | Venue | Link |
|---|---|---|
| SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM | CVPR 2024 | [Project](https://spla-tam.github.io/) |
| MonoGS: Monocular RGB Gaussian SLAM | arXiv 2024 | [GitHub](https://github.com/muskie82/MonoGS) |
| MonoGS++: Fast and Accurate Monocular RGB Gaussian SLAM | arXiv 2025 | [HTML](https://arxiv.org/html/2504.02437v1) |
| Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians | arXiv 2024 | [arXiv](https://arxiv.org/abs/2405.16544) · [GitHub](https://github.com/google-research/Splat-SLAM) |
| RP-SLAM: Real-time Photorealistic SLAM with Efficient 3DGS | arXiv 2024 | [PDF](https://arxiv.org/pdf/2412.09868) |

---

## Transparent / Refractive Surfaces (→ Q2-C)

| Paper | Venue | Link |
|---|---|---|
| TransparentGS: Fast Inverse Rendering of Transparent Objects | arXiv 2025 | [PDF](https://arxiv.org/pdf/2504.18768) |
| 3DGUT: Distorted Cameras + Secondary Rays (NVIDIA) | arXiv 2024 | [arXiv](https://arxiv.org/abs/2402.11755) |
| Reflective Gaussian Splatting | ICLR 2025 | [PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/abf3682c9cf9245a0294a4bebe4544ff-Paper-Conference.pdf) |

---

## Surgical / Medical (→ Q2-D)

| Paper | Venue | Link |
|---|---|---|
| EndoGaussian: Real-time GS for Dynamic Endoscopic Scene Reconstruction | TMI 2025 | [GitHub](https://github.com/CUHK-AIM-Group/EndoGaussian) |
| SurgicalGS: Dynamic 3DGS for Robotic-Assisted Surgical Scene Reconstruction | MICCAI 2025 | [Springer](https://link.springer.com/chapter/10.1007/978-3-032-05141-7_55) |
| EndoPlanar: Deformable Planar-based GS for Surgical Scene Reconstruction | MICCAI 2025 | [Paper](https://papers.miccai.org/miccai-2025/0295-Paper3722.html) |
| EndoFlow-SLAM: Real-Time Endoscopic SLAM with Flow-Constrained GS | MICCAI 2025 | [Paper](https://papers.miccai.org/miccai-2025/0290-Paper3495.html) |
| PR-ENDO: Relightable Endoscopic 3DGS | arXiv 2024 | [PDF](https://arxiv.org/pdf/2411.12510) |

---

## Style Transfer (→ Q3-C)

| Paper | Venue | Link |
|---|---|---|
| StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting (Liu et al.) | SIGGRAPH Asia 2024 | [PDF](https://arxiv.org/pdf/2403.07807) · [GitHub](https://github.com/Kunhao-Liu/StyleGaussian) |
| SGSST: Scaling Gaussian Splatting Style Transfer | arXiv 2024 | [PDF](https://arxiv.org/pdf/2412.03371) |

---

## Curated GitHub Lists (keep bookmarked)

- [Awesome 3D Gaussian Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) — comprehensive paper tracker updated weekly
- [Awesome 3DGS Applications (TPAMI 2026 survey)](https://github.com/heshuting555/Awesome-3DGS-Applications) — segmentation, editing, generation
- [Awesome Feed-Forward 3D](https://github.com/ziplab/Awesome-Feed-Forward-3D) — feed-forward reconstruction methods
- [Awesome 3DGS SLAM](https://github.com/KwanWaiPang/Awesome-3DGS-SLAM) — SLAM-specific paper list
