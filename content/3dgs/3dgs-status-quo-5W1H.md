# 3D Gaussian Splatting: Status Quo — A 5W1H Deep Research Report

> **As of June 2026.** Based on multi-source research covering academic papers, industry announcements, and practitioner guides.

---

## WHAT — What Is 3D Gaussian Splatting?

3D Gaussian Splatting (3DGS) is a **real-time, photorealistic 3D scene representation and rendering technique** that models scenes as collections of millions of anisotropic 3D Gaussian ellipsoids. Each Gaussian is parameterized by:

- **Position** (3D mean, μ)
- **Covariance** (shape/orientation, decomposed into rotation matrix R and scale S)
- **Opacity** (α)
- **View-dependent color** (encoded via spherical harmonics coefficients)

This is fundamentally different from both traditional mesh-based rendering (polygons + textures) and Neural Radiance Fields (NeRFs, which use ray-marching through a neural network). 3DGS uses **differentiable rasterization**: each Gaussian is projected ("splatted") onto the 2D image plane, sorted by depth, and alpha-blended front-to-back. Because this is a GPU-friendly rasterization operation — not an expensive ray-march through a neural network — it achieves real-time rendering rates.

### Core Variants

| Variant                 | What it adds                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| **Static 3DGS**         | Original formulation; assumes a fixed scene                                              |
| **4DGS / Dynamic 3DGS** | Adds temporal coherence; Gaussians move and deform over time                             |
| **Compressed 3DGS**     | Quantized/codebook-compressed attributes for edge deployment                             |
| **Feed-forward 3DGS**   | A neural network predicts Gaussian parameters directly, bypassing per-scene optimization |
| **Generative 3DGS**     | AI generates complete Gaussian scenes from text, image, or video prompts                 |

---

## WHO — Who Built It, Who Uses It?

### Origins

3DGS was introduced by **Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis** at INRIA (France) and published at **ACM SIGGRAPH 2023** in July 2023. Kerbl has since moved to TU Wien, where active development continues.

### Key Research Groups

- **INRIA / TU Wien** — original authors, ongoing theoretical work
- **NVIDIA Research** — physical AI simulations, 3DGUT distortion correction
- **Google DeepMind** — Brush (in-browser WebGPU training)
- **Apple Machine Learning Research** — SHARP (instant 3DGS from single image)
- **World Labs** (co-founded by Fei-Fei Li) — Spark renderer, Marble generative 3DGS
- **Academic groups worldwide** — CVPR, ECCV, ICCV, SIGGRAPH are publishing hundreds of 3DGS extensions per year 

### Industry Adopters (2025–2026)

| Sector             | Company / Tool              | Application                                                                          |
| ------------------ | --------------------------- | ------------------------------------------------------------------------------------ |
| Film/VFX           | Framestore                  | Dynamic 4DGS in *Superman* (2025) — first major motion picture                       |
| Rendering          | OTOY                        | OctaneRender 2026: full path-traced Gaussian splat support                           |
| VFX Tools          | The Foundry                 | Nuke 17.0 open beta: native 3DGS import and manipulation                             |
| Real Estate        | Zillow                      | SkyTours — first major real estate platform to ship 3DGS                             |
| Real Estate        | Apartments.com / Matterport | 3D Exteriors — exterior 3DGS via Matterport                                          |
| Maps               | Apple                       | Apple Maps 3DGS Flyover — announced WWDC June 2026 (300+ locations)                  |
| VR/XR              | Meta                        | Quest: integrated splat viewing                                                      |
| Social VR          | VRChat                      | Community tools for hyper-realistic virtual spaces                                   |
| Standards          | OpenUSD (AOUSD)             | v26.03 (March 2026): `UsdVolParticleField3DGaussianSplat` — native USD schema        |
| Web Rendering      | World Labs                  | Spark (Three.js renderer) — named one of GitHub's most influential libraries of 2025 |
| Consumer Capture   | Polycam / Luma AI           | Mobile apps for 3DGS capture and processing                                          |
| Game Engines       | Unreal Engine / Unity       | Plugins for virtual production and game integration                                  |
| Virtual Production | Volinga / XGRIDS            | LiDAR capture → Unreal Engine VP stages in hours                                     |

---

## WHEN — Timeline and Milestones

```
2020–2022  NeRF era: photorealistic 3D but renders in seconds per frame; hours to train
           Per-scene, no real-time use

2023-Jul   Kerbl et al. publish "3D Gaussian Splatting for Real-Time Radiance Field
           Rendering" at SIGGRAPH 2023 — immediate impact across academia and industry

2023 H2    Explosion of follow-up research: dynamic 3DGS, compressed variants, SLAM,
           robotics, medical imaging, editing; >1,000 papers within 12 months

2024       Key extensions ship:
           - Dynamic 3D Gaussians (Luiten et al., 3DV 2024) — tracking identity over time
           - 4D Gaussian Splatting for real-time dynamic scenes (Wu et al., CVPR 2024)
           - DrivingGaussian for autonomous driving (CVPR 2024)
           - PixelSplat: feed-forward reconstruction from image pairs
           - Mip-Splatting: alias-free rendering
           - GS-LRM: 0.23s reconstruction from 2–4 images on A100
           - VRSplat: 72+ fps stereo for VR headsets

2025       "Year of production reality":
           - Superman becomes first major film with dynamic 3DGS in VFX pipeline
           - Zillow ships SkyTours (3DGS real estate)
           - The Foundry ships Nuke 17.0 open beta with native 3DGS
           - 20+ SLAM scanning companies demonstrate 3DGS at Intergeo
           - Apple releases SHARP (open-source, single image → 3DGS in <1 second)
           - World Labs' Spark library rises to GitHub's most influential list
           - SurgicalGS presented at MICCAI 2025 (robotic surgical scene reconstruction)
           - Echoes of the Coliseum: live sports 3DGS streaming demonstrated at ACM SIGGRAPH 2025

2026-Mar   OpenUSD v26.03 adds native Gaussian Splat schema (UsdVolParticleField3DGaussianSplat),
           integrating 3DGS into industry-standard VFX/animation pipeline

2026-Jun   Apple announces 3DGS in Apple Maps Flyover at WWDC 2026 (current month)
           4DGS, generative splats, and near-instant reconstruction mature as active frontiers
```

---

## WHERE — Where Is It Being Deployed?

### Publicly Deployed (Consumer / Production Use)

**Film & Virtual Production**
The 2025 film *Superman* used Houdini-based 4DGS tools in its VFX pipeline via Framestore — the first major motion picture to do so. Virtual production studios are using 3DGS to bridge real locations and LED stages, replacing multi-week photogrammetry workflows with LiDAR-captured Gaussian scenes deliverable in hours.

**Real Estate**
Zillow's SkyTours and Apartments.com's exterior 3DGS (via Matterport) bring property walkthroughs captured on smartphones to millions of users. Movement is continuous and smooth — no "teleportation" between fixed 360° nodes as in older Matterport experiences.

**Maps & Navigation**
Apple announced at WWDC 2026 (June 8, 2026) that 3DGS is coming to Apple Maps Flyover for 300+ global locations, replacing photogrammetry-based models. This is the largest consumer-scale 3DGS deployment announced to date.

**Web Browsers**
Three.js-based libraries (Spark by World Labs; Babylon.js 8.0) enable fully browser-based 3DGS viewing with no installs. WebGPU support (universally available across Chrome, Edge, Firefox, Safari as of 2026) delivers 2–5× performance over WebGL. File sizes after compression (SPZ, SOG, KSPLAT formats) are 15–100 MB depending on device target.

**Consumer Capture Tools**
Polycam and Luma AI offer mobile 3DGS capture and processing with free tiers — putting 3DGS creation in anyone's hands. DIY capture is free; processing takes minutes. The Utsubo guide benchmarks desktop rendering at 60+ FPS and mobile (iPhone 14+) at 30–45 FPS with optimized splats.

**Game Engines & XR**
Unreal Engine (XScene-UEPlugin and others on Fab marketplace) and Unity support 3DGS natively via plugins. Meta Quest shows splat content in XR contexts. VRChat supports community-built 3DGS environments.

**Industry Tools**
OTOY's OctaneRender 2026 supports path-traced Gaussian splats. The Foundry's Nuke 17.0 (open beta, late 2025) lets compositors manipulate Gaussian scenes as first-class assets. OpenUSD v26.03 means 3DGS is now part of the standard USD scene graph used across Pixar, Apple, NVIDIA, Adobe, and Autodesk pipelines.

### Active Research Deployments (Not Yet Public)

- Surgical robotics (SurgicalGS, EndoGaussian) — endoscopic scene reconstruction for robotic-assisted surgery (MICCAI 2025)
- Autonomous driving simulation (IDSplat, SplatAD, LiV-GS) — sensor simulation for perception/planning model training
- Live sports streaming volumetric video (Echoes of the Coliseum)
- Large-scale SLAM for outdoor mapping (LSG-SLAM, GARAD-SLAM)
- Robotic manipulation (ManiGaussian)

---

## WHY — Why 3DGS? Why Does It Matter?

### vs. NeRF

| Dimension | NeRF | 3DGS |
|---|---|---|
| Representation | Implicit (neural network) | Explicit (Gaussian ellipsoids) |
| Rendering speed | Seconds per frame | Real-time (60+ FPS) |
| Training time | Hours to days | Minutes to hours |
| Memory (storage) | Small (neural weights) | Medium (50–500 MB raw; 15–100 MB compressed) |
| Editability | Very limited | Limited but better |
| Web-deployable | Poor (ray-marching too slow) | Yes (WebGL/WebGPU rasterization) |
| Reflective surfaces | Good | Improving |

3DGS keeps NeRF's photorealistic quality while achieving the rendering speed of traditional rasterization pipelines. This combination — quality + speed — is what unlocked every production use case above.

### vs. Traditional Photogrammetry (Mesh + Textures)

- 3DGS captures subtle material qualities (fur, hair, translucency, reflections) that mesh reconstruction misses
- No manual cleanup of mesh artifacts or UV seams
- Capture workflow is faster (video walkthrough vs. calibrated scan stations)
- Weakness: meshes are fully editable (Boolean ops, sculpting); Gaussian scenes are not

### Why It Matters at a Macro Level

3DGS is collapsing the cost and time of creating photorealistic 3D content from weeks to hours. The economic implications span real estate, e-commerce, film, medical training, autonomous driving simulation, robotics, and cultural preservation. OpenUSD v26.03 integration (March 2026) and Apple Maps integration (June 2026) signal that 3DGS is graduating from a research curiosity to a foundational infrastructure technology.

---

## HOW — How Does It Work?

### Pipeline

```
1. CAPTURE
   ├─ Photographs or video from multiple viewpoints
   └─ Requires sufficient overlap; works with smartphone or professional rig

2. STRUCTURE FROM MOTION (SfM)
   ├─ Compute camera poses from image correspondences (e.g., COLMAP)
   └─ Produces a sparse point cloud of 3D feature points

3. INITIALIZATION
   └─ Place one Gaussian ellipsoid at each SfM point (isotropic, low opacity)

4. OPTIMIZATION (gradient descent, ~minutes–hours)
   ├─ Render scene from training camera poses using differentiable rasterizer
   ├─ Compute photometric loss (L1 + SSIM) against ground-truth images
   ├─ Backpropagate gradients through rasterizer
   └─ Update: position (μ), rotation (R), scale (S), opacity (α), color (SH coefficients)

5. DENSIFICATION
   ├─ SPLIT: large Gaussians in over-reconstructed areas → two smaller Gaussians
   ├─ CLONE: small Gaussians in under-reconstructed areas → duplicate nearby
   └─ PRUNE: near-transparent Gaussians (α below threshold) are removed

6. RENDERING (real-time, GPU rasterization)
   ├─ Project each Gaussian to 2D ellipse from camera viewpoint
   ├─ Sort all Gaussians by depth (back-to-front)
   └─ Alpha-composite front-to-back: C = Σ cᵢαᵢ ∏ⱼ﹤ᵢ(1−αⱼ)
```

### Feed-Forward / Instant Reconstruction (research frontier)

Instead of per-scene optimization, a neural network trained across many scenes can predict Gaussian parameters directly from input images:

- **PixelSplat** (CVPR 2024): 3DGS scene from an image *pair* in milliseconds
- **GS-LRM** (ECCV 2024): 3DGS from 2–4 posed images in **~0.23 seconds** on A100
- **Apple SHARP**: 3DGS from a *single image* in under 1 second; open-source
- **On-the-fly reconstruction** (ACM ToG 2025): handles unposed image streams; usable scene available right after capture

### Dynamic / 4DGS

- Gaussians are given **temporal identity** — they persist and move frame-to-frame rather than being re-fit from scratch
- A **temporal Gaussian hierarchy** separates static background (reused across frames) from rapidly changing foreground (updated locally), keeping GPU memory bounded even for minute-long video

---

## What Has Been Covered vs. What Still Needs Work

### ✅ Implemented and Public (Solved Problems)

- Real-time rendering on desktop GPUs (60+ FPS, high fidelity)
- Consumer capture via smartphone + cloud processing (Polycam, Luma AI)
- VR rendering at 72+ fps stereo with artifact-free quality (VRSplat)
- Browser-based viewing with WebGPU (Spark, Babylon.js 8.0)
- Anti-aliasing / scale-aware filtering (Mip-Splatting, Multi-Scale 3DGS)
- Compression to web-deliverable sizes (SPZ, SOG, KSPLAT)
- Industry pipeline integration (OpenUSD v26.03, Nuke 17.0, OctaneRender 2026)
- Feed-forward reconstruction from a handful of images (GS-LRM, SHARP)
- Robotic surgical scene reconstruction (SurgicalGS, MICCAI 2025)
- Autonomous driving sensor simulation (SplatAD, IDSplat)
- Static SLAM for indoor environments

### 🔬 On Active Research (Not Yet Production-Ready)

| Frontier | Key Work | Status |
|---|---|---|
| **4D / Dynamic scenes at scale** | Temporal Gaussian Hierarchy, 4DGS | Research; minute-long captures tractable but not consumer-ready |
| **Generative splats** (text/image → 3DGS) | World Labs Marble, Apple SHARP | Early; SHARP is open-source; quality/controllability improving |
| **Semantic editing** | 3DSceneEditor, TRACE, GaussianEditor | Research; global edits remain hard |
| **Live volumetric streaming** | Echoes of the Coliseum (SIGGRAPH 2025) | Demonstrated for sports events; not productized |
| **Large-scale outdoor SLAM** | LSG-SLAM, LiV-GS, GS-Scale | Active; memory and consistency challenges remain |
| **Medical: endoscopic reconstruction** | SurgicalGS, EndoGaussian | Research; narrow field-of-view and deformable tissue are hard |
| **Robotic manipulation** | ManiGaussian | Research; real-world noise and partial observability are unsolved |
| **Mobile training** | Brush (Google DeepMind, WebGPU) | Demonstrated in browser; quality lags desktop |
| **Mathematical projection accuracy** | 3DGUT, AAA-Gaussians | Shipping in some tools; not universally adopted |

### ❌ Known Gaps — What Still Needs Improvement

**1. Editing and modifiability** — This is the biggest open problem. Mesh workflows allow artists to sculpt, Boolean-union, and UV-edit freely. With Gaussians, meaningful geometric edits require either re-capture or diffusion-model-guided edits that are inconsistent across viewpoints and resolution-limited (512–768px). Global edits to lighting, material, or structure remain unsolved.

**2. Mobile performance and file size** — Raw Gaussian scenes are 200 MB–500 MB. Even after aggressive compression, mobile devices need scenes downsampled to 200K–500K splats to hit 30+ FPS. Quality loss is visible at this density. The pipeline from capture to mobile-optimized web delivery is not yet seamless.

**3. Large-scale scenes** — Training 3DGS on large outdoor environments (city blocks, campuses) requires multi-GPU setups or host-offloading tricks (GS-Scale). Memory scales roughly with scene complexity. Per-scene optimization also means large scenes take much longer to train.

**4. Transparent and reflective surfaces** — Specular reflections and transparency violate the assumptions of the 3DGS appearance model (which uses simple spherical harmonics). Mirrors, glass, and water remain poorly reconstructed without special handling.

**5. Dynamic scene generalization** — Current 4DGS methods can handle short clips (seconds to a few minutes) but reconstruction of hours-long dynamic content (e.g., a full sports game) at viewable quality is computationally intractable.

**6. File format fragmentation** — The ecosystem splits across PLY, SPLAT, KSPLAT, SPZ, SOG. OpenUSD v26.03 adds a USD schema, but interoperability between tools is still inconsistent. No universal browser-native standard exists yet.

**7. SfM dependency** — All production-quality 3DGS still requires a clean SfM point cloud. Featureless surfaces (blank walls, glossy floors), uncalibrated cameras, or dynamic content during capture breaks the SfM pipeline and degrades output quality significantly. Feed-forward methods (GS-LRM, SHARP) partially bypass this but sacrifice quality on complex scenes.

**8. Perceptual quality metrics** — No agreed-upon perceptual quality metric for 3DGS exists. PSNR/SSIM miss artifacts like floaters and temporal inconsistencies that humans notice immediately. A 2025 subjective dataset was released to address this but is not yet widely adopted.

---

## Summary Scorecard

| Dimension | Status |
|---|---|
| Core rendering (desktop) | ✅ Solved, production-ready |
| Web/mobile viewing | ✅ Solved (with optimization overhead) |
| Consumer capture | ✅ Polycam, Luma AI — widely available |
| Real estate / e-commerce | ✅ Deployed by Zillow, Apartments.com |
| Film VFX | ✅ Superman (2025); Nuke 17.0; OctaneRender 2026 |
| Apple Maps | ✅ Announced June 2026 |
| VR headsets | ✅ VRSplat 72+ fps stereo (research → commercial) |
| OpenUSD pipeline integration | ✅ v26.03 (March 2026) |
| Dynamic/4DGS | 🔬 Research-stage; early production pilots |
| Generative 3DGS | 🔬 SHARP open-source; quality improving |
| Scene editing | 🔬 Hard open problem |
| Large-scale outdoor SLAM | 🔬 Active research |
| Medical / surgical | 🔬 MICCAI 2025; not yet clinical |
| Mobile training | 🔬 Browser demos (Brush by DeepMind) |
| Transparent/reflective surfaces | ❌ Significant gap |
| Non-destructive artist editing | ❌ Fundamental limitation |
| Hours-long dynamic content | ❌ Computationally intractable today |

---

## Sources

- [Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," ACM SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Kerbl, "The Impact and Outlook of 3D Gaussian Splatting," arXiv:2510.26694, Oct 2025](https://arxiv.org/pdf/2510.26694)
- [Volinga, "The Next Big Thing in Splats: 2025's Turning Point and 2026's Acceleration," Feb 2026](https://web.volinga.ai/2025-turning-point-and-2026-trends-blog/)
- [Utsubo, "3D Gaussian Splatting: Complete Guide to Services, Use Cases & Web Viewers (2026)"](https://www.utsubo.com/blog/gaussian-splatting-guide)
- [Apple Machine Learning Research — SHARP (monocular view synthesis)](https://machinelearning.apple.com/research/sharp-monocular-view)
- [Apple Brings 3D Gaussian Splatting to Apple Maps at WWDC 2026](https://pasqualepillitteri.it/en/news/4534/apple-maps-3d-gaussian-splatting-wwdc-2026)
- [AOUSD, "OpenUSD v26.03 adds support for 3D Gaussian Splats," March 2026](https://aousd.org/blog/openusd-v26-03/)
- [CG Channel: OpenUSD 26.03 adds support for 3D Gaussian Splats](https://www.cgchannel.com/2026/03/openusd-26-03-adds-support-for-3d-gaussian-splats/)
- [SurgicalGS: Dynamic 3DGS for Robotic-Assisted Surgical Scene Reconstruction (MICCAI 2025)](https://link.springer.com/chapter/10.1007/978-3-032-05141-7_55)
- [3D Gaussian Splatting in Robotics: A Survey, arXiv:2410.12262](https://arxiv.org/pdf/2410.12262)
- [IDSplat: Instance-Decomposed 3DGS for Driving Scenes, arXiv:2511.19235](https://arxiv.org/abs/2511.19235)
- [Large-Scale Gaussian Splatting SLAM, arXiv:2505.09915](https://arxiv.org/pdf/2505.09915)
- [Frontiers in AI: Human reconstruction using 3DGS — brief survey (2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1709229/full)
- [SIGGRAPH Asia 2025: 3D Gaussian Splatting Challenge](https://gaplab.cuhk.edu.cn/projects/gsRaceSIGA2025/)
- [Compression in 3DGS: A Survey, arXiv:2502.19457](https://arxiv.org/pdf/2502.19457)
- [Wikipedia — Gaussian splatting](https://en.wikipedia.org/wiki/Gaussian_splatting)
- [HuggingFace — Introduction to 3D Gaussian Splatting](https://huggingface.co/blog/gaussian-splatting)
- [PyImageSearch — 3DGS vs NeRF: The End Game of 3D Reconstruction?](https://pyimagesearch.com/2024/12/09/3d-gaussian-splatting-vs-nerf-the-end-game-of-3d-reconstruction/)
