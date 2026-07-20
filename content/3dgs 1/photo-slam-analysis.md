# Photo-SLAM: Deep Dive & Improvement Directions

> Paper: [Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2311.16728) — Huang et al., CVPR 2024
> Code: [GitHub](https://github.com/HuajianUP/Photo-SLAM)

---

## What Photo-SLAM Does

Photo-SLAM maintains a **hyper primitives map** — point clouds that carry both ORB geometric features (for tracking) and Gaussian splatting attributes (for rendering). The system runs four parallel threads: localization (ORB-SLAM3-based factor graph), geometry mapping (sparse point creation via triangulation), photorealistic mapping (3DGS optimization), and loop closure.

Two key contributions on top of this architecture:

1. **Geometry-based densification**: uses inactive 2D ORB keypoints (the ~70% that don't triangulate into 3D points) to create additional Gaussians, densifying the map where texture complexity is highest.

2. **Gaussian-Pyramid (GP) learning**: supervises training with progressively higher-resolution versions of the ground-truth image (coarse → fine), analogous to curriculum learning. This stabilizes convergence for the online incremental setting and is especially important for monocular input where depth is ambiguous.

### Results at a Glance

| Metric | Photo-SLAM | Best Competitor | Dataset |
|---|---|---|---|
| PSNR (mono) | **33.3** | 23.2 (Orbeez-SLAM) | Replica |
| PSNR (RGB-D) | **35.0** | 34.6 (Point-SLAM) | Replica |
| Rendering FPS | **911–1084** | 3.7 (Co-SLAM) | Replica |
| GPU memory | **5–6 GB** | 24 GB (Point-SLAM) | Replica |
| Runs on Jetson | Yes (100 FPS render) | No others | — |

The speed and efficiency story is extremely strong. The rendering quality story is strong on Replica but weaker on TUM (PSNR ~20–22), which reveals the limitations.

---

## Identified Weaknesses

### 1. No Semantic Understanding

Photo-SLAM's map is purely geometric + photometric. It cannot answer spatial queries ("where is the chair?"), segment objects, or support downstream task like manipulation planning. Every Gaussian is just color + geometry — no language or semantic feature. This is the single largest gap relative to what modern robotics and AR demand.

### 2. Weak Monocular Depth → Noisy Geometry in Textureless Regions

In monocular mode, depth for inactive keypoints is estimated by nearest-neighbor interpolation from active keypoints. This is crude — textureless walls and floors get wildly inaccurate depth. The ablation study (Table 4) shows that without GP learning, geometry-based densification actually *hurts* quality in mono (PSNR drops from 22.9 to 20.0), precisely because densified Gaussians have bad positions. GP learning compensates but doesn't fix the root cause.

### 3. Fragile Loop Closure for Gaussian Maps

Photo-SLAM's loop closure corrects poses via similarity transformation, but the **Gaussian map itself is not globally re-optimized** after loop closure. Corrected poses shift viewpoints, but Gaussians trained under the old poses create ghosting artifacts in revisited regions. LoopSplat (2024) and GLC-SLAM (2024) address this with submap-based approaches, but Photo-SLAM's monolithic map makes this hard.

### 4. Static Scene Assumption

The system has no mechanism to handle dynamic objects. A person walking through the scene creates ghost Gaussians. This limits deployment in any real robotics scenario (warehouses, homes, hospitals — all have moving things).

### 5. Mapping Quality Degrades on Real-World Data

On Replica (synthetic indoor, clean images, perfect lighting), PSNR is 33–35. On TUM (real-world, motion blur, lighting variation), PSNR drops to 19–22. The ORB features that drive tracking are sensitive to blur, and the SH-based color model (degree 1, only 16 coefficients) can't handle complex illumination.

### 6. No Global Map Optimization

Unlike Splat-SLAM (Google, 2024) which deforms Gaussians in response to pose graph updates, Photo-SLAM freezes the Gaussian map structure after local optimization. Accumulated drift in long sequences gradually degrades map quality, and there's no mechanism to propagate corrections globally.

### 7. Limited Appearance Model

SH degree 1 (16 coefficients) is compact but cannot represent specular reflections, view-dependent effects, or complex materials. This is fine for diffuse indoor scenes but fails on glossy floors, windows, and metallic surfaces.

---

## Five Improvement Directions

### Direction 1: Add Open-Vocabulary Semantic Features (→ Semantic Photo-SLAM)

**Gap**: Photo-SLAM is semantically blind. Hier-SLAM and SemGauss-SLAM (2025) show semantic 3DGS SLAM is viable but neither starts from Photo-SLAM's efficient ORB-based architecture.

**Approach**:
- Attach a compressed CLIP embedding (16D, via trained autoencoder following LangSplatV2) to each hyper primitive.
- Every K-th keyframe: run SAM → segments, run CLIP → embeddings per segment, compress → assign 16D feature to Gaussians in that segment.
- Run in a 5th parallel thread so the SLAM main loop is unaffected.
- Enable text queries ("where is the exit?") at 30+ FPS — just dot products over the 16D features.

**Why it fits Photo-SLAM specifically**: Photo-SLAM already has the ORB features spatially distributed in the map. The ORB keypoint distribution correlates with object boundaries (texture edges), providing natural anchor points for assigning semantic labels to Gaussian clusters. Other GS-SLAM methods (MonoGS, SplaTAM) don't have this geometric structure.

**Venue target**: ECCV / IROS / ICRA

---

### Direction 2: Monocular Depth Prior Injection (→ Depth-Aware Photo-SLAM)

**Gap**: Monocular depth estimation in Photo-SLAM relies on nearest-neighbor interpolation from sparse ORB points — the weakest part of the system.

**Approach**:
- Replace nearest-neighbor depth with Depth Anything V2 (frozen, ViT-S variant, runs at ~30ms per frame).
- Use the monocular depth as a **confidence-weighted initialization** for inactive keypoint depth: trust Depth Anything where it's geometrically consistent with the sparse SfM points, downweight where they disagree.
- Add a depth regularization loss during Gaussian optimization: `L_depth = confidence × |rendered_depth - DA_depth|`.
- Learn the confidence map jointly (same idea as Q3-A from the approach plan).

**Why it fits Photo-SLAM specifically**: Photo-SLAM's geometry-based densification creates Gaussians from inactive keypoints using estimated depth. If that depth is bad, the Gaussians are bad, and GP learning has to fight uphill. Better depth initialization directly amplifies the value of both existing contributions (Geo densification + GP learning).

**Expected improvement**: PSNR on TUM mono could jump from ~20 to ~25+ (based on what DN-Gaussian and SplatMAP report with depth priors in similar settings).

**Venue target**: WACV / 3DV / ECCV

---

### Direction 3: Submap-Based Global Consistency (→ Photo-SLAM++)

**Gap**: Photo-SLAM's loop closure corrects poses but doesn't re-optimize Gaussians. Long trajectories accumulate ghosting.

**Approach**:
- Divide the Gaussian map into **submaps** (one per N keyframes, e.g., N=50). Each submap is a self-contained set of Gaussians in a local coordinate frame.
- On loop closure detection (already in Photo-SLAM via ORB-SLAM3's loop closure):
  1. Compute relative transform between submaps via Gaussian-to-Gaussian registration (follow LoopSplat's ICP variant).
  2. Run pose graph optimization over submap origins.
  3. Apply rigid transforms to each submap's Gaussians — fast, no re-training needed.
  4. Optionally: fine-tune the boundary Gaussians between adjacent submaps for seamless transitions (200 iterations, ~5 seconds).

**Why it fits Photo-SLAM specifically**: Photo-SLAM already has the keyframe selection + loop detection infrastructure from ORB-SLAM3. The only missing piece is organizing Gaussians into submaps and adding the submap alignment step. The system's C++ / CUDA implementation makes the rigid transform + fine-tune step very fast.

**Venue target**: CVPR / ICCV / 3DV

---

### Direction 4: Dynamic Object Handling (→ Robust Photo-SLAM)

**Gap**: Any moving object creates persistent ghost Gaussians.

**Approach**:
- **Detection**: Use optical flow magnitude between consecutive frames (RAFT-lite or GMFlow-lite) to identify dynamic pixels. Threshold at 3× median flow to flag outliers.
- **Masking**: Exclude dynamic pixels from:
  - ORB feature matching (don't track features on moving objects)
  - Gaussian photometric loss (don't supervise rendering in dynamic regions)
  - Geometry-based densification (don't create new Gaussians from dynamic keypoints)
- **Cleanup**: Periodically (every 100 frames), identify Gaussians that have been masked out consistently across multiple viewpoints. Prune them.
- **Optional extension**: maintain a small set of temporary "dynamic Gaussians" for short-term tracking of moving objects (useful for robotics — know where the person is NOW, just don't permanently add them to the map).

**Why it fits Photo-SLAM specifically**: The ORB tracking pipeline already does outlier rejection on feature matches (via RANSAC). Adding a flow-based dynamic mask is a natural extension of the existing robustification. And since Photo-SLAM's tracking is decoupled from the Gaussian optimization (separate threads), the dynamic mask only needs to be passed between threads — no architectural change.

**Venue target**: IROS / ICRA / ECCV

---

### Direction 5: Higher-Fidelity Appearance Model (→ HD Photo-SLAM)

**Gap**: SH degree 1 (16 coefficients) limits rendering quality, especially on real-world data with complex lighting.

**Approach**:
- **Upgrade SH to degree 2** (48 coefficients per Gaussian). The ablation shows the jump from degree 0→1 was huge; degree 1→2 should give another meaningful boost at moderate memory cost.
- **Add a compact per-Gaussian feature vector** (8D) decoded by a tiny shared MLP (2 layers, 32 hidden) to model view-dependent effects beyond what SH captures. This follows the Scaffold-GS / Neural Gaussians approach.
- **Adapt GP learning** to include perceptual loss (LPIPS) at the finest pyramid level. Currently only L1 + SSIM are used. LPIPS at the final level pushes sharpness on fine details without slowing coarse-level convergence.

**Expected improvement**: PSNR on TUM could increase by 1–3 dB. Rendering FPS would drop from ~1000 to ~400–500 FPS due to the MLP overhead — still far above real-time.

**Venue target**: SIGGRAPH Asia / ECCV / ACM MM

---

## Which Improvement to Pursue?

| Direction | Impact | Difficulty | Timeline | Best Venue |
|---|---|---|---|---|
| 1. Semantic features | High (enables new apps) | Medium | 8–10 mo | ECCV / IROS |
| 2. Depth prior | High (fixes core weakness) | Low | 4–6 mo | WACV / 3DV |
| 3. Submap global consistency | High (fixes drift) | Medium | 8–10 mo | CVPR / 3DV |
| 4. Dynamic handling | Medium (practical) | Low-Medium | 6–8 mo | IROS / ICRA |
| 5. Better appearance | Medium (incremental) | Low | 4–6 mo | SIGGRAPH Asia |

**Recommended combination**: Direction 2 + Direction 4 together. Depth prior injection fixes the monocular quality problem (the biggest weakness), and dynamic object handling makes it work in real environments. Both are relatively low-risk, both address the gap between Replica results (great) and real-world results (mediocre), and together they make a strong "Photo-SLAM for real-world robotics" story. Timeline: 8–10 months.

**If targeting Q1 venues**: Direction 1 (Semantic) or Direction 3 (Global consistency) alone is a strong standalone paper.

---

## Key Papers to Read Alongside Photo-SLAM

| Paper | Why | Link |
|---|---|---|
| MonoGS (Matsuki et al., 2024) | Competing monocular GS-SLAM, different architecture | [GitHub](https://github.com/muskie82/MonoGS) |
| MonoGS++ (Li et al., 2025) | 5.57× faster than MonoGS, dynamic Gaussian insertion | [arXiv](https://arxiv.org/abs/2504.02437) |
| Splat-SLAM (Sandström et al., 2024) | Global Gaussian deformation on pose updates | [arXiv](https://arxiv.org/abs/2405.16544) |
| LoopSplat (Zhu et al., 2024) | Submap-based loop closure for GS-SLAM | [arXiv](https://arxiv.org/abs/2408.10154) |
| GLC-SLAM (2024) | Global-to-local loop closure for GS-SLAM | [arXiv](https://arxiv.org/abs/2409.10982) |
| SplatMAP (2025) | Dense monocular SLAM with depth densification | [arXiv](https://arxiv.org/abs/2501.07015) |
| DenseSplat (2025) | Neural radiance prior for denser Gaussian maps | [arXiv](https://arxiv.org/abs/2502.09111) |
| Hier-SLAM (2024) | Hierarchical semantic categories in GS-SLAM | [arXiv](https://arxiv.org/abs/2409.12518) |
| SemGauss-SLAM (2024) | Dense semantic GS-SLAM | [arXiv](https://arxiv.org/abs/2403.07494) |
| SGAD-SLAM (2026) | Adjusted depth for better radiance fields, CVPR 2026 | [PDF](https://arxiv.org/pdf/2603.21055) |
| LangSplatV2 (2025) | Compressed CLIP in Gaussians at 450+ FPS | [HTML](https://arxiv.org/html/2507.07136v1) |
| Depth Anything V2 (2024) | Monocular depth foundation model | [arXiv](https://arxiv.org/abs/2406.09414) |
| MGS-SLAM (2024) | MVS-based depth for monocular GS-SLAM | [arXiv](https://arxiv.org/abs/2405.06241) |