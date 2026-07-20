# 3DGS Research — Execution Plans

> Concrete implementation approach for each of the 10 research topics from `research-topic-recommendations.md`. Each plan specifies the technical bet, architecture, losses, experiment design, and month-by-month timeline.

---
---

# Q1-Tier Execution Plans

---

## Q1-A: Multi-View Consistent Language-Guided 3DGS Editing

### Core Technical Bet

Replace the current "edit per-view then hope for consistency" paradigm with a **multi-view diffusion → Gaussian distillation** loop. The key insight: if we generate edited views that are *already* geometrically consistent before touching the Gaussians, the downstream 3DGS optimization is dramatically easier.

### Architecture

```
Input: trained 3DGS scene + text edit instruction ("make the chair red")

Step 1 — Semantic Grounding
  └─ Run LangSplat on the scene → per-Gaussian CLIP features
  └─ Query with edit target text → binary mask of affected Gaussians
  └─ Project mask to 2D for each training view → per-view edit regions

Step 2 — Multi-View Consistent Edit Generation
  └─ Select K=8 anchor viewpoints (diverse camera angles covering the target region)
  └─ For each anchor: render current view from 3DGS, composite with edit mask
  └─ Feed all K masked renders + text prompt to Zero123++ (multi-view diffusion)
  └─ Zero123++ outputs K edited images that are geometrically self-consistent
  └─ Upscale from 512px → 1024px using Real-ESRGAN (handle resolution bottleneck)

Step 3 — Consistency-Regularized Gaussian Distillation
  └─ Freeze unmasked Gaussians (background stays locked)
  └─ Optimize only masked Gaussians' color (SH coefficients) + opacity
  └─ Loss = L_photometric(rendered, edited_views) 
         + λ₁ · L_consistency(cross-view color agreement)
         + λ₂ · L_geometric(preserve original depth map in edited region)
  └─ L_consistency: for each Gaussian, enforce that its SH-decoded color
     matches the edited image from ALL viewpoints, not just one
  └─ 500–1000 iterations, ~10 min per edit on T4
```

### Losses (specific formulations)

- **L_photometric**: 0.8 × L1 + 0.2 × D-SSIM (standard 3DGS loss, applied only in edit regions)
- **L_consistency**: render the same Gaussian from 2 random views → penalize color difference after SH evaluation. Formally: `Σ_i ||SH(g_i, d_j) - SH(g_i, d_k)||₂` for random view directions d_j, d_k
- **L_geometric**: `||depth_edited - depth_original||₁` in the edit region — prevents Gaussians from drifting spatially during color edits
- λ₁ = 0.1, λ₂ = 0.05 (tune on NeRF Synthetic validation set)

### Novel Metric: Multi-View Edit Consistency Score (MECS)

For each edit, render from N=20 novel views. For each pair (i,j), warp view i to view j using depth and compute SSIM in the edited region only. MECS = mean pairwise SSIM. This directly measures what we claim to solve.

### Experiment Design

| Experiment | What it shows | Scenes |
|---|---|---|
| Main comparison vs. GaussianEditor, InstructNeRF2NeRF | MECS + PSNR + LPIPS + user study | Mip-NeRF 360 (5 scenes) + LERF (5 scenes) |
| Ablation: remove L_consistency | Proves consistency loss matters | 3 scenes |
| Ablation: single-view diffusion vs. Zero123++ multi-view | Proves multi-view generation matters | 3 scenes |
| Ablation: K=4 vs 8 vs 16 anchor views | Cost-quality tradeoff | 2 scenes |
| Resolution study: native 512 vs. upscaled 1024 | Proves upscaling helps | 2 scenes |
| User study: 20 participants, A/B pairwise comparison | Human preference over baselines | 5 edits across 5 scenes |

### Month-by-Month Timeline (12 months)

| Month | Milestone |
|---|---|
| 1–2 | Reproduce GaussianEditor + LangSplat on Mip-NeRF 360. Get training pipeline working end-to-end. |
| 3 | Integrate Zero123++ for multi-view edit generation. Test on 2 simple edits (color changes). |
| 4 | Implement consistency-regularized distillation (Step 3). Tune λ₁, λ₂ on validation scenes. |
| 5–6 | Run full comparison experiments. Implement MECS metric. Run ablations. |
| 7 | User study design, IRB if needed, data collection via Prolific (~$150). |
| 8–9 | Write paper. Iterate on experiments based on writing gaps. |
| 10 | Internal review, advisor feedback, revisions. |
| 11 | Submit to CVPR/ICCV. |
| 12 | Address reviewer feedback / prepare rebuttal or backup submission to ECCV. |

### What Could Go Wrong

- **Zero123++ may not generalize to scene-level edits** (it's trained on objects). Mitigation: fall back to SyncDreamer or fine-tune Zero123++ on scene crops.
- **Upscaling introduces artifacts that break consistency**. Mitigation: use tile-based Real-ESRGAN with overlap blending; or accept 512px and focus contribution on consistency.
- **LangSplat mask is imprecise → edit bleeds into background**. Mitigation: refine masks with SAM, or use Gaussian Grouping instead.

---

## Q1-B: Efficient Sparse-View Feed-Forward 3DGS Reconstruction

### Core Technical Bet

Combine **pose-free reconstruction** (no SfM needed) with a **lightweight backbone** (5× smaller than GS-LRM). The niche: GS-LRM and PixelSplat assume calibrated poses; DUSt3R/MASt3R provide poses but don't output Gaussians directly. We close the gap with a single model that takes 1–3 uncalibrated images and outputs both poses and Gaussians.

### Architecture

```
Input: 1–3 unposed RGB images

Backbone: EfficientViT-L (pretrained on ImageNet, ~30M params vs. ViT-L's 300M)
  └─ Encode each image to feature maps at 1/8 resolution

Pose Estimation Head (lightweight):
  └─ Cross-attention between image features → relative pose prediction
  └─ Parameterize as rotation (6D representation) + translation (3D)
  └─ Supervised with ground-truth relative poses during training

Depth Prior Injection:
  └─ Run Depth Anything V2 (frozen, ViT-S variant) on each input image
  └─ Concatenate monocular depth as 4th channel to backbone features
  └─ This gives strong geometric prior without training cost

Gaussian Prediction Head:
  └─ Per-pixel MLP: features → (Δxyz, scale, rotation_quat, opacity, SH_deg1)
  └─ Δxyz = offset from unprojected depth → Gaussian 3D position
  └─ Only SH degree 1 (4 coefficients × 3 channels = 12) to keep model small
  └─ Output: ~50K Gaussians per input image

Rendering + Loss:
  └─ Differentiable rasterization (gsplat) from novel viewpoints
  └─ Loss = L1 + 0.2 × D-SSIM + 0.1 × L_depth(predicted depth vs. Depth Anything prior)
```

### Key Design Decisions

1. **Why EfficientViT, not ViT-L?** GS-LRM's transformer is 300M+ params, needs A100 to train. EfficientViT-L is 30M params, trainable on a single RTX 4090. This IS the contribution — same task, 10× lighter.
2. **Why Depth Anything V2 injection?** Monocular depth gives scale + geometry prior that compensates for the smaller backbone. It's frozen, so no training cost.
3. **Why SH degree 1, not 3?** Degree 1 has 12 color params vs. 48 for degree 3. Acceptable quality loss (~0.5 dB PSNR) but 4× fewer params per Gaussian. Consistent with the "lightweight" angle.
4. **6D rotation representation** for pose prediction — continuous, no gimbal lock, better gradients than quaternions for pose estimation.

### Training Strategy

- **Stage 1 (pose pretraining, 2 days):** Train pose estimation head only on RealEstate10K with ground-truth poses. Freeze backbone partially.
- **Stage 2 (joint training, 5 days):** Unfreeze everything. Train jointly on RealEstate10K + CO3Dv2. Novel view synthesis loss drives both pose and Gaussian accuracy.
- **Data augmentation:** Random crop, color jitter, random number of input views (1, 2, or 3).
- **Batch size:** 4 scenes × 3 views = 12 images per batch (fits in 24 GB VRAM).

### Experiment Design

| Experiment | What it shows | Data |
|---|---|---|
| Main: compare vs. PixelSplat, GS-LRM, SHARP | Quality vs. model size vs. speed | RealEstate10K test, CO3Dv2 test, DTU |
| Zero-shot generalization | Trains on RE10K, tests on ScanNet (never seen) | ScanNet 8 scenes |
| Ablation: remove Depth Anything prior | Proves depth injection matters | RE10K subset |
| Ablation: EfficientViT-S vs. L vs. ViT-L | Model size vs. quality tradeoff | RE10K subset |
| Ablation: SH degree 0 vs. 1 vs. 3 | Quality vs. model size per Gaussian | RE10K subset |
| Efficiency table: params, FLOPs, latency, memory | Core claim — lighter model | Single A100, RTX 3080 |

### Month-by-Month Timeline (12 months)

| Month | Milestone |
|---|---|
| 1 | Set up training pipeline: dataloader for RealEstate10K + CO3Dv2. Reproduce PixelSplat results. |
| 2 | Implement EfficientViT backbone + per-pixel Gaussian head. Get basic rendering working. |
| 3 | Integrate Depth Anything V2 injection. Implement pose estimation head. |
| 4 | Stage 1 pose pretraining. Debug and validate pose accuracy against GT. |
| 5–6 | Stage 2 joint training. Hyperparameter sweep (learning rate, loss weights). |
| 7–8 | Full evaluation: comparisons, ablations, efficiency benchmarks. |
| 9–10 | Write paper. Fill experimental gaps. |
| 11 | Submit CVPR/ICCV. |
| 12 | Rebuttal / backup submission. |

### What Could Go Wrong

- **Small backbone can't match GS-LRM quality on complex scenes.** Mitigation: accept a small quality gap and frame the contribution as "90% quality at 10% cost." The efficiency story is the paper.
- **Pose estimation from 1 image is ill-defined.** Mitigation: for single-image input, use Depth Anything V2's metric depth to set scale, predict identity pose, and frame it as "monocular view synthesis" (like SHARP) rather than multi-view reconstruction.
- **Training instability with joint pose + Gaussian optimization.** Mitigation: curriculum — start with 3-view input (easiest poses), gradually add 2-view and 1-view.

---

## Q1-C: 3DGS for Long-Duration Dynamic Scene Reconstruction

### Core Technical Bet

Current 4DGS methods create per-frame or per-timestep Gaussians — memory grows linearly with video length. We propose **static-dynamic decomposition** with a **shared canonical deformation field**, so memory is roughly O(scene_complexity) not O(video_length). Target: reconstruct 1–2 minute monocular phone videos, which no current method handles.

### Architecture

```
Input: monocular video (1–2 min, 30 fps → 1800–3600 frames)
  └─ Run COLMAP on sampled keyframes (every 10th frame) → camera poses
  └─ Run Depth Anything V2 on all frames → monocular depth
  └─ Run RAFT optical flow on consecutive frames → flow fields

Phase 1 — Static Background Reconstruction
  └─ Select frames with minimal motion (flow magnitude < threshold)
  └─ Train standard 3DGS on these frames → static Gaussians G_static
  └─ These Gaussians are shared across ALL timesteps (no temporal cost)

Phase 2 — Dynamic Foreground via Canonical Deformation
  └─ Initialize dynamic Gaussians G_dynamic in a canonical frame (t=0)
  └─ Deformation network D(g, t) → (Δposition, Δrotation, Δscale, Δopacity)
     - Architecture: tiny MLP (4 layers, 128 hidden) with positional encoding
     - Input: Gaussian center xyz (3D) + time t (1D), both with PE
     - This MLP is SHARED across all Gaussians — compact
  └─ At time t: render G_static ∪ D(G_dynamic, t) from camera pose at t
  └─ Loss = L1 + D-SSIM + λ_flow · L_flow + λ_depth · L_depth

Phase 3 — Temporal Segmentation for Long Videos
  └─ Split video into overlapping segments of 300 frames (~10 sec)
  └─ Train one deformation MLP per segment (shared G_static, shared G_dynamic canonical)
  └─ Stitch segments via linear blending in overlap regions
  └─ This keeps each MLP small while covering minute-long videos
```

### Losses

- **L_photometric**: 0.8 × L1 + 0.2 × D-SSIM (standard)
- **L_flow**: `||flow_predicted - flow_RAFT||₁` where flow_predicted is computed by projecting Gaussian motion between consecutive frames. Forces deformation to match observed 2D motion.
- **L_depth**: `||rendered_depth - depth_anything||₁` per frame. Scale-invariant version (normalize both to [0,1] per frame).
- **L_rigidity**: `||D(g_i, t) - D(g_j, t)||₂ - ||g_i - g_j||₂` for nearby Gaussians i,j. Encourages locally rigid motion (reduces artifacts in textureless regions).
- Weights: λ_flow = 0.05, λ_depth = 0.01, λ_rigidity = 0.001.

### Static-Dynamic Separation Strategy

- Compute per-pixel motion magnitude from RAFT flow, averaged over all frames.
- Pixels with mean flow < 1px are "static." Use these to identify static Gaussians during initial 3DGS training (Gaussians that contribute only to static pixels → freeze as G_static).
- Remaining Gaussians → G_dynamic, refined in Phase 2.
- Alternatively: train a binary mask per Gaussian (static vs. dynamic) jointly during optimization, supervised by the flow-based motion mask.

### Experiment Design

| Experiment | What it shows | Data |
|---|---|---|
| Main comparison vs. 4DGS, Deformable-3DGS, MoDGS | Quality (PSNR/SSIM/LPIPS) + memory + max video length | HyperNeRF (4 scenes), iPhone Dataset (4 scenes) |
| Long video stress test | Reconstruction quality at 30s, 60s, 120s | 3 custom phone captures (capture yourself) |
| Ablation: remove static-dynamic decomposition | Proves decomposition saves memory | 2 scenes |
| Ablation: remove flow loss | Proves flow supervision helps | 2 scenes |
| Ablation: single MLP vs. per-segment MLPs | Proves segmentation needed for long videos | 1 long scene |
| Memory profiling: GPU VRAM vs. video length | Core efficiency claim | Synthetic varying-length sequences |

### Month-by-Month Timeline (12 months)

| Month | Milestone |
|---|---|
| 1–2 | Reproduce 4DGS and Deformable-3DGS on HyperNeRF. Set up RAFT + Depth Anything V2 preprocessing. |
| 3 | Implement static-dynamic decomposition. Train static Gaussians on filtered frames. |
| 4–5 | Implement canonical deformation MLP. Train on short clips first (D-NeRF Synthetic). Debug flow loss. |
| 6 | Implement temporal segmentation (Phase 3). Test on 60-second clips. |
| 7 | Capture custom long videos with phone. Run full pipeline on 1–2 minute videos. |
| 8–9 | Full comparisons + ablations + memory profiling. |
| 10–11 | Write paper. |
| 12 | Submit CVPR/SIGGRAPH. |

### What Could Go Wrong

- **COLMAP fails on dynamic scenes.** Mitigation: use only high-static-ratio frames for COLMAP; or use DUSt3R/MASt3R for robustness.
- **Deformation MLP can't model complex articulated motion** (e.g., a person dancing). Mitigation: use per-Gaussian local deformation codes (small latent vector per Gaussian, 8D) that condition the MLP.
- **Segment stitching creates visible seams.** Mitigation: increase overlap from 30 to 60 frames; use learnable blending weights.

---
---

# Q2-Tier Execution Plans

---

## Q2-A: Perceptually-Guided Compression for Mobile 3DGS

### Core Technical Bet

Current compression methods optimize for PSNR — but a scene compressed 10× might score well on PSNR while having obvious floaters in empty space. We train a **Gaussian-specific perceptual metric** on human preference data, then use it as a differentiable loss to guide which Gaussians to prune. The human study IS the contribution, not the algorithm.

### Architecture

```
Part 1 — Generate Comparison Dataset
  └─ Take 5 3DGS methods × 4 compression levels × 9 Mip-NeRF 360 scenes
  └─ For each: render 10 viewpoints → 1,800 unique renders
  └─ Create 1,000 pairwise comparisons (method A vs. B at same viewpoint)
  └─ Each pair shows the same scene/viewpoint at different compression levels

Part 2 — Crowdsource Human Preferences
  └─ Platform: Prolific (or MTurk with student credit)
  └─ Task: "Which image looks better?" (2AFC — two-alternative forced choice)
  └─ Each comparison judged by 5 workers → majority vote
  └─ Quality control: 10% catch trials (obvious quality differences)
  └─ Budget: ~1000 comparisons × 5 judgments × $0.03 = ~$150
  └─ Collect artifact annotations: for each "worse" image, worker selects
     artifact type(s) from {floaters, blurry edges, color banding, popping, other}

Part 3 — Train Perceptual Metric Network
  └─ Architecture: LPIPS backbone (pretrained VGG, frozen lower layers)
     + fine-tune top 2 blocks + new comparison head (2-layer MLP)
  └─ Input: image pair (rendered_compressed, rendered_reference)
  └─ Output: scalar quality score
  └─ Training: Bradley-Terry model on pairwise preferences
     P(A > B) = σ(f(A) - f(B)), trained with cross-entropy loss
  └─ Train/val split: 800/200 comparisons
  └─ Training: ~2 GPU-hours on T4

Part 4 — Metric-Guided Pruning
  └─ Start from a fully trained 3DGS scene
  └─ For each Gaussian, compute importance = Δ(quality score) if removed
     - Approximate via gradient: ∂(metric_score)/∂(opacity_i)
  └─ Prune Gaussians with lowest importance first (keep top-K%)
  └─ Fine-tune remaining Gaussians for 200 iterations with metric as loss:
     Loss = L_photometric + λ_percept · (1 - metric_score(rendered))
  └─ λ_percept = 0.5 (tuned on validation)
```

### Experiment Design

| Experiment | What it shows | Data |
|---|---|---|
| Human preference accuracy: our metric vs. PSNR/SSIM/LPIPS | Our metric better predicts human choice | Held-out 200 comparisons |
| Metric-guided pruning vs. LightGaussian / Compact3DGS | Better human ratings at same file size | Mip-NeRF 360 (9 scenes) + Tanks & Temples (4 scenes) |
| Rate-distortion curves (file size vs. metric score) | Our pruning is more perceptually efficient | Same scenes, 5 compression levels |
| Artifact-type analysis | Which artifacts matter most to humans | Annotation data from Part 2 |
| User study: 20 participants, 50 A/B comparisons | Final human validation | 5 scenes, our method vs. LightGaussian at 10× compression |

### Month-by-Month Timeline (10 months)

| Month | Milestone |
|---|---|
| 1 | Reproduce LightGaussian + Compact3DGS on Mip-NeRF 360. Generate renders at multiple compression levels. |
| 2 | Design comparison pairs. Build crowdsourcing interface (simple HTML + Prolific integration). |
| 3 | Run crowdsourcing study. Collect and clean preference data. |
| 4 | Train perceptual metric network. Validate on held-out set. |
| 5–6 | Implement metric-guided pruning. Integrate metric as differentiable loss. Tune λ_percept. |
| 7 | Full comparison experiments + rate-distortion curves. |
| 8 | Final user study (A/B validation). |
| 9–10 | Write and submit to ECCV / ACM MM. |

### What Could Go Wrong

- **Human preferences are noisy / inconsistent.** Mitigation: 5 workers per comparison + catch trials. Compute inter-annotator agreement (Krippendorff's α). If low, increase workers to 7.
- **Metric doesn't generalize to scenes outside training set.** Mitigation: train on Mip-NeRF 360, test on Tanks & Temples (zero-shot transfer).
- **Pruning with the metric is slower than PSNR-based pruning.** Mitigation: approximate importance via a single backward pass, not per-Gaussian evaluation.

---

## Q2-B: Semantic 3DGS SLAM with Open-Vocabulary Scene Understanding

### Core Technical Bet

Fuse CLIP language features into a Gaussian SLAM system **incrementally** — not as a post-processing step on a finished map, but during live SLAM operation. This means a robot can answer "where is the door?" while it's still mapping.

### Architecture

```
Base SLAM: MonoGS (monocular RGB) or Splat-SLAM (RGB-only)
  └─ Provides: camera tracking + incremental Gaussian map construction

Semantic Feature Pipeline (runs in parallel):
  └─ Every K-th keyframe (K=5):
     1. Run SAM on the frame → segment masks
     2. Run CLIP (ViT-B/32, frozen) on each masked segment → 512D embedding
     3. Compress via autoencoder (follow LangSplatV2):
        - Encoder: 512D → 16D (trained offline on CLIP embeddings from ScanNet)
        - This 16D vector is what gets stored per-Gaussian
     4. For each new Gaussian added by SLAM in this keyframe:
        - Assign the 16D feature from the segment it belongs to
        - Backproject segment mask to 3D using the depth from SLAM

Query Interface:
  └─ Input: text query ("red chair", "exit sign")
  └─ Encode text with CLIP text encoder → 512D
  └─ Compress to 16D via same autoencoder
  └─ For each Gaussian: cosine_similarity(gaussian_feature_16D, query_16D)
  └─ Threshold → highlight matching Gaussians in 3D view
  └─ Runs at 30+ FPS (just dot products over Gaussians)

Consistency Refinement:
  └─ Problem: same object seen from multiple keyframes → features may disagree
  └─ Solution: for each Gaussian, maintain running average of its 16D feature
     weighted by viewing confidence (closer + more frontal → higher weight)
  └─ Every 50 keyframes: run a quick global pass to smooth features
     between spatially adjacent Gaussians (graph Laplacian on k-NN graph)
```

### Key Design Decisions

1. **Why 16D, not 512D?** Memory. At 500K Gaussians, 512D floats = 1 GB extra. 16D = 32 MB. Acceptable on mobile hardware.
2. **Why autoencoder, not PCA?** LangSplatV2 showed autoencoder preserves query accuracy better than PCA at same dimensionality. We follow their approach.
3. **Why SAM + CLIP, not a single model like ODISE?** SAM gives cleaner masks; CLIP gives open-vocabulary. Combined pipeline is more robust than end-to-end alternatives for SLAM's incremental setting.
4. **Run SAM every 5th keyframe, not every frame.** SAM is slow (~200ms per frame). Every 5th keyframe amortizes cost while maintaining coverage (SLAM keyframes already subsample the video).

### Experiment Design

| Experiment | What it shows | Data |
|---|---|---|
| Mapping quality: compare SLAM metrics (ATE, PSNR) with and without semantic head | Semantic features don't degrade SLAM | Replica (8 scenes), TUM RGB-D (5 seq) |
| Query accuracy: text query → mIoU against GT semantic labels | Open-vocabulary accuracy | ScanNet (8 scenes with GT semantics), Replica (has GT labels) |
| Compare vs. LangSplat (offline) | Our online method matches offline quality | 5 Replica scenes |
| Query latency benchmark | Queries at 30+ FPS | Various map sizes (100K–1M Gaussians) |
| Ablation: 8D vs. 16D vs. 32D feature dimension | Sweet spot for quality vs. memory | 3 scenes |
| Ablation: remove consistency refinement | Proves smoothing helps | 3 scenes |

### Month-by-Month Timeline (10 months)

| Month | Milestone |
|---|---|
| 1 | Reproduce MonoGS on Replica + TUM RGB-D. Verify SLAM metrics match paper. |
| 2 | Train LangSplatV2-style autoencoder on CLIP embeddings extracted from ScanNet. |
| 3–4 | Integrate semantic pipeline: SAM + CLIP + autoencoder → per-Gaussian features during SLAM. |
| 5 | Implement query interface. Test on simple queries ("chair", "table") in Replica. |
| 6 | Implement consistency refinement (running average + Laplacian smoothing). |
| 7–8 | Full evaluation: SLAM metrics, query accuracy, latency, ablations. |
| 9–10 | Write and submit to ECCV / IROS. |

### What Could Go Wrong

- **Semantic head slows down SLAM below real-time.** Mitigation: run SAM + CLIP in a separate thread. SLAM main loop is unaffected; features arrive asynchronously and are assigned to Gaussians retroactively.
- **16D is too compressed for fine-grained queries** (e.g., "the red cup" vs. "the blue cup"). Mitigation: increase to 32D if memory allows, or condition on color separately.
- **SAM segments are inconsistent across views.** Mitigation: the running average + Laplacian smoothing is designed exactly for this.

---

## Q2-C: 3DGS for Transparent and Refractive Objects

### Core Technical Bet

Add a **refraction Gaussian** primitive type that carries an index-of-refraction (IOR) parameter. During rasterization, when a ray hits a refractive Gaussian, it bends according to Snell's law (first-order approximation) and continues to sample Gaussians behind it. This is physically grounded but stays within the rasterization framework (no full ray tracing).

### Architecture

```
Extension to standard 3DGS rasterizer (CUDA):

Per Gaussian, add:
  └─ is_refractive: binary flag (0 = standard, 1 = refractive)
  └─ ior: index of refraction (scalar, learnable, initialized to 1.5 for glass)
  └─ transmittance: how much light passes through (scalar, learnable)

Modified Rendering (front-to-back alpha compositing):

For standard Gaussians: unchanged.
For refractive Gaussians:
  1. Compute surface normal n̂ from Gaussian covariance (shortest eigenvector)
  2. Compute refracted ray direction via Snell's law:
     d_refracted = (η₁/η₂) · d_incident + (η₁/η₂ · cos(θ_i) - cos(θ_t)) · n̂
     where η₁=1.0 (air), η₂=ior
  3. FIRST-ORDER APPROXIMATION: instead of tracing the refracted ray,
     compute a 2D pixel offset Δuv from the ray direction change
  4. Sample the color at (u + Δu, v + Δv) from the accumulated
     background behind this Gaussian → this is the "seen-through" color
  5. Blend: C = transmittance × C_background(u+Δu, v+Δv)
           + (1 - transmittance) × C_gaussian(SH-evaluated)

This is a screen-space refraction approximation — fast, differentiable,
and good enough for common transparent objects (glass, water, plastic).
```

### Key Implementation Detail: Two-Pass Rendering

1. **Pass 1**: Render all non-refractive Gaussians → background color buffer + depth buffer.
2. **Pass 2**: For each refractive Gaussian, compute Δuv offset, sample from Pass 1 background buffer at offset location, blend with Gaussian's own color.

This avoids the combinatorial explosion of tracing secondary rays through multiple refractive surfaces. It's approximate but handles the most common cases (single glass surface, water surface).

### Custom Dataset Capture

- 10 objects: wine glass, water bottle, clear plastic container, glass vase, glass ball, acrylic block, glass jar, reading glasses, glass table, glass door
- Capture protocol: smartphone (iPhone/Pixel), ~100 images per object, controlled indoor lighting
- Each object placed on a textured background (checkerboard pattern — makes refraction distortion visible)
- This dataset is a contribution in itself — no prior 3DGS-specific transparent object benchmark exists

### Losses

- **L_photometric**: standard L1 + D-SSIM
- **L_ior_prior**: `(ior - 1.5)² × 0.01` — soft regularization toward typical glass IOR (prevents ior from diverging)
- **L_transmittance_sparsity**: `|transmittance| × 0.001` — encourage most Gaussians to be opaque (only a few should be refractive)

### Experiment Design

| Experiment | What it shows | Data |
|---|---|---|
| Compare vs. standard 3DGS, TransparentGS on transparent objects | PSNR/SSIM on transparent regions + full image | Shiny dataset + NeRF Synthetic (glass ball) + our custom dataset |
| Mixed scene test | Handles opaque + transparent objects together | Custom scenes with glass on table |
| Ablation: remove refraction (just transparency via alpha) | Proves refraction modeling matters | Glass ball, wine glass |
| Ablation: remove IOR learning (fixed at 1.5) | Proves learnable IOR helps | Water (IOR 1.33) vs. glass (IOR 1.5) |
| Rendering speed benchmark | Still real-time despite two-pass | All scenes, report FPS |
| Qualitative comparison | Visual quality on challenging cases | Water caustics, thick glass |

### Month-by-Month Timeline (12 months)

| Month | Milestone |
|---|---|
| 1–2 | Study 3DGS CUDA rasterizer source code. Understand the rendering kernel deeply. Reproduce baseline on standard scenes. |
| 3 | Add per-Gaussian IOR + transmittance parameters. Implement two-pass rendering (Pass 1: opaque, Pass 2: refractive). |
| 4–5 | Implement screen-space refraction (Snell's law + pixel offset). Make it differentiable. Debug gradient flow. |
| 6 | Capture custom transparent object dataset with smartphone. |
| 7–8 | Full training pipeline. Tune losses. Run on Shiny dataset + custom data. |
| 9 | Comparison experiments + ablations + speed benchmarks. |
| 10–11 | Write paper. |
| 12 | Submit to ECCV/ICCV. |

### What Could Go Wrong

- **CUDA kernel modification is hard.** This is the biggest risk. Mitigation: start with a Python-only prototype using gsplat's Python API (slower but debuggable), then port to CUDA once the approach is validated.
- **Screen-space refraction is too approximate for thick objects** (e.g., a solid glass sphere with double refraction). Mitigation: acknowledge this limitation; focus on single-surface refraction (glasses, windows, bottles). Leave multi-surface for future work.
- **IOR optimization is unstable.** Mitigation: warm-start with known IOR values for common materials; use a slow learning rate (0.1× the main LR) for IOR.

---

## Q2-D: 3DGS-Based Surgical Scene Reconstruction with Tissue Deformation

### Core Technical Bet

Combine **temporal deformation modeling** (from 4DGS) with **specular highlight separation** (from PR-ENDO) in a single unified framework for endoscopic scenes. No existing method handles both simultaneously at real-time speed.

### Architecture

```
Input: stereo endoscopic video (SCARED / C3VD dataset)
  └─ Stereo pair → depth via stereo matching (RAFT-Stereo or built-in)
  └─ Segment specular highlights: simple thresholding on HSV (V > 0.95)

3DGS Base:
  └─ Initialize Gaussians from stereo depth point cloud (frame 0)

Deformation Module:
  └─ Tiny MLP: (gaussian_center_xyz, time_t) → (Δxyz, Δrotation, Δscale)
  └─ Positional encoding: 6 freq for xyz, 4 freq for t
  └─ Architecture: 3 layers × 64 hidden units (very small — surgical clips are short)
  └─ This handles breathing-induced tissue deformation

Specular Decomposition Module:
  └─ Per Gaussian, decompose color into:
     C_diffuse (Lambertian, SH degree 2) + C_specular (view-dependent, SH degree 4)
  └─ During training, mask out specular highlights from photometric loss
     (don't try to reconstruct them — suppress them)
  └─ At inference: render C_diffuse only → clean tissue appearance
     OR render C_diffuse + C_specular → full appearance with highlights

Combined Rendering:
  └─ At time t: deform Gaussians → render → decompose color → compute loss
  └─ Loss = L_diffuse(non-specular pixels) + 0.1 × L_specular(all pixels)
         + λ_deform_smooth · L_deform_smooth
         + λ_depth · L_depth(stereo depth)
```

### Losses

- **L_diffuse**: L1 + D-SSIM on non-specular pixels only (masked by HSV threshold)
- **L_specular**: L1 on all pixels using full (diffuse + specular) render — soft supervision
- **L_deform_smooth**: `||D(g, t) - D(g, t-1)||₂` — temporal smoothness of deformation (tissue doesn't teleport)
- **L_depth**: `||rendered_depth - stereo_depth||₁` — stereo depth is strong supervision
- **L_rigidity**: same as Q1-C — locally rigid deformation prior

### Experiment Design

| Experiment | What it shows | Data |
|---|---|---|
| Main comparison vs. EndoGaussian, SurgicalGS, EndoNeRF | PSNR/SSIM/LPIPS + FPS | SCARED (3 datasets), EndoNeRF dataset (2 scenes) |
| Ablation: remove deformation module | Proves deformation matters for quality | 2 SCARED sequences with visible breathing |
| Ablation: remove specular decomposition | Proves specular handling matters | Sequences with strong highlights |
| Ablation: both modules together vs. separate | Proves joint is better | 2 scenes |
| Real-time benchmark | 30+ FPS on RTX 3080 | All scenes |
| Relighting demo | Show tissue under different virtual light positions | 2 scenes (qualitative) |

### Month-by-Month Timeline (9 months)

| Month | Milestone |
|---|---|
| 1 | Get SCARED + C3VD dataset access (academic registration). Reproduce EndoGaussian baseline. |
| 2 | Implement deformation MLP on top of 3DGS. Test on SCARED without specular handling. |
| 3 | Implement specular decomposition (diffuse/specular SH split + masking). |
| 4–5 | Combine both modules. Joint training. Tune loss weights. |
| 6–7 | Full comparisons + ablations. Relighting demo. |
| 8–9 | Write and submit to MICCAI. |

### What Could Go Wrong

- **SCARED dataset access takes weeks.** Mitigation: start development on EndoNeRF dataset (no registration needed), port to SCARED when access arrives.
- **Specular mask from HSV thresholding is too crude.** Mitigation: train a small U-Net specular segmentation model on a synthetic endoscopy dataset (SimCol), or use PR-ENDO's learned decomposition directly.
- **Deformation + specular together slow down below 30 FPS.** Mitigation: the deformation MLP adds ~2ms per frame, specular decomposition adds ~1ms. Total overhead ~3ms on top of ~12ms base = ~60 FPS. Should be fine.

---
---

# Q3-Tier Execution Plans

---

## Q3-A: Robust Depth-Prior Integration for 3DGS with Cheap Sensors

### Core Technical Bet

Consumer LiDAR (iPhone, iPad) gives depth that's noisy (2–5 cm error) and sparse. Neural monocular depth (Depth Anything) is dense but scale-ambiguous. We learn a **per-pixel confidence map** that tells 3DGS where to trust the real sensor depth and where to ignore it. This is the simplest possible contribution that addresses a real gap.

### Architecture

```
Input: RGB images + iPhone LiDAR depth maps (noisy, sparse)

Confidence Estimation Module:
  └─ Lightweight CNN: U-Net with EfficientNet-B0 encoder
  └─ Input: RGB image (3ch) + raw depth (1ch) + depth gradient magnitude (1ch)
  └─ Output: per-pixel confidence map c ∈ [0, 1]
  └─ Intuition: low confidence on textureless walls (LiDAR noisy there),
     reflective floors (LiDAR unreliable), depth discontinuities (LiDAR bleeds)

Training the Confidence Module (self-supervised):
  └─ No ground-truth confidence labels needed
  └─ Train jointly with 3DGS:
     L_depth_weighted = Σ_pixels c(p) · |depth_rendered(p) - depth_lidar(p)|
     + (1 - mean(c)) · λ_coverage  ← prevents trivial solution of c=0 everywhere
  └─ The 3DGS optimization "teaches" the confidence net which pixels have
     reliable depth by finding where depth supervision helps vs. hurts

Modified 3DGS Training:
  └─ Standard photometric loss (all pixels)
  └─ + confidence-weighted depth loss (see above)
  └─ + during densification: only use LiDAR depth for Gaussian placement
     where c > 0.5 (don't place new Gaussians based on noisy depth)

Full pipeline:
  1. Run COLMAP on RGB images → camera poses + sparse SfM points
  2. Initialize 3DGS from SfM points
  3. Train 3DGS + confidence net jointly for 30K iterations
  4. Confidence net learns to downweight noisy depth regions automatically
```

### Experiment Design

| Experiment | What it shows | Data |
|---|---|---|
| Main: ours vs. DN-Gaussian, standard 3DGS, 3DGS+raw-depth | PSNR/SSIM/LPIPS + depth accuracy | ARKitScenes (10 scenes), ScanNet (5 scenes) |
| Challenging scenes: textureless walls, reflective floors | Our method handles failure cases | ARKitScenes subset (select worst-case scenes) |
| Confidence map visualization | Interpretable — shows where sensor is trusted | Qualitative on 5 scenes |
| Ablation: remove confidence weighting (uniform depth loss) | Proves confidence helps | 5 scenes |
| Ablation: remove densification gating (use all depth for placement) | Proves gating helps | 5 scenes |
| Compare real LiDAR depth vs. Depth Anything neural depth | Motivates the whole paper — real depth has different failure modes | 5 scenes |

### Month-by-Month Timeline (6 months)

| Month | Milestone |
|---|---|
| 1 | Download ARKitScenes. Reproduce standard 3DGS on 5 scenes. Add naive depth supervision loss and measure. |
| 2 | Implement confidence U-Net. Joint training with 3DGS. Debug self-supervised confidence learning. |
| 3 | Add densification gating. Tune λ_coverage. Validate on ScanNet. |
| 4 | Full comparison experiments + ablations. |
| 5–6 | Write and submit to WACV / 3DV. |

### What Could Go Wrong

- **Confidence net collapses to uniform values.** Mitigation: the λ_coverage penalty on mean confidence prevents all-zero; add entropy regularization on confidence map to encourage bimodal distribution.
- **ARKitScenes depth is "too good"** (Apple's post-processing cleans it up). Mitigation: also test on raw LiDAR scans from ScanNet (noisier) or synthetically degrade depth.
- **Contribution seen as too incremental.** Mitigation: emphasize the real-sensor angle (not neural depth), and make the confidence visualization a strong qualitative result.

---

## Q3-B: 3DGS Quality Benchmark with Perceptual Evaluation

### Core Technical Bet

This is a **benchmark paper**, not an algorithm paper. The contribution is a dataset of human preferences + a trained metric + a re-ranking of all major 3DGS methods. If PSNR says method A is best but humans prefer method B, that's a publishable finding.

### Execution Plan

```
Phase 1 — Generate Baseline Renders (Months 1–3)

  Run 15 methods on 3 standard datasets:
  └─ Methods: original 3DGS, Mip-Splatting, 2DGS, SuGaR, 
     LightGaussian, Compact3DGS, EAGLES, 4DGS, Deformable-3DGS,
     GS-LRM, PixelSplat, SplaTAM output, gsplat-MCMC, 
     Scaffold-GS, GOF
  └─ Datasets: Mip-NeRF 360 (9 scenes), NeRF Synthetic (8 scenes),
     Tanks & Temples (4 scenes)
  └─ For each method × scene: render 20 novel viewpoints at 1080p
  └─ Total: ~15 × 21 × 20 = 6,300 rendered images
  └─ For 4DGS methods: also render video sequences (30 frames each)

  Compute time: stagger across 2–3 months on Kaggle free tier
  (most methods take <1h per scene; some need Colab Pro)

Phase 2 — Design Subjective Study (Month 3)

  5 artifact categories to evaluate:
  1. Floaters (stray Gaussians in empty space)
  2. Edge sharpness (blurry object boundaries)
  3. Specular accuracy (reflections look correct?)
  4. Temporal consistency (4DGS only — popping/flickering?)
  5. Overall preference

  Comparison design:
  └─ 2AFC pairwise: "which looks better?" for each artifact type
  └─ 2,000 total comparisons across all categories
  └─ Each comparison judged by 5 workers
  └─ Budget: 2000 × 5 × $0.04 = ~$400 (higher per-task for quality)
  └─ Platform: Prolific (better quality than MTurk for visual tasks)
  └─ 10% catch trials with obviously degraded images

Phase 3 — Train Quality Metric (Month 4)

  └─ Same approach as Q2-A Part 3:
     LPIPS backbone + fine-tuned top layers + Bradley-Terry comparison head
  └─ Train 5 separate metric heads (one per artifact type) + 1 overall
  └─ Cross-validation: 5-fold on the 2,000 comparisons
  └─ Report: prediction accuracy, Kendall's τ with human rankings

Phase 4 — Re-rank Methods (Month 5)

  └─ For each of the 15 methods: compute mean metric score across all scenes
  └─ Compare ranking under PSNR vs. SSIM vs. LPIPS vs. our metric
  └─ The MAIN FINDING: methods that rank high on PSNR but low on 
     human preference (or vice versa)
  └─ Per-artifact analysis: which methods have worst floaters? 
     Best edges? etc.
  └─ This table IS the paper's contribution
```

### Experiment Design

| Experiment | What it shows |
|---|---|
| Metric prediction accuracy (held-out comparisons) | Our metric predicts human preference better than PSNR/SSIM/LPIPS |
| Method re-ranking: PSNR rank vs. human rank | Rankings disagree — motivates the whole paper |
| Per-artifact analysis | Different methods fail in different ways |
| Cross-dataset generalization | Train metric on Mip-NeRF 360, test on Tanks & Temples |
| Inter-annotator agreement (Krippendorff's α) | Study is reliable |
| Temporal consistency evaluation (4DGS methods only) | Unique contribution — no prior benchmark covers this |

### Month-by-Month Timeline (8 months)

| Month | Milestone |
|---|---|
| 1–2 | Run first 8 methods on all datasets. Automate rendering pipeline. |
| 3 | Run remaining 7 methods. Design and pilot subjective study (10 comparisons, internal). |
| 4 | Full crowdsourcing study on Prolific. Collect + clean data. |
| 5 | Train metric. Compute re-rankings. Generate all analysis tables. |
| 6–7 | Write paper. Strong emphasis on findings and analysis. |
| 8 | Submit to WACV / IEEE TVCG. |

### What Could Go Wrong

- **Running 15 methods is a huge engineering effort.** Mitigation: use nerfstudio (supports many methods via config), and for methods not in nerfstudio, use their official codebases with the same dataset preprocessing. Start early, parallelize.
- **Some methods don't have open-source code.** Mitigation: only include methods with public code. 15 is aspirational; 10 is sufficient for a strong paper.
- **Reviewers say "just a benchmark, no algorithmic contribution."** Mitigation: the trained metric IS a contribution (can be used by future work). Frame the paper as "community infrastructure." Benchmark papers at WACV/TVCG are well-received.

---

## Q3-C: Stylized 3DGS for Artistic Scene Rendering

### Core Technical Bet

Apply style transfer to **selected regions only** (e.g., "Van Gogh style on the walls, leave the person photorealistic") using language-guided segmentation from LangSplat/Gaussian Grouping + CLIP-based style loss. No prior work supports spatial control over 3D style transfer.

### Architecture

```
Input: trained 3DGS scene + style image + text region query

Step 1 — Region Selection
  └─ Option A: Use Gaussian Grouping to segment scene into objects
     Query: "walls" → get mask of wall Gaussians
  └─ Option B: Use LangSplat to query with text → relevancy scores per Gaussian
     Threshold at 0.5 → binary mask of Gaussians to stylize
  └─ Output: binary mask M (1 = stylize this Gaussian, 0 = keep photorealistic)

Step 2 — Style Feature Extraction
  └─ Run VGG-19 (pretrained, frozen) on style image
  └─ Extract Gram matrices at layers relu1_1, relu2_1, relu3_1, relu4_1
  └─ These are the style targets

Step 3 — Selective Style Optimization
  └─ Freeze all Gaussians where M=0 (photorealistic regions: locked)
  └─ For Gaussians where M=1, optimize SH coefficients only (not position/scale/rotation)
  └─ Render the scene from N=20 training viewpoints
  └─ For each render, extract VGG features and compute:
     
     L_style = Σ_layers ||Gram(content_layer) - Gram(style_layer)||_F²
     (computed only in the masked region of each rendered view)
     
     L_content = ||VGG_features(rendered) - VGG_features(original)||₂²
     (light content preservation — don't stray too far from original structure)
     
     L_total = L_style + 0.01 × L_content

  └─ Optimize for 500 iterations, ~5 min per edit on T4
  └─ Because we optimize Gaussians (not 2D images), the style is
     automatically multi-view consistent — render from any angle
```

### Alternative: CLIP-Based Style (for text-only style specification)

```
If no style image is provided, use text-based style:
  └─ Text prompt: "in the style of Van Gogh's Starry Night"
  └─ Loss = -cosine_similarity(CLIP(rendered_region), CLIP(text_prompt))
  └─ This is less precise than Gram matrix matching but more flexible
  └─ Can combine: L_total = L_gram_style + 0.5 × L_clip_style
```

### Experiment Design

| Experiment | What it shows | Data |
|---|---|---|
| Main: spatially controlled style vs. StyleGaussian (global style) | Our selective editing preserves photorealism outside the region | Mip-NeRF 360 (5 scenes) × 3 styles |
| Multi-view consistency test | Same style appearance from all angles | Render 360° video, compute frame-to-frame SSIM in styled region |
| Multiple regions, different styles | "Walls: Van Gogh, ceiling: Mondrian, floor: keep real" | 2 indoor scenes |
| User study: 15 participants, A/B preference | Humans prefer spatial control | 10 style edits |
| Speed benchmark | <5 min per edit | All experiments |
| Ablation: remove content loss | Style destroys structure without it | 2 scenes |
| Ablation: VGG style vs. CLIP style vs. both | Which approach gives better results | 3 scenes × 3 styles |

### Month-by-Month Timeline (5 months)

| Month | Milestone |
|---|---|
| 1 | Reproduce StyleGaussian on Mip-NeRF 360. Set up Gaussian Grouping or LangSplat for segmentation. |
| 2 | Implement selective style optimization (freeze masked Gaussians, VGG Gram loss on region). |
| 3 | Add CLIP-based style loss. Test multi-region editing. Tune content loss weight. |
| 4 | Full experiments + user study. |
| 5 | Write and submit to ACM MM / WACV. |

### What Could Go Wrong

- **VGG Gram loss produces texture-level style but misses structural style** (e.g., Van Gogh's brushstrokes vs. just his color palette). Mitigation: use multi-scale Gram matrices; add style loss at higher VGG layers.
- **Region boundary is visible** (sharp transition between styled and photorealistic). Mitigation: use a soft mask (Gaussian blur on the binary mask) and blend style strength at the boundary.
- **Reviewer says "trivial combination of existing methods."** Mitigation: the spatial control is genuinely new for 3DGS. Emphasize the multi-region, multi-style capability and the user study.

---
---

# Cross-Topic Summary

| Topic | Core Technical Bet | Hardest Part | Fastest Win |
|---|---|---|---|
| Q1-A | Multi-view diffusion → consistent edits | Zero123++ generalization to scenes | MECS metric is novel regardless |
| Q1-B | Lightweight backbone + depth prior + pose-free | Joint pose+Gaussian training stability | Efficiency table is undeniable |
| Q1-C | Static/dynamic decomposition + shared deformation | COLMAP on dynamic video | Minute-long video demo is impressive |
| Q2-A | Human preference metric for 3DGS | Crowdsourcing quality control | Metric + re-ranking is the paper |
| Q2-B | Incremental CLIP distillation during SLAM | Real-time semantic processing | Open-vocabulary query demo |
| Q2-C | Physics-based refraction in rasterizer | CUDA kernel modification | Custom transparent dataset |
| Q2-D | Deformation + specular decomposition | Dataset access registration | Low compute, clear MICCAI fit |
| Q3-A | Confidence-weighted noisy depth fusion | Proving it's not incremental | 6-month timeline, very doable |
| Q3-B | Benchmark + perceptual metric | Running 15 methods | High citation potential |
| Q3-C | Spatially-controlled 3D style transfer | Reviewer "trivial combo" criticism | 5-month timeline, fun demos |