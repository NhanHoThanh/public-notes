---
title: "3D Gaussian Splatting: The Quietly Massive Shift in 3D Graphics"
date: 2026-07-01
tags: [graphics, AI, 3D, research]
publish: true
---

Three years ago, creating a photorealistic 3D scene from photos meant either waiting hours for a NeRF to train — with no real-time playback — or spending weeks on a professional photogrammetry pipeline. Today you can walk around a room with your phone, upload the video to Polycam, and get a fully interactive 3D scene back in minutes, viewable in a browser at 60+ FPS. That shift has a name: **3D Gaussian Splatting** (3DGS).

## What it actually is

Instead of representing a scene as a mesh (triangles + textures) or as a neural network (NeRF), 3DGS uses millions of tiny 3D ellipsoids — Gaussians. Each one stores its position, shape, opacity, and view-dependent color via spherical harmonics. To render a frame, the engine projects all these ellipsoids onto the 2D image plane, sorts them by depth, and alpha-blends them together.

That last part is the key insight: it's *rasterization*, not ray-marching. GPUs have been optimized for rasterization for 30 years. So 3DGS gets NeRF-quality visuals at rasterization speeds — real-time, on consumer hardware.

The technique came from four researchers at INRIA, published at SIGGRAPH 2023. Within 12 months, over 1,000 follow-up papers appeared.

## Where it's already shipping

The adoption has been faster than most graphics technologies:

- **Apple Maps** announced 3DGS Flyover at WWDC June 2026 — 300+ global locations, replacing the photogrammetry models that have been there for years.
- **Zillow SkyTours** put 3DGS in front of millions of home buyers. Smooth walkthroughs from a smartphone capture, no LiDAR required.
- The 2025 film *Superman* used dynamic 4DGS in its VFX pipeline via Framestore — the first major feature film to do so.
- **OpenUSD v26.03** (March 2026) added a native Gaussian splat schema, integrating 3DGS into the same pipeline used by Pixar, Apple, NVIDIA, and Adobe.
- Nuke 17.0 (The Foundry) ships native 3DGS import for compositors.

## How it compares to what came before

| | NeRF | 3DGS | Mesh/Photogrammetry |
|---|---|---|---|
| Rendering speed | Seconds/frame | Real-time (60+ FPS) | Real-time |
| Training time | Hours–days | Minutes–hours | Hours–weeks |
| Photorealism | High | High | Medium–High |
| Web-deployable | Poor | Yes | Yes |
| Editable by artists | Very limited | Limited | Full |

3DGS essentially occupies the gap that NeRF opened up — photorealistic quality from unstructured photos — while closing the rendering speed gap that made NeRF impractical for production.

## What's still hard

For all the momentum, a few real limitations remain:

**Editing** is the biggest one. With a mesh, an artist can sculpt, cut, and texture-paint freely. With Gaussians, meaningful geometric edits are still largely unsolved. You can change the scene, but you can't really modify it the way you'd modify a model in Blender.

**Transparent and reflective surfaces** break the appearance model. Mirrors, glass, and water require special handling that isn't yet standardized.

**File format fragmentation** is a practical annoyance — the ecosystem splits across PLY, SPLAT, KSPLAT, SPZ, and SOG. OpenUSD v26.03 is the most credible push toward a standard, but interoperability between tools is still inconsistent.

**Large outdoor scenes** require multi-GPU setups and careful engineering. Capturing a city block is tractable; capturing a city is not yet.

## The frontier

The active research areas right now are genuinely interesting:

- **Feed-forward reconstruction**: instead of optimizing per-scene, a neural network predicts Gaussian parameters from 2–4 input images in under a second (GS-LRM, Apple SHARP). This means near-instant 3D from a handful of photos.
- **Generative 3DGS**: text or image in, a full 3D Gaussian scene out. World Labs' Marble and Apple's SHARP are early versions of this.
- **4DGS**: Gaussians that persist and move over time. Deployed in *Superman*'s VFX; not yet consumer-accessible.
- **Surgical robotics**: SurgicalGS (MICCAI 2025) reconstructs deformable tissue in real time from endoscope video. Still research, but the potential for surgical training and guidance is significant.

## Why it matters beyond graphics

3DGS is collapsing the cost of photorealistic 3D content creation by roughly two orders of magnitude — from weeks and specialist pipelines to minutes and a smartphone. That's the kind of shift that changes who can afford to build in 3D.

E-commerce product visualization, real estate, cultural preservation, autonomous driving simulation, surgical training — the economics of all of these change when high-quality 3D capture is accessible to anyone with a phone. Apple Maps' adoption of 3DGS for 300+ cities is the clearest signal yet that this is graduating from a research technique to infrastructure.

The editing gap will close, the format fragmentation will resolve, and the mobile pipeline will get smoother. The underlying bet — that explicit, differentiable 3D representations are the right abstraction for this moment — looks increasingly well-placed.
