# Compact, Physics-Aware Dynamic 3D Gaussian Representations for Online Perception and Interaction

> **Literature review and research-gap map — as of 2026-07-12.** Core evidence: 78 unique technical works. Detailed records, status labels, evaluation fields, and scope boundaries are in the [companion evidence matrix](quartz/content/3dgs%201/physics-aware-dynamic-3dgs-evidence-matrix.md).

---

## Abstract

3D Gaussian Splatting (3DGS) changed neural scene representation by combining an explicit, optimizable set of anisotropic particles with fast differentiable rasterization. That combination made high-quality rendering fast enough to become a shared substrate for dynamic reconstruction, dense SLAM, compression, scene editing, robotic perception, and physical simulation. The same explicitness is also deceptive: a Gaussian has a position, covariance, opacity, and appearance, but it does not automatically have persistent object identity, mass, material parameters, contacts, or a causal motion model. Most “dynamic 3DGS” papers therefore reconstruct observed time; they do not predict how a scene will respond to a new action or force.

Across 78 core works, the literature forms six connected but still weakly integrated strands. Offline dynamic rendering is the most mature: canonical deformation fields, explicit trajectories, native 4D primitives, and topology-aware surfaces all achieve strong novel-view synthesis. Static Gaussian SLAM is also established, including global correction and loop closure. Physics-aware work now supports continuum simulation, inverse material fitting, rigid-contact identification, and deformable digital twins. Compression has progressed from pruning and vector quantization to contextual entropy models, progressive bitstreams, and temporal codecs. Human interaction is effective for static semantic queries and diffusion-guided editing, while robot interaction ranges from language grounding and grasping to early action-conditioned world models.

The missing system is not another isolated improvement in one strand. It is a compact, uncertainty-aware representation that can map a scene online, maintain camera and object states, infer deformable material/contact state, predict interventions, and stream both appearance and solver state under a real-time budget. Current work typically solves at most two or three of those requirements simultaneously. The highest-confidence research gaps are therefore at the intersections: physics-aware deformable SLAM, compression of physical state and contact topology, persistent action-conditioned object maps, topology-aware lifecycle models, and benchmarks that measure geometry, dynamics, task success, and bandwidth together.

## Review Questions and Method

This review answers six questions:

1. **Representation:** How are dynamic, articulated, and genuinely deformable objects represented with Gaussians?
2. **Estimation:** Which systems reconstruct offline, stream online, or jointly perform SLAM and object-state estimation?
3. **Physics:** Which methods are merely motion-regularized, and which encode or identify forces, constitutive laws, material parameters, and contact?
4. **Efficiency:** What is compressed—appearance, geometry, motion, or solver state—and which costs are actually reported?
5. **Interaction:** How do human semantic/editing interfaces differ from robot perception, action prediction, and contact-rich control?
6. **Research opportunity:** Which combinations are still unsupported, measurable, and feasible for a 6–12 month student project?

The search used the local notes as a seed bibliography, followed by multiple semantic searches rather than one literal query. Query families covered dynamic/deformable 3DGS, non-rigid and topology-changing objects, Gaussian SLAM, continuum and contact physics, physical optics, static/dynamic compression, robotic manipulation, language fields, and editing. Strong anchors were expanded through related-paper, reference, and citer searches. Canonical metadata was checked against the official arXiv API; representative papers in every strand were read at full-text passage level.

The core window is August 2023 through July 12, 2026. Six older roots were retained because they define dynamic radiance fields, topology-changing representations, non-rigid reconstruction, feature SLAM, or continuum system identification. Inclusion required a technical method and evaluation relevant to at least one review question. Duplicate preprint/proceedings versions were merged. Blogs and vendor claims were excluded from novelty evidence. If a venue could not be confirmed, the matrix conservatively labels the work as a preprint. This is a reproducible scoping review and gap map, not a claim of exhaustive systematic coverage.

## A Unifying Taxonomy

The literature becomes clearer when methods are classified by state and causality rather than by whether “4D” appears in the title.

| Axis                    | Levels                                                                                                                                      | What changes between levels                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Temporal representation | Per-frame sets → persistent trajectories → canonical deformation → native 4D primitives → object/surface models                             | Storage, identity, interpolation, and topology handling                             |
| Estimator               | Offline fitting → on-the-fly streaming → camera SLAM → camera + rigid-object SLAM → non-rigid 4D SLAM                                       | Whether poses and scene state are observed, optimized jointly, and available online |
| Dynamics                | Appearance-only → smooth/rigid regularization → learned action-conditioned motion → constitutive simulation → inverse system identification | Whether motion is descriptive, predictive, or physically identifiable               |
| Object model            | Static → rigid → articulated → deformable → topology-changing                                                                               | Required state, constraints, and correspondence assumptions                         |
| Interaction             | Passive viewing → semantic query/edit → grasp/affordance → action-conditioned prediction → contact-aware planning/control                   | Whether an intervention changes future state                                        |
| Efficiency              | Pruning/quantization → contextual entropy coding → progressive/LOD → temporal residual coding → solver-state/contact coding                 | Which information is budgeted and whether partial decoding is useful                |

Three distinctions should remain explicit throughout:

- **Map deformation is not dynamic-scene modeling.** [Splat-SLAM](https://arxiv.org/abs/2405.16544) deforms a nominally static Gaussian map after global pose/depth correction; [DynaGSLAM](https://arxiv.org/abs/2503.11979) keeps explicit moving-object state; [4DTAM](https://arxiv.org/abs/2505.22859) estimates a non-rigid surface field with ego-motion.

- **Motion regularity is not physics.** Optical-flow losses, ARAP constraints, splines, local rigidity, and Wasserstein smoothness improve correspondence but do not identify mass, stiffness, friction, collision response, or unseen-force behavior. Constitutive or contact state appears in work such as [PhysGaussian](https://arxiv.org/abs/2311.12198), [PhysFlow](https://arxiv.org/abs/2411.14423), [PIN-WM](https://arxiv.org/abs/2504.16693), and [PhysTwin](https://arxiv.org/abs/2503.17973).

- **Fast rendering is not a real-time system.** A representation may render at hundreds of FPS while reconstruction, segmentation, entropy decoding, global optimization, or physical parameter fitting takes seconds to hours. End-to-end latency must include sensing, state estimation, update, decode, simulation, and control.

### Status-quo map

![Two horizontal bar charts showing the number of reviewed works by primary research strand and by overlapping capability tag. Dynamic and deformable work is the largest group, while physics and interaction form smaller subsets.](quartz/content/3dgs%201/assets/physics-aware-dynamic-3dgs/01-status-quo-corpus.png)

**Figure 1 — Corpus and capability coverage.** The left panel assigns each of the 78 works to its evidence-matrix section; the right panel counts overlapping tags. The imbalance makes the current status quo visible: the field has a deep base in time-varying visual reconstruction, but substantially fewer works connect that base to physics, SLAM, or interaction. Counts describe the review corpus, not the total global publication volume.

## Foundations: Why Gaussians Became the Common Substrate

[NeRF](https://arxiv.org/abs/2003.08934) established photorealistic view synthesis with an implicit radiance field, while [D-NeRF](https://arxiv.org/abs/2011.13961) and [HyperNeRF](https://arxiv.org/abs/2106.13228) introduced canonical deformation and higher-dimensional topology handling. [VolumeDeform](https://arxiv.org/abs/1603.08161) showed real-time RGB-D non-rigid reconstruction with an explicit volumetric deformation field, and [ORB-SLAM3](https://arxiv.org/abs/2007.11898) established robust camera-state estimation across visual and inertial configurations. [PAC-NeRF](https://arxiv.org/abs/2303.05512) demonstrated that rendering observations can support continuum-system identification, although with an implicit field and calibrated multi-view capture.

[3DGS](https://arxiv.org/abs/2308.04079) replaced repeated neural ray queries with explicit anisotropic primitives, adaptive densification, visibility sorting, and alpha compositing. This made each primitive easy to move, prune, quantize, tag semantically, or reinterpret as a simulation particle. It did not solve surface completeness or volumetric state: splats cluster where appearance gradients support reconstruction, interiors are usually empty, and covariance is a rendering footprint rather than a material tensor. Every physics-aware extension must therefore decide whether to fill interiors, bind splats to a mesh/surface, pair them with separate particles, or accept a surface-biased approximation.

## 1. Dynamic Objects: Reconstructing Observed Time

### Persistent identity and explicit trajectories

[Dynamic 3D Gaussians](https://arxiv.org/abs/2308.09713) keeps persistent primitives and optimizes their time-varying position and rotation, producing dense tracks as well as novel views. The benefit is explicit identity; the cost is synchronized multi-view capture and a fixed-identity assumption. [Spacetime Gaussian Feature Splatting](https://arxiv.org/abs/2312.16812), [4D-Rotor](https://arxiv.org/abs/2402.03307), and [Fully Explicit Dynamic Gaussian Splatting](https://arxiv.org/abs/2410.15629) replace or reduce deformation-network queries with temporal opacity, parametric trajectories, native 4D covariance, or explicit motion parameters. These methods improve throughput and compactness, but duration and motion complexity still increase representation cost.

### Canonical deformation fields

[Deformable 3D Gaussians](https://arxiv.org/abs/2309.13101) and [4D Gaussian Splatting](https://arxiv.org/abs/2310.08528) learn a shared canonical set and map it through time. Canonicalization is compact and interpolatable, but it entangles camera error, occlusion, appearance changes, and object motion inside one learned warp. [MotionGS](https://arxiv.org/abs/2410.07707) explicitly decouples camera flow from object flow, showing that better motion supervision helps, yet optical flow remains an image-space kinematic cue rather than a causal model.

### Editable, structured, and topology-aware motion

[SC-GS](https://arxiv.org/abs/2312.14937) drives dense Gaussians from sparse 6-DoF control nodes and supports user-authored motion edits. Its ARAP prior is useful but kinematic. [SplineGS](https://arxiv.org/abs/2412.09982) prunes spline control points for compact monocular trajectories, but smooth splines cannot naturally express impact discontinuities or topology change. [GauSTAR](https://arxiv.org/abs/2501.10283) takes a more structural approach: Gaussians bind to mesh faces, unbind when topology changes, and guide selective remeshing. This is one of the clearest demonstrations that lifecycle/topology state must be explicit rather than hidden in a continuous warp.

### Efficiency enters the dynamic model

[ADC-GS](https://arxiv.org/abs/2505.08196) shares motion through anchors and performs rate–distortion optimization; [SpeeDe3DGS](https://arxiv.org/abs/2506.07917) uses temporal-sensitivity pruning and groupwise SE(3) motion. Both show that deformation and compression should be designed together. Neither encodes material/contact state, and group motion can erase precisely the local degrees of freedom needed for cloth, tissue, or collision.

**Consensus.** The field is converging on shared structures—anchors, control nodes, groups, splines, or canonical fields—to avoid one neural query or one full parameter set per Gaussian per frame. Persistent identity and static/dynamic decomposition improve both quality and efficiency.

**Unresolved issue.** These representations interpolate and reconstruct; they usually cannot extrapolate under a new force, explain why motion occurred, or guarantee conservation/contact constraints. “Dynamic” should therefore be read as *time-indexed visual state* unless a paper evaluates interventions or identified physical parameters.

## 2. Deformable Objects: From Visual Warp to Material State

Deformation spans several problems that should not share one benchmark. Articulated objects have low-dimensional joint constraints; cloth and tissue have continuum or discrete elastic state; humans often use a body prior; topology-changing scenes require primitive birth/death or remeshing.

### Surgical tissue

[EndoGaussian](https://arxiv.org/abs/2401.12561), [EH-SurGS](https://arxiv.org/abs/2501.01101), and [SurgicalGaussian](https://arxiv.org/abs/2407.05023) demonstrate fast Gaussian reconstruction of deforming endoscopic scenes. They introduce depth priors, deformation hierarchies, tool masks, surface regularization, or Gaussian life cycles. Their success is operationally important, but the recovered warp is generally supervised by images rather than measured forces. Specular light, tool occlusion, narrow baselines, unknown tissue properties, irreversible cutting/shearing, and the absence of deformation ground truth make “physical correctness” hard to establish.

[SimEndoGS](https://arxiv.org/abs/2405.00956) is a stronger bridge to mechanics: it fills reconstructed tissue, applies MPM with a Neo-Hookean model, and permits user forces. Yet its material parameters and boundary conditions are supplied manually, so it demonstrates forward plausibility rather than patient-specific identification.

### Cloth, articulated objects, and general soft bodies

[PBDyG](https://arxiv.org/abs/2412.04433) uses position-based dynamics for movement-dependent cloth on a human body prior. [ArtGS](https://arxiv.org/abs/2502.19459) constructs interactable articulated replicas from two RGB-D articulation states and estimates part/joint kinematics. These are useful specializations, but neither gives a general deformable object model.

[PhysTwin](https://arxiv.org/abs/2503.17973) is closer to the target: a generative complete-shape prior, Gaussian appearance, spring–mass mechanics, contact, and differentiable parameter fitting are evaluated across ropes, cloth, stuffed objects, and unseen interactions. Its three-view RGB-D capture and per-object optimization show the remaining cost of turning a visual asset into a predictive digital twin.

**Main gap.** No common representation simultaneously supports partial observation, persistent correspondence, topology change, heterogeneous constitutive state, contact, and online update. The visual reconstruction literature is strongest on appearance and interpolation; the physics literature is strongest in controlled, object-centric real-to-sim settings.

## 3. SLAM: Static Maps, Dynamic Outliers, and Non-Rigid State

### Static Gaussian SLAM is established

[SplaTAM](https://arxiv.org/abs/2312.02126), [GS-SLAM](https://arxiv.org/abs/2311.11700), [Gaussian Splatting SLAM/MonoGS](https://arxiv.org/abs/2312.06741), and [Photo-SLAM](https://arxiv.org/abs/2311.16728) show that Gaussians can serve as dense map, photometric model, and sometimes the tracking representation. Their evaluations combine trajectory error with view-synthesis metrics, but most early systems assume a static room and lack scalable loop closure.

[Splat-SLAM](https://arxiv.org/abs/2405.16544) propagates globally optimized pose/depth updates into associated Gaussians. [LoopSplat](https://arxiv.org/abs/2408.10154) organizes RGB-D Gaussians into submaps and registers them for loop closure. These solve global consistency for a nominally static map; calling the correction “Gaussian deformation” does not make them dynamic-scene systems.

### Dynamic SLAM has three distinct goals

1. **Anti-dynamic SLAM** identifies and removes movers so that the camera and static background remain stable. [DGS-SLAM](https://arxiv.org/abs/2411.10722), [DG-SLAM](https://arxiv.org/abs/2411.08373), and [GARAD-SLAM](https://arxiv.org/abs/2502.03228) belong here. They improve ATE and reduce ghosting, but throw away information useful for interaction.
2. **Dynamic-object SLAM** keeps separate object state. [DynaGSLAM](https://arxiv.org/abs/2503.11979) tracks/renders moving Gaussians and extrapolates short motion horizons. This is closer to robotics, but relies on depth, segmentation, and flow, and its object model is not articulated or contact-aware.
3. **Non-rigid 4D SLAM** jointly estimates a deforming map and ego-motion. [4DTAM](https://arxiv.org/abs/2505.22859) uses surface Gaussians plus a warp field and evaluates both geometry and trajectory. It remains small-scene and roughly 1.5 FPS, highlighting how much harder non-rigid observability is than masking motion.

### What SLAM still lacks

A deployable system must decide which state persists through occlusion, loop closure, and interaction. Current systems rarely maintain uncertainty over static/dynamic membership, object identity, deformation, and material parameters together. They also report rendering FPS more often than end-to-end sensor-to-map latency. Long-horizon memory becomes a joint compression and data-association problem, not merely a renderer problem.

## 4. Physics Awareness: Simulation, Identification, and Optics

> **Detailed expansion:** [Physics Awareness in 3D Gaussian Splatting: Simulation, Identification, and Physical Image Formation](quartz/content/3dgs%201/physics-aware-3dgs-simulation-identification-optics-literature-review.md) reviews 115 strict Gaussian-centric work families through 2026-07-20, including paywalled novelty, exact recovered hardware constraints, an [auditable matrix](quartz/content/3dgs%201/physics-aware-3dgs-simulation-identification-optics-evidence-matrix.md), and six generated status-quo figures.

### Forward mechanics

[PhysGaussian](https://arxiv.org/abs/2311.12198) established the “what you see is what you simulate” formulation: Gaussian kernels are also MPM material points carrying kinematic and stress state. It demonstrates elastic/plastic solids, granular media, fluids, fracture, and collision. Its key limitation is equally clear: the simulation region, interior filling, constitutive model, boundary conditions, and parameters are manually chosen.

[SimEndoGS](https://arxiv.org/abs/2405.00956) applies this idea to tissue, while [OmniPhysGS](https://arxiv.org/abs/2501.18982) expands generative dynamics to heterogeneous constitutive behavior. These methods make physically plausible animation easier, but forward simulation with chosen parameters is not system identification.

### Inverse physical parameter estimation

[PAC-NeRF](https://arxiv.org/abs/2303.05512) is the essential root: rendering observations supervise geometry and continuum parameters jointly. [PhysDreamer](https://arxiv.org/abs/2404.13026) and [PhysFlow](https://arxiv.org/abs/2411.14423) use video-generation or optical-flow priors to compensate for missing material ground truth. Their output is often plausible, but a generative prior can hide non-identifiability: different stiffness, density, damping, force, and boundary combinations may yield similar pixels.

[PIN-WM](https://arxiv.org/abs/2504.16693) provides unusually concrete contact/control evidence for rigid objects by identifying mass/inertia, restitution, and friction and then evaluating pushing/flipping policies. [PhysTwin](https://arxiv.org/abs/2503.17973) supplies the stronger deformable counterpart through resimulation and unseen interactions. Both rely on controlled observation and substantial priors, which is precisely why physics-aware online SLAM remains open.

### Learned physics fields

[Physics-Informed Deformable Gaussian Splatting](https://arxiv.org/abs/2511.06299) treats time-varying Gaussians as a material field constrained by learned constitutive behavior. [Neural Gaussian Force Fields](https://arxiv.org/abs/2602.00148) learns force-conditioned 4D dynamics to reduce repeated explicit simulation. These approaches may improve speed and generalization, but must be evaluated on physical rollout, interventions, and recovered parameters—not only novel-view PSNR.

### Physical optics is adjacent, not equivalent

[3DGUT](https://arxiv.org/abs/2412.12507) supports nonlinear camera models and aligns Gaussian rendering with secondary rays; [TransparentGS](https://arxiv.org/abs/2504.18768) targets transparent-object geometry and refraction. These are important for perception because unmodeled optics corrupt geometry and tracking. They do not model mechanics, and the taxonomy should keep physical image formation separate from material dynamics.

## 5. Compression: From Static Assets to Dynamic Streams

### Static compression

[LightGaussian](https://arxiv.org/abs/2311.17245), [Compact 3DGS](https://arxiv.org/abs/2311.13681), and [EAGLES](https://arxiv.org/abs/2312.04564) established the main ingredients: importance pruning, learned masks, codebooks/vector quantization, reduced spherical harmonics, and lightweight attribute decoding. [HAC](https://arxiv.org/abs/2403.14530) and [HAC++](https://arxiv.org/abs/2501.12255) exploit anchor and hash-grid context for entropy coding. [RDO-Gaussian](https://arxiv.org/abs/2406.01597) makes rate–distortion an explicit training objective, while [PCGS](https://arxiv.org/abs/2503.08511) makes partial bitstreams useful through progressive decoding.

Published compression ratios are not directly comparable. Some count only Gaussian attributes; others include anchors, neural decoders, codebooks, or reference frames. Some optimize one rate point; others expose several. Decode latency and viewer compatibility are often less visible than model size and rendering FPS.

### Dynamic compression and streaming

[QUEEN](https://arxiv.org/abs/2412.04469) encodes sparse inter-frame residuals with dynamic/static separation. [V³/VideoGS](https://arxiv.org/abs/2409.13648) maps dynamic Gaussian attributes into 2D planes for hardware video codecs and demonstrates mobile playback. [4DGC](https://arxiv.org/abs/2503.18421) integrates motion-aware representation, quantization, and entropy modeling. [ADC-GS](https://arxiv.org/abs/2505.08196) and [SpeeDe3DGS](https://arxiv.org/abs/2506.07917) push compression into deformation structure itself.

These codecs compress visual and motion representations, not physical state. None of the core dynamic codecs jointly budgets stress, velocity, material parameters, contacts, uncertainty, and the appearance required to render them. A physically meaningful distortion measure is also absent: PSNR can remain high while contact timing, trajectory, volume preservation, or task success becomes wrong.

## 6. Interaction: Human Control and Robot Intervention

### Human semantic querying and editing

[LangSplat](https://arxiv.org/abs/2312.16084) distills hierarchical language features into Gaussians for fast open-vocabulary queries. [Gaussian Grouping](https://arxiv.org/abs/2312.00732) adds object identities for segmentation and edits. [GaussianEditor](https://arxiv.org/abs/2311.14521), [3DitScene](https://arxiv.org/abs/2405.18424), [TIGER](https://arxiv.org/abs/2405.14455), [3DSceneEditor](https://arxiv.org/abs/2412.01583), and [EditSplat](https://arxiv.org/abs/2412.11520) improve controllability and multi-view consistency.

This strand is effective for a static, already reconstructed scene. It usually depends on SAM/CLIP or 2D diffusion priors, per-scene feature fitting, and per-edit optimization. Semantic identity is rarely maintained through deformation, occlusion, topology change, or loop closure, and edits are not constrained by stability, collision, or material response.

### Robot perception, grasping, and scene updates

[GaussianGrasper](https://arxiv.org/abs/2403.09637), [Splat-MOVER](https://arxiv.org/abs/2405.04378), [GraspSplats](https://arxiv.org/abs/2409.02084), [SparseGrasp](https://arxiv.org/abs/2412.02140), and [3DAffordSplat](https://arxiv.org/abs/2504.11218) use semantic/affordance features and explicit geometry for language-conditioned grasping. The representation speeds queries and enables localized scene updates, but segmentation, grasp proposals, and tracking are commonly supplied by separate foundation models. This is interaction-aware perception, not necessarily interaction dynamics.

### Action-conditioned world models

[ManiGaussian](https://arxiv.org/abs/2403.08321) predicts action-conditioned Gaussian deformation to supervise manipulation in simulation. [Dynamic Gaussian Tracking for Graph-Based Neural Dynamics](https://arxiv.org/abs/2410.18912) converts dense tracks into control particles, learns graph dynamics for ropes/cloth/soft objects, and uses predictions for model-based planning. [GAF](https://arxiv.org/abs/2506.14135) stores action/motion attributes in a 4D field, while the robotics [ArtGS](https://arxiv.org/abs/2507.02600) integrates articulated visual and physical parameters.

These systems begin to connect perception and intervention, but training distributions are narrow, sensing is often multi-view, contacts are simplified, and uncertainty is rarely propagated into planning. The next step is not merely a better rendered future; it is a calibrated distribution over object/contact states that supports safe closed-loop action.

## Themes and Consensus

1. **Shared structure is the main efficiency lever.** Anchors, motion groups, control nodes, splines, submaps, and object identities all reduce redundant per-Gaussian state.
2. **Explicit Gaussians are an integration interface, not a complete physical model.** Their differentiability and render speed make it easy to attach semantics or simulation state, but geometry completion and interior material state require extra machinery.
3. **Static/dynamic decomposition helps every strand.** It reduces SLAM corruption, deformation queries, codec bitrate, and simulation cost. Hard binary separation is fragile around articulated or deformable boundaries.
4. **Evaluation remains siloed.** Rendering uses PSNR/SSIM/LPIPS; SLAM uses ATE/RPE; compression uses bytes and FPS; physics uses parameter/trajectory error or user preference; robotics uses task success. Few papers report more than two families together.
5. **Causal evidence is rare.** A method deserves “physics-aware” most strongly when it predicts measured unseen interventions, recovers identifiable parameters, or improves closed-loop task outcomes—not merely when motion looks plausible.

![Heatmap showing the percentage and count of papers in each research strand that report appearance, geometry, state-estimation, physical-behavior, efficiency, or task-outcome metrics. Each strand concentrates on its own metric family.](quartz/content/3dgs%201/assets/physics-aware-dynamic-3dgs/02-metric-landscape.png)

**Figure 2 — Metric landscape.** A cell records whether a paper's evidence field reports at least one metric in that family; it does not score the paper's performance. Keyword coding makes the siloing concrete: SLAM consistently measures state estimation, compression consistently measures resource cost, and robot work most often measures task outcomes, while joint coverage remains sparse. A credible physics-aware dynamic system should report across these columns rather than selecting only its home strand's metrics.

## Open Questions and Debates

### Canonical deformation or native 4D state?

Canonical fields are compact and interpolate well but struggle with topology and identity changes. Native 4D or per-frame state handles appearance changes and discontinuities but increases storage. Surface binding/lifecycles, as in GauSTAR and EH-SurGS, may be a better middle ground for objects whose topology genuinely changes.

### Physical truth or perceptual plausibility?

Video-prior methods can synthesize convincing motion without unique physical parameters. For entertainment, perceptual preference may be enough. For robotics, surgery, or scientific use, parameter uncertainty, resimulation error, contact timing, and intervention success are necessary.

### Remove moving objects or model them?

Anti-dynamic SLAM is the safest route to camera localization, but it discards state needed for collision avoidance and manipulation. Persistent object maps increase data association and memory risk. A useful hybrid would preserve high-confidence object tracks while marginalizing ambiguous movers from camera estimation.

### What exactly counts as compression?

Model size, stream bitrate, GPU memory, training memory, decode time, update bandwidth, and rendering cost are different budgets. A codec that saves disk space but requires a heavy neural decoder may be unsuitable for SLAM or mobile control. Future work should report an explicit accounting boundary.

### How much geometry is enough for physics?

Photorealistic splats can be sparse inside objects and inaccurate at occluded surfaces. Physics needs volume, collision geometry, and material connectivity. Mesh binding, generative completion, particle filling, and hybrid Gaussian–particle models make different failure tradeoffs; none is universally reliable.

![Qualitative capability heatmap comparing dynamic rendering, Gaussian SLAM, physics-aware Gaussians, dynamic compression, and robot action models across online estimation, persistent identity, deformation, material identification, contact, rate accounting, intervention prediction, task evaluation, and uncertainty. No strand covers all capabilities.](quartz/content/3dgs%201/assets/physics-aware-dynamic-3dgs/03-capability-gap-map.png)

**Figure 3 — Capability-gap map.** This is a qualitative synthesis, not a paper-count statistic. “Established” means a capability is routine in that strand; “emerging” means multiple concrete systems exist; “rare” denotes isolated or limited evidence; and “missing” means the review found no strand-level demonstration. The nearly empty vertical slices for calibrated uncertainty and the weak coupling between online mapping, physical identification, compression, and control explain why integration—not another isolated renderer improvement—is the main gap.

## Emerging Trends

- **Non-rigid SLAM is becoming explicit.** 4DTAM moves beyond masking into joint ego-motion and deformation, but is still small and slow.
- **Topology is becoming state.** GauSTAR remeshing and EH-SurGS life cycles treat birth/death explicitly instead of forcing a continuous warp.
- **Physics is moving from forward animation to identification and control.** PhysFlow, PIN-WM, and PhysTwin evaluate recovered parameters, unseen interactions, or robot success more directly than early generative physics work.
- **Dynamic codecs are becoming stream-aware.** QUEEN, V³, and 4DGC exploit temporal residuals, hardware video coding, or explicit rate–distortion models.
- **Interaction is moving from semantics to action.** ManiGaussian, graph dynamics, GAF, and ArtGS attach actions or physical state to the Gaussian representation rather than using it only as a queryable map.

## Ranked Research Agenda

Scores use **gap confidence** (how clearly the literature leaves the problem open) and **student feasibility** under the folder’s assumed one-consumer-GPU, 6–12 month setting.

| Rank | Direction                                                       | Gap confidence | Feasibility | Minimum credible contribution and evaluation                                                                                                                                                           |
| ---- | --------------------------------------------------------------- | -------------: | ----------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | **Physics-state-aware dynamic 3DGS compression benchmark**      |            5/5 |         4/5 | Extend a dynamic codec to include velocity/material/contact state; measure bitrate versus rendering, trajectory, conservation/contact, and rollout error on PhysGaussian/PhysTwin-style scenes         |
| 2    | **Persistent dynamic-object Photo-SLAM**                        |            5/5 |         4/5 | Replace pure masking with a temporary/persistent object map; compare anti-dynamic and modeled-object variants on ATE, object tracks, ghosting, memory, and end-to-end latency                          |
| 3    | **Uncertainty-aware material identification from sparse views** |            5/5 |         3/5 | Add parameter posterior/confidence to a PhysFlow or PhysTwin-style inverse pipeline; test identifiability across view count, force ambiguity, occlusion, and material families                         |
| 4    | **Topology/lifecycle-aware deformable Gaussians**               |            4/5 |         3/5 | Combine Gaussian birth/death or surface unbinding with compact canonical motion; evaluate cutting, tearing, disocclusion, correspondence survival, storage, and temporal artifacts                     |
| 5    | **Action-conditioned object-centric GS-SLAM**                   |            5/5 |         2/5 | Jointly estimate camera, object, and action-conditioned future state; evaluate prediction calibration and MPC/task success, not only novel views                                                       |
| 6    | **Physics-aware endoscopic digital twin**                       |            4/5 |         3/5 | Infer tissue parameters or force response rather than manually choosing them; use surgical reconstruction quality plus deformation/resimulation error and uncertainty                                  |
| 7    | **Unified cross-strand benchmark**                              |            5/5 |         4/5 | Create common scenes and accounting rules for appearance, geometry, camera/object tracking, physics, bitrate, latency, and interaction success; include failure annotations and reproducible baselines |

![Scatter plot of seven research directions by gap confidence and student feasibility. Physics-state-aware compression, persistent dynamic-object Photo-SLAM, and a unified benchmark occupy the high-confidence, high-feasibility region.](quartz/content/3dgs%201/assets/physics-aware-dynamic-3dgs/04-research-agenda.png)

**Figure 4 — Research-direction frontier.** The point labels are the table ranks. Physics-state-aware compression, persistent dynamic-object Photo-SLAM, and the unified benchmark share the strongest confidence/feasibility score; uncertainty-aware material identification is compelling but harder, while action-conditioned object-centric GS-SLAM carries the greatest integration risk.

### Recommended starting point

For the best novelty-to-risk balance, start with **Rank 1** or **Rank 2**. A physics-state-aware compression benchmark can reuse existing simulators and codecs without solving online perception first. Persistent dynamic-object Photo-SLAM can reuse the existing Photo-SLAM analysis and dynamic masking pipeline while adding measurable object memory. Ranks 5 and 6 are more ambitious because errors in segmentation, geometry, state estimation, physics, and control compound.

## Key Papers and Reading Order

1. Representation roots: [3DGS](https://arxiv.org/abs/2308.04079), [D-NeRF](https://arxiv.org/abs/2011.13961), [HyperNeRF](https://arxiv.org/abs/2106.13228).
2. Dynamic representations: [Dynamic 3D Gaussians](https://arxiv.org/abs/2308.09713), [Deformable 3DGS](https://arxiv.org/abs/2309.13101), [4DGS](https://arxiv.org/abs/2310.08528), [GauSTAR](https://arxiv.org/abs/2501.10283).
3. SLAM transition: [SplaTAM](https://arxiv.org/abs/2312.02126), [Photo-SLAM](https://arxiv.org/abs/2311.16728), [Splat-SLAM](https://arxiv.org/abs/2405.16544), [4DTAM](https://arxiv.org/abs/2505.22859).
4. Physics: [PAC-NeRF](https://arxiv.org/abs/2303.05512), [PhysGaussian](https://arxiv.org/abs/2311.12198), [PhysFlow](https://arxiv.org/abs/2411.14423), [PIN-WM](https://arxiv.org/abs/2504.16693), [PhysTwin](https://arxiv.org/abs/2503.17973).
5. Compression: [HAC](https://arxiv.org/abs/2403.14530), [RDO-Gaussian](https://arxiv.org/abs/2406.01597), [QUEEN](https://arxiv.org/abs/2412.04469), [V³](https://arxiv.org/abs/2409.13648), [4DGC](https://arxiv.org/abs/2503.18421).
6. Interaction: [LangSplat](https://arxiv.org/abs/2312.16084), [Splat-MOVER](https://arxiv.org/abs/2405.04378), [ManiGaussian](https://arxiv.org/abs/2403.08321), [Graph-Based Gaussian Dynamics](https://arxiv.org/abs/2410.18912), [GAF](https://arxiv.org/abs/2506.14135).

## Sources

The [evidence matrix](quartz/content/3dgs%201/physics-aware-dynamic-3dgs-evidence-matrix.md) is the authoritative source index for this review. It records all 78 core works with stable IDs, publication status, tags, setting, method, evaluation evidence, code status, and scope boundary. Survey papers and local narrative notes were used only for discovery and were not counted as primary technical evidence.

All four figures are reproducible with [`generate_literature_review_visualizations.py`](quartz/content/3dgs%201/generate_literature_review_visualizations.py). Figures 1 and 2 are parsed from the evidence matrix, Figure 3 encodes the review's explicit qualitative maturity rubric, and Figure 4 uses the ranked-agenda scores above. This separation prevents qualitative judgments from being presented as measured paper counts.

One seed correction matters for future reruns: the previously recorded `arxiv:2402.11755` is unrelated to Gaussian rendering. The verified paper is [3DGUT, arxiv:2412.12507](https://arxiv.org/abs/2412.12507).

## Rerun Inputs
