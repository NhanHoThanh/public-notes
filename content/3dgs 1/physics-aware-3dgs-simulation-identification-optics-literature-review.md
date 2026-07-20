# Physics Awareness in 3D Gaussian Splatting: Simulation, Identification, and Physical Image Formation

> **Detailed literature review — evidence frozen 2026-07-20.** Core corpus: **115** Gaussian-centric work families (**90 E2 full-text, 25 E1 primary abstract/metadata**). Publication access, novelty, causal evidence, and hardware reporting are coded independently; a paywalled venue does not disqualify novelty, and a fast renderer does not by itself establish a deployable physics system.

**Companions:** [auditable evidence matrix](quartz/content/3dgs%201/physics-aware-3dgs-simulation-identification-optics-evidence-matrix.md) · [machine-readable dataset](quartz/content/3dgs%201/physics-aware-3dgs-simulation-identification-optics-data.json) · [Matplotlib generator](quartz/content/3dgs%201/generate_physics_aware_3dgs_review_assets.py)

## Abstract

Physics-aware 3D Gaussian Splatting (3DGS) is not one problem. It is a rapidly widening family of methods in which Gaussian primitives play at least four different roles: visible material points in a mechanical solver, a differentiable observation model for identifying latent physical parameters, a state representation for learned prediction and control, or a primitive for modeling light and sensor transport. Those roles share an explicit, differentiable scene representation, but they answer different scientific questions. Forward simulation asks whether a reconstructed or generated asset can move under a stated law. System identification asks whether the law, parameters, forces, or boundary conditions can be recovered from observations. Learned physics asks whether future state under an intervention can be predicted or controlled. Physics-based image formation asks whether emitted, reflected, refracted, scattered, or sensor-integrated radiance is modeled well enough to recover geometry, illumination, optical material, or camera state.

The status quo is promising but uneven. [PhysGaussian](https://arxiv.org/abs/2311.12198) established the influential dual-use formulation in which Gaussians are both renderable kernels and material points for continuum simulation. Subsequent work expanded to fluids, granular media, tissue, cloth, spring–mass systems, position-based dynamics, rigid contact, and heterogeneous materials. A smaller identification strand—including [PhysDreamer](https://arxiv.org/abs/2404.13026), [PhysFlow](https://arxiv.org/abs/2411.14423), [PhysTwin](https://arxiv.org/abs/2503.17973), and [PIN-WM](https://arxiv.org/abs/2504.16693)—tries to recover physical variables or validate them through resimulation and unseen actions. In parallel, a substantially larger image-formation literature has moved beyond view-dependent spherical harmonics toward explicit BRDFs, lighting decomposition, secondary rays, refraction, global illumination, participating media, nonlinear cameras, and non-RGB sensors. Representative anchors include [R3DG](https://arxiv.org/abs/2311.16043), [GS-IR](https://arxiv.org/abs/2311.16473), [GaussianShader](https://arxiv.org/abs/2311.17977), [3DGUT](https://arxiv.org/abs/2412.12507), and [TransparentGS](https://arxiv.org/abs/2504.18768).

Across these branches, the main limitation is not a lack of visually impressive results. It is a lack of joint evidence. Most papers evaluate appearance, geometry, physical rollout, parameter recovery, task success, or efficiency—but rarely all metrics required by their implied use case. Causal evidence is also sparse: a constitutive equation or BRDF inside an optimizer is stronger than a smoothness prior, yet neither alone proves that recovered variables are correct. Evidence becomes substantially stronger when a method recovers known parameters, predicts measured unseen interventions, or improves a closed-loop task. Hardware reporting is a separate weakness. Papers frequently report renderer FPS while omitting simulation rate, fitting time, VRAM, sensor requirements, or sensor-to-action latency. This review therefore treats novelty and hardware constraints as independent axes and ends with a research agenda centered on identifiability, contact and topology, optomechanical coupling, joint benchmarks, uncertainty, and complete cost accounting.

## 1. Scope, Review Questions, and Evidence Policy

### 1.1 Strict Gaussian-centric inclusion

A work is in scope when Gaussian splatting is structurally necessary to the method rather than a replaceable visualization at the end. Eligible roles include:

- Gaussians as the simulated mechanical state or as particles explicitly bound to a solver state;
- Gaussian rendering as the differentiable observation operator used to infer forces, material parameters, contact, or dynamics;
- Gaussians as the learned state on which action-conditioned prediction or control operates;
- Gaussians as primitives for explicit optical transport, inverse rendering, camera formation, or radiative sensing.

Generic dynamic 3DGS is not counted as physics merely because it has a deformation field, motion prior, smooth trajectory, or local-rigidity loss. Those mechanisms can regularize observed motion without representing mass, stress, constitutive response, forces, contact, or an image-formation law. Conversely, a physics-aware paper does not have to be open access. A novel ACM, IEEE, Elsevier, or other publisher record remains eligible even if only primary metadata or an abstract is available. Access status constrains the strength of claims we can make, not whether novelty exists.

### 1.2 Questions

This review asks:

1. **Simulation:** Which mechanical state and laws are attached to Gaussians, and what has to be supplied manually?
2. **Identification:** Which latent variables are actually inferred from observations, and how are identifiability and resimulation validated?
3. **Prediction and control:** Which methods predict an unseen action, force, or contact, and which close the loop on a task?
4. **Image formation:** Which methods model reflectance, illumination, visibility, transmission, scattering, emission, camera optics, or non-RGB sensing explicitly?
5. **Evaluation:** Which metric families are reported, and where do appearance metrics substitute for stronger physical evidence?
6. **Hardware:** What compute, memory, fitting time, update rate, rendering rate, and capture setup are actually reported?
7. **Research gap:** Which intersections remain both scientifically meaningful and realistically testable?

### 1.3 Evidence, access, and causal maturity

The companion matrix distinguishes three evidence tiers: **E3**, where the paper plus supplement or code was inspected; **E2**, where the full paper was inspected; and **E1**, where only primary abstract/metadata was recoverable. E1 records remain in the novelty census but should not support detailed numerical comparisons. Preprint and proceedings versions are merged into one work family. Publisher pages and DOI records are used to establish venue status; lawful open preprints are preferred for full-text claims. No paywall is bypassed.

Causal maturity is coded separately:

| Level | Evidence type | What it supports | What it does not yet support |
|---|---|---|---|
| **C1 — physically inspired** | A plausibility prior, smoothness/rigidity constraint, or physics-themed generative cue | Motion or appearance may be more plausible | Explicit state, parameter correctness, or causal prediction |
| **C2 — explicit law/state** | A constitutive law, contact model, BRDF, transport equation, or sensor model is optimized or simulated | The pipeline enforces a stated model | The chosen model or recovered variables are necessarily true |
| **C3 — identified variables** | Parameters, forces, lighting, optical materials, or other latent physical variables are quantitatively recovered | Some inverse problem is evaluated | Generalization to a new intervention or closed-loop use |
| **C4 — unseen intervention** | A held-out force, action, viewpoint-dependent transport path, or resimulation is predicted and measured | Evidence of causal transfer beyond fitting observations | Reliable task-level deployment |
| **C5 — closed loop** | The inferred/predicted state improves planning or control | Task-relevant causal utility | General safety, calibration, or broad-domain validity |

The scale measures the strongest demonstrated evidence, not paper quality. Physical image formation can legitimately reach C3 or C4 without mechanical intervention, for example by recovering held-out relighting or secondary-ray effects. Likewise, a forward simulator may be technically sophisticated but remain C2 if its physical parameters are chosen by the author and only rendered plausibility is shown.

The access-neutral policy matters in practice. Open preprints often coexist with paywalled venue records—for example [GASP](https://arxiv.org/abs/2409.05819) in CVIU, [Deferred Reflection](https://arxiv.org/abs/2404.18454) at SIGGRAPH, and [PRTGS](https://arxiv.org/abs/2408.03538) at ACM Multimedia—so the open copy supports detailed coding while the publisher record establishes venue status. Publisher-only or limited-access discoveries such as [FluidGS](https://dl.acm.org/doi/10.1145/3746027.3755500) and [Dynamic Global Illumination](https://doi.org/10.1109/TVCG.2026.3689199) remain at E1 when full details cannot be lawfully recovered. Conversely, paywalled ACM records with an author manuscript, including the [thermal inverse-radiative-transport work](https://doi.org/10.1145/3757377.3763938) and [differentiable area-light shading](https://doi.org/10.1145/3757377.3763865), can support E2 claims. The review never treats lack of access as lack of novelty or silently fills missing hardware fields.

## 2. A Unifying Taxonomy

The field becomes coherent when classified by **inverse problem and Gaussian role**, not by title keywords.

| Branch | Central question | Typical latent or simulated state | Gaussian role | Strongest appropriate evidence |
|---|---|---|---|---|
| **Forward mechanics** | What happens under a chosen law, parameter set, force, and boundary condition? | Position, velocity, deformation gradient, stress, density, constraints, contact | Material point, surface carrier, or renderable shell coupled to particles/mesh | Conservation/contact behavior, trajectory or deformation error, solver rate |
| **Inverse identification / digital twin** | Which parameters, forces, or boundary conditions explain observations? | Elastic moduli, density, damping, friction, restitution, mass/inertia, force, geometry | Differentiable observation model and sometimes the solver state | Parameter error, resimulation under held-out excitation, uncertainty |
| **Learned physics / control** | Can future physical state or task outcome be predicted without running the full solver? | Learned latent dynamics, force field, action-conditioned Gaussian or graph state | Persistent learned state, control particles, or differentiable rollout | Long-horizon rollout, unseen-action error, closed-loop success, calibration |
| **Physics-based image formation** | Which scene and sensor variables explain measured radiance? | Normals, BRDF, albedo, roughness, illumination, visibility, refraction, scattering, emission, camera response | Surface/radiance primitive, ray-intersection proxy, or volumetric transport primitive | Novel illumination/view, material/geometry error, transport-specific tests, runtime |

Two terms require particular care. **Mechanical material** normally means constitutive properties such as elasticity, plasticity, viscosity, density, or damping. **Optical material** means reflectance, roughness, metallicity, index of refraction, absorption, or scattering. A method that recovers a BRDF is performing a physically based inverse problem, but it has not identified stiffness. The review keeps both under “physics awareness” while never merging their evidence.

![Stacked publication timeline and corpus counts for forward mechanics, inverse identification, learned physics/control, and physics-based image formation.](quartz/content/3dgs%201/assets/physics-aware-3dgs-detailed/01-status-quo-timeline.png)

**Figure 1 — Status quo.** The strict corpus contains 57 physics-based image-formation works, 23 identification/digital-twin works, 20 forward-mechanics works, and 15 learned-physics/control works. Counts use first-public date and merge preprint/venue versions; 2026 is partial through 20 July. The image-formation total includes 24 E1 records, so the plot measures novelty breadth rather than equal depth of evidence. It is not a claim that every paper published worldwide has been enumerated.

![Heatmap of demonstrated capabilities by eight stable analysis families.](quartz/content/3dgs%201/assets/physics-aware-3dgs-detailed/02-capability-coverage.png)

**Figure 2 — Capability coverage.** “Explicit physical state,” “parameter identification,” “contact,” “unseen intervention,” “closed loop,” and “optical transport” are coded from demonstrated method/evaluation evidence. The heatmap makes a central fact visible: no single home discipline covers the whole stack. Forward solvers encode state and sometimes contact but rarely infer it; inverse-rendering systems recover optical quantities but rarely mechanics; learned prediction sometimes supports interventions but often weakens physical interpretability.

## 3. Forward Mechanical Simulation: From Renderable Kernels to Solver State

### 3.1 The dual-use particle idea

[PhysGaussian](https://arxiv.org/abs/2311.12198) is the pivotal work because it makes an explicit representational claim: a Gaussian can remain the visible primitive while its center participates in Material Point Method (MPM) updates, with deformation affecting the rendered covariance. This yields a compact “what you see is what you simulate” interface for elastic and plastic solids, granular material, fluids, fracture, and collision. It also reveals the fundamental mismatch. Photometric reconstruction places splats where they explain images—typically near visible surfaces—whereas continuum mechanics requires interior quadrature, connectivity through a grid or mesh, material volume, collision geometry, boundary conditions, and constitutive state. PhysGaussian therefore relies on user-selected simulation regions, particle filling or assumptions about the represented volume, and chosen constitutive parameters. Its contribution is a forward mechanics/rendering bridge, not automatic material discovery.

Later work explores alternative couplings rather than one universally superior solver. [Gaussian Splashing](https://arxiv.org/abs/2401.15318) couples Gaussian assets to fluid dynamics; [Spring-Gaus](https://arxiv.org/abs/2403.09434) uses a spring representation; [PBDyG](https://arxiv.org/abs/2412.04433) adopts position-based dynamics for cloth; [SimEndoGS](https://arxiv.org/abs/2405.00956) combines Gaussian tissue reconstruction with MPM and a Neo-Hookean model; [VR-GS](https://arxiv.org/abs/2401.16663) and [VR-Doh](https://arxiv.org/abs/2412.00814) emphasize interactive manipulation; and [Physically Embodied Gaussian Splatting](https://arxiv.org/abs/2406.10788) targets fast embodied response. These choices encode different tradeoffs:

- MPM naturally handles large deformation, heterogeneous materials, fracture, and grid-mediated contact, but simulation cost and parameterization can dominate rendering.
- Spring–mass models are simple and differentiable, but stiffness depends on discretization and may not map cleanly to continuum parameters.
- Position-based dynamics is robust and interactive, yet its constraint parameters are often chosen for behavior rather than identified as physical material properties.
- A Gaussian shell bound to a mesh or separate particle system may preserve rendering quality, but then the Gaussian set is no longer the complete physical state.

The literature should therefore be compared by what is **simulated**, not just by whether the paper uses the word “physics.” Relevant distinctions include rigid versus articulated versus continuum state; solid versus cloth versus fluid versus granular material; elastic versus plastic or viscoelastic response; and whether contact, fracture, topology change, or multi-material coupling is demonstrated.

### 3.2 Interactive forward systems and the latency boundary

Several systems report interactive rendering or simulation, but their numbers describe different stages. PhysGaussian reports roughly 25–36 simulation frames per second on an RTX 3090 in its tested settings. A physically embodied system reports a 30 Hz loop with about 5 ms attributed to physics on an RTX 3090. VR-GS reports a 73.8 FPS example on an RTX 4090 and Quest Pro after about 30 minutes of Gaussian preparation plus 2.5 hours of scene preparation. Gaussian Splashing reports approximately 1.6–8.1 seconds per simulation step in the cited workloads even though rendering components can take only hundredths of a second. These are all legitimate results; they are not comparable “FPS” measurements.

For an interactive digital twin, the accounting boundary should be:

`capture → calibration/pose → segmentation or object extraction → Gaussian update → physical-state update → contact solve → rendering/observation → policy/control`

Reporting only the last stage systematically overstates deployability. A renderer can exceed headset refresh rate while asset preparation takes hours or the solver updates below 1 Hz. The field would benefit from stage-separated latency traces rather than one headline FPS.

### 3.3 Forward plausibility is not identification

Forward systems often demonstrate multiple visually convincing material classes, but the result is conditional on the authors’ chosen parameters, forcing, filled volume, and boundary conditions. This evidence answers “can the representation and solver render this class of response?” It does not answer “does this observed object have the recovered Young’s modulus, friction coefficient, or density?” That distinction matters for robotics and science, where two parameter sets can create visually similar trajectories under one weak excitation and diverge under another.

The strongest forward evaluations should therefore report more than a preference study or video metric. Depending on the domain, useful evidence includes center-of-mass trajectory, pointwise deformation, surface and volume error, energy/momentum behavior, penetration and contact timing, free-oscillation frequency and damping, fluid height/velocity, and robustness across discretization. When the asset begins from photographs, geometry error and occluded-shape uncertainty must also be separated from solver error.

## 4. System Identification and Digital Twins

### 4.1 The inverse problem

System identification changes the direction of inference. Observations $I_{1:T}$, calibrated cameras, and possibly actions or applied forces supervise an unknown physical state and parameter vector. A schematic objective is

$$
\min_{\theta,\,x_0,\,u_{1:T}} \sum_t \mathcal{L}_{\text{image}}\!\left(R(G(x_t),\phi), I_t\right)
+ \lambda_g\mathcal{L}_{\text{geometry}}
+ \lambda_p\mathcal{L}_{\text{physics}},
\quad x_{t+1}=S(x_t,u_t;\theta),
$$

where `G` binds solver state to Gaussians, `R` renders them, `S` is the simulator, θ contains physical parameters, and φ contains optical/camera variables. The equation clarifies why identification is difficult: camera, geometry, force, boundary, material, and appearance errors can compensate for one another in image space.

[PAC-NeRF](https://arxiv.org/abs/2303.05512), although not a Gaussian-splatting method, is the essential conceptual root because it shows how differentiable rendering and continuum simulation can identify parameters from multi-view video. Gaussian-centric successors improve explicitness and speed but inherit the same observability problem.

### 4.2 Video priors and perceptual excitation

[PhysDreamer](https://arxiv.org/abs/2404.13026) uses video-generation priors to create or supervise plausible object dynamics from static captured scenes. [PhysFlow](https://arxiv.org/abs/2411.14423) uses motion/flow information to optimize physical behavior. These approaches address a real data bottleneck: most internet or casually captured videos do not provide synchronized forces, dense trajectories, or ground-truth material parameters. Yet a generative or flow prior can resolve ambiguity by choosing a plausible motion rather than identifying the unique real material. Its output may be ideal for animation while remaining insufficient for measurement.

This is not a criticism of using priors; it is a requirement to label evidence correctly. A method driven by learned video statistics should distinguish:

- agreement with generated or estimated motion;
- agreement with measured held-out trajectories;
- recovery of known simulator parameters;
- prediction under a new force or boundary condition;
- calibration of uncertainty when parameters are not identifiable.

### 4.3 Object-centric deformable twins

[PhysTwin](https://arxiv.org/abs/2503.17973) is a key step because it combines shape completion, Gaussian appearance, a spring–mass model, contact, and differentiable fitting, then tests resimulation and unseen interactions on ropes, cloth, and soft objects. It directly confronts the incompleteness of visible splats by using a generative shape prior. The cost is a controlled capture regime—three RGB-D viewpoints in the reported setup—and per-object optimization. The result is stronger causal evidence than a fitted animation but not yet an online general-purpose twin.

Other inverse methods expand the parameter families and domains: [D-REX](https://arxiv.org/abs/2603.01151) reports reconstruction and identification stages on an RTX 4090, with roughly 30–35 minutes for reconstruction, 5–20 minutes for identification, and about 0.5 seconds for inference in the reported configurations; [GaussTwin](https://arxiv.org/abs/2603.05108) reports a roughly 25 Hz / 40 ms total loop on an RTX 4090; [DiffWind](https://arxiv.org/abs/2603.09668) identifies wind-related dynamics but reports around two hours per scene and 12.3–21 GB in its configurations; and domain-specific work targets granular media, landslides, tissue, fluids, or articulated bodies. The important comparison is not simply runtime. It is which parameters are free, which are fixed, how excitation is observed, and whether a held-out intervention disambiguates the solution.

### 4.4 Rigid contact and closed-loop evidence

[PIN-WM](https://arxiv.org/abs/2504.16693) is especially important in the causal taxonomy because it identifies rigid-body quantities such as mass/inertia, restitution, and friction and evaluates pushing or flipping policies. Task outcome supplies evidence that purely image-based scores cannot: if a wrong parameter estimate produces the wrong contact impulse, the controller will fail even when the rendered object looks plausible. This makes PIN-WM a model for evaluation design, although its rigid, controlled setting does not directly solve deformable or topology-changing identification.

### 4.5 Identifiability is the unresolved center

A small photometric residual does not imply a correct parameter vector. Stiffness can trade against density; damping can trade against unobserved forcing; friction and restitution can be weakly excited; light and reflectance can compensate; missing geometry can be absorbed into material parameters. A credible identification paper should therefore characterize at least one of:

- a sensitivity or Fisher-information analysis;
- parameter-recovery curves across view count, noise, occlusion, and excitation;
- multiple initializations or posterior uncertainty;
- held-out actions designed to excite ambiguous modes;
- cross-simulator or real-to-sim validation;
- task degradation as parameters are perturbed.

Without such evidence, “inverse physics” may mean finding one plausible explanation, not recovering the physical truth.

## 5. Learned Physics, Prediction, and Control

Learned dynamics methods occupy a spectrum. At one end, a network predicts Gaussian motion with physical regularizers; at the other, it learns a force-conditioned surrogate of a solver or a graph-based transition model used in model-predictive control. The representation is attractive because persistent Gaussians provide dense visual tracks and can be clustered into control particles or object graphs.

The promise is computational. A learned transition can amortize repeated simulation, infer unobserved state, and provide gradients for planning. Recent directions include [graph-based Gaussian dynamics](https://arxiv.org/abs/2410.18912), [Neural Gaussian Force Fields (NGFF)](https://arxiv.org/abs/2602.00148), [SoMA](https://arxiv.org/abs/2602.02402), [MonoPhysics](https://arxiv.org/abs/2605.30320), and related action-conditioned world models. But training hardware underscores the gap between fast rollout and cheap learning: NGFF reports training on eight 80 GB H100 GPUs for roughly 48 hours, while SoMA reports about 24 hours on four H200 GPUs and around 12 rendering/prediction FPS on one H200 in its cited setting. Such work may still be novel and valuable; it simply belongs in a different deployment class from an RTX 3090 interactive solver.

Three evaluation errors recur in this branch:

1. **Short-horizon visual accuracy masquerades as stable dynamics.** Rollout error can compound after the evaluation window.
2. **In-distribution motion masquerades as intervention prediction.** A model trained on one action family may interpolate trajectories without learning the relevant causal variable.
3. **Fast network inference masquerades as end-to-end speed.** Per-scene reconstruction, feature extraction, graph construction, or action optimization may dominate.

The minimum strong evaluation combines image/geometry error with state rollout, held-out forces or actions, stability over longer horizons, and either task success or parameter/state calibration. Conservation should be tested when the architecture claims it; otherwise the claim should be framed as predictive rather than physically faithful.

![Distribution of causal-evidence levels across the four review branches.](quartz/content/3dgs%201/assets/physics-aware-3dgs-detailed/03-causal-evidence.png)

**Figure 3 — Causal evidence.** A C2 paper can contain a rigorous solver or transport law, while C4–C5 require evidence beyond reconstructing the input observations. The distribution is expected to be asymmetric: optical inverse rendering often validates held-out illumination, whereas mechanical closed-loop studies require actuated capture and are much rarer.

## 6. Physics-Based Image Formation

### 6.1 Why the default 3DGS appearance model is insufficient

Original 3DGS stores view-dependent color through spherical harmonics and composites projected ellipses. This is highly effective for view synthesis, but it entangles illumination, material, visibility, and view dependence. The representation can reproduce training images without recovering a stable surface, normal, BRDF, environment map, refractive interface, or sensor response. That is a problem whenever the scene is relit, viewed through a different camera, contains mirrors or glass, or serves as the observation model for mechanics.

Physics-based image-formation work decomposes the problem into some subset of:

- geometry and surface normals;
- diffuse albedo and specular/roughness parameters;
- direct illumination and shadows;
- secondary reflection/refraction rays;
- indirect/global illumination;
- absorption, scattering, emission, or spectral response;
- camera projection, aperture, motion integration, rolling shutter, or sensor modality.

The largest risk is again non-identifiability. A highlight can be explained by roughness, illumination, normal orientation, or geometry. A dark underwater region can be absorption, occlusion, low albedo, or camera response. Consequently, relighting PSNR alone does not establish correct decomposition.

### 6.2 Inverse rendering and relighting

Early Gaussian inverse-rendering systems—[R3DG](https://arxiv.org/abs/2311.16043), [GS-IR](https://arxiv.org/abs/2311.16473), [GaussianShader](https://arxiv.org/abs/2311.17977), and [GIR](https://arxiv.org/abs/2312.05133)—replace or augment view-dependent color with normals, BRDF-like attributes, lighting, visibility, and shading. They demonstrate that the explicit splat representation can support fast relighting while retaining high-quality novel views. They also make different geometric commitments: some derive normals from Gaussian covariance or depth, while others introduce surface or shading regularization. Comparing only PSNR can obscure whether a method recovers physically stable normals and materials.

Later work strengthens surface structure or transport. [DeferredGS](https://arxiv.org/abs/2404.09412) adopts deferred shading to separate geometry/material attributes from illumination. [Phys3DGS](https://arxiv.org/abs/2409.10335) reports a staged pipeline on an RTX 4090—up to roughly one hour for its first stage, ten minutes for a second stage, and about 30 minutes for PBR fitting—with approximately 42.75 FPS at 64 samples per pixel in its reported setting. [GS³](https://arxiv.org/abs/2410.11419) ([publisher DOI](https://doi.org/10.1145/3680528.3687576)) reports about 40–70 minutes of training on a dual-EPYC, 768 GB RAM, RTX 4090 system and over 90 FPS in its tested rendering setup. [GlossyGS](https://arxiv.org/abs/2410.13349) explicitly targets glossy surfaces and reports around one hour on a 32 GB V100 and roughly 30 FPS, contrasting with a cited prior requiring four A100 GPUs for days. [GeoSplatting](https://arxiv.org/abs/2410.24204) improves surface-oriented reconstruction but exposes resolution cost: roughly 15–20 minutes on a 24 GB RTX 4090 at a 96³ setting, while a 128³ configuration exceeds one hour and 60 GB.

These numbers matter, but they do not rank novelty. They show that physically decomposed appearance can range from a lightweight extension to an expensive multi-stage optimization, and that scene resolution or samples per pixel can change the deployment class.

### 6.3 Secondary rays, mirrors, and glossy reflection

Rasterized alpha compositing is efficient because it follows camera rays through a sorted set of projected splats. Mirror and glossy transport requires rays that leave the first surface and query the scene again. [3D Gaussian Ray Tracing](https://arxiv.org/abs/2407.07090) and [3DGUT](https://arxiv.org/abs/2412.12507) move Gaussian representations toward ray tracing and support secondary rays or more general camera models. 3DGUT reports approximately 3.77–4.98 ms in cited configurations, corresponding to about 200–277 FPS, but the exact workload and quality settings must accompany those rates. [3DGS Deferred Reflection](https://arxiv.org/abs/2404.18454) ([ACM DOI](https://doi.org/10.1145/3641519.3657456)) and later reflection-oriented methods model mirror or glossy effects without folding everything into spherical harmonics.

The research frontier is not simply “add a reflected color.” Stronger systems must address reflected-scene geometry, visibility, multiple bounces, rough reflection footprints, energy behavior, and editing consistency. An evaluation should isolate mirror regions, changing illumination, and off-training reflected viewpoints rather than average all pixels together.

### 6.4 Refraction and transparent objects

Transparent geometry is a particularly sharp test because the observed color usually originates behind the reconstructed surface. [TransparentGS](https://arxiv.org/abs/2504.18768) ([ACM TOG DOI](https://doi.org/10.1145/3730892)) explicitly models transparent-object geometry and refractive behavior and reports about one hour of fitting with roughly 31–51 FPS in the cited configurations. The matrix also retains abstract-level novelty for [TranSplat](https://arxiv.org/abs/2503.22676), [RTR-GS](https://arxiv.org/abs/2507.07733) ([ACM DOI](https://doi.org/10.1145/3746027.3755197)), [RefracGS](https://arxiv.org/abs/2603.21695), and [RT-Splatting](https://arxiv.org/abs/2605.18263). These works explore transmissive primitives, radiance transfer, ray tracing, or more robust reconstruction, but E1 entries are not used for detailed performance claims.

The main ambiguity is between interface geometry and index/transport. A refracted background can be matched by a wrong surface with compensated appearance. Transparent-object evaluation should therefore include independent geometry, novel backgrounds or illumination, multi-view refraction paths, and—where possible—known indices of refraction. Ordinary novel-view PSNR is not enough.

### 6.5 Global illumination and radiative transfer

[PRTGS](https://arxiv.org/abs/2408.03538) ([ACM DOI](https://doi.org/10.1145/3664647.3680893)), [GI-GS](https://arxiv.org/abs/2410.02619), [IRGS](https://arxiv.org/abs/2412.15867), and a 2026 [dynamic global-illumination study](https://doi.org/10.1109/TVCG.2026.3689199) extend Gaussians beyond direct relighting. Publisher evidence also contributes an [inverse thermal-radiative-transport method](https://doi.org/10.1145/3757377.3763938) and [differentiable area-light shading](https://doi.org/10.1145/3757377.3763865). These papers are important because shadows, interreflection, area-light effects, and emission/reflection separation otherwise leak into albedo or view-dependent coefficients.

The technical tension is accuracy versus interactivity. Precomputed radiance transfer can make relighting fast but assumes restricted transport or fixed geometry. Ray/path tracing is general but may need many samples or acceleration structures. Learned residual transport is efficient but can fail outside its training illumination. Strong comparisons should therefore report not just rendered FPS, but preprocessing/training, samples per pixel, memory, scene update cost, and behavior under light or geometry edits.

### 6.6 Participating media, underwater imaging, and emission

[WaterSplatting](https://arxiv.org/abs/2408.08206), [SeaSplat](https://arxiv.org/abs/2409.17345), and [DehazeGS](https://arxiv.org/abs/2501.03659) model some combination of attenuation, backscatter, color formation, and medium effects. [Subsurface Scattering for 3DGS](https://arxiv.org/abs/2408.12282) extends the question to internal light transport, while [SpectralGaussians](https://arxiv.org/abs/2408.06975), [Thermal3D-GS](https://arxiv.org/abs/2409.08042), and radiative X-ray/tomographic methods broaden the measurement model beyond ordinary RGB reflection.

These works should not all be grouped under generic “enhancement.” A physics-based underwater method should expose an image-formation equation and recover or condition on medium parameters; a purely learned color-restoration network with Gaussians is not equivalent. Metrics should include medium-parameter or color-consistency evidence where available, geometry under scattering, and downstream reconstruction—not only no-reference image quality.

### 6.7 Cameras and non-RGB sensors

[3DGUT](https://arxiv.org/abs/2412.12507) is a major camera-model result because it removes the requirement that Gaussian rasterization use a simple pinhole projection, enabling nonlinear cameras and secondary rays. [Cinematic Gaussians](https://arxiv.org/abs/2406.07329), [Gaussian Splatting on the Move](https://arxiv.org/abs/2403.13327), and [Every Camera Effect](https://arxiv.org/abs/2509.10759) address aperture, blur, motion integration, or broader camera effects. Mirror reconstruction methods—including [Mirror-3DGS](https://arxiv.org/abs/2404.01168), [MirrorGaussian](https://arxiv.org/abs/2405.11921), and [RefGaussian](https://arxiv.org/abs/2406.05852)—treat the environment as part of the observation geometry.

The sensor branch extends further: [Thermal3D-GS](https://arxiv.org/abs/2409.08042) handles thermal data; [X-Gaussian](https://arxiv.org/abs/2403.04116) and [R²-Gaussian](https://arxiv.org/abs/2405.20693) target X-ray/radiative formation; and event-camera work includes [E2GS](https://arxiv.org/abs/2406.14978), [Ev-GS](https://arxiv.org/abs/2407.11343), [EventSplat](https://arxiv.org/abs/2412.07293), and [SweepEvGS](https://arxiv.org/abs/2412.11579). These E1 records remain conservative novelty entries until full-text details are inspected.

These methods demonstrate that “rendering” is really a differentiable measurement operator. Their evaluation must be modality-specific: event polarity/timing, projection/integration error for radiative sensors, thermal consistency, or SAR formation. RGB PSNR cannot serve as a universal metric.

## 7. Metric Landscape: What the Field Measures and What It Misses

![Heatmap showing metric-family reporting across mechanics, identification, learned dynamics/control, inverse rendering, camera/sensor formation, and media/radiative imaging.](quartz/content/3dgs%201/assets/physics-aware-3dgs-detailed/04-metric-landscape.png)

**Figure 4 — Metric landscape.** The heatmap marks whether a paper reports at least one metric in a family; it does not compare magnitudes across incompatible datasets. The concentration along disciplinary columns is itself the result: inverse rendering measures appearance and sometimes material/light; simulation measures physical behavior; identification measures parameters; systems papers measure efficiency; control papers measure tasks. Few studies span the full chain their claims imply.

| Metric family | Typical measures | What it can establish | Common failure of interpretation |
|---|---|---|---|
| **Appearance** | PSNR, SSIM, LPIPS, relighting image error, perceptual preference | Fidelity to held-out images | High scores can hide wrong geometry, BRDF, force, or material parameters |
| **Geometry** | depth error, Chamfer distance, F-score, normal angular error, surface completeness | Reconstructed surface quality | Visible-surface accuracy does not establish interior volume or collision geometry |
| **Material or light** | albedo/roughness error, normal error, environment-map error, relighting accuracy | Quality of optical decomposition | Parameters can compensate; synthetic ground truth may not transfer to real scenes |
| **Physical behavior** | trajectory/deformation error, contact timing, penetration, frequency/damping, conservation, rollout error | Whether simulated/predicted state behaves correctly | A single observed motion may not identify the cause |
| **Parameter error** | Young’s modulus, Poisson ratio, density, viscosity, damping, friction, restitution, mass/inertia, force error | Direct inverse-physics accuracy | Units, parameter ranges, and equivalent parameterizations can differ across solvers |
| **Efficiency** | fitting/training time, simulation/update rate, renderer FPS, end-to-end latency, GPU memory, model size | Resource and deployment cost | Renderer FPS is often mistaken for complete system throughput |
| **Task outcome** | manipulation/pushing success, planning reward, control error, completion rate | Utility under intervention | Success on a narrow task does not prove generally correct dynamics |

### Recommended minimum evaluation bundles

**Forward simulation:** appearance/geometry of the initial asset; state trajectory or deformation; contact/conservation diagnostics appropriate to the solver; simulation step rate and VRAM; renderer rate reported separately.

**System identification:** parameter error on controlled synthetic or instrumented data; held-out resimulation on a different excitation; geometry/camera sensitivity; uncertainty or multi-initialization; fitting time and memory.

**Learned prediction/control:** short- and long-horizon state error; held-out actions/materials; rollout stability; inference plus planning latency; closed-loop task outcome; uncertainty calibration where safety matters.

**Physical image formation:** ordinary novel-view fidelity; geometry/normals; material/light or modality-specific parameter recovery; held-out illumination/background/camera effects; preprocessing, sample count, memory, and rendering rate.

## 8. Hardware Constraints as an Independent Evidence Axis

Novelty and feasibility answer different questions. A method trained for two days on H100s can be scientifically novel; a 200 FPS renderer can still require hours of per-scene fitting; a paper that omits VRAM is not necessarily slow. The correct policy is to transcribe the reported setup and mark missing fields as **not reported/recovered**, without inferring “consumer-ready,” “real time,” or “efficient” from the GPU name alone.

![Heatmap of hardware-field reporting by review branch.](quartz/content/3dgs%201/assets/physics-aware-3dgs-detailed/05-hardware-reporting.png)

**Figure 5 — Recovered hardware-field coverage.** The denominator is E2/E3 full-text evidence, not the entire novelty census. GPU, VRAM, training/fitting time, simulation/update rate, rendering rate, end-to-end latency, memory/model size, and sensor setup are counted independently. A rendering FPS entry does not fill simulation/update or end-to-end latency; sparse end-to-end and memory reporting makes deployment claims especially difficult to reproduce.

### Representative reported setups

| Work | Reported setup/cost | Appropriate interpretation |
|---|---|---|
| [PhysGaussian](https://arxiv.org/abs/2311.12198) | RTX 3090 / Intel i9; roughly 25–36 simulation FPS in reported cases | Demonstrates interactive forward examples, not automatic reconstruction or identification |
| [Gaussian Splashing](https://arxiv.org/abs/2401.15318) | Roughly 1.6–8.1 s per simulation step; rendering components roughly 0.015–0.116 s | Solver dominates; rendering FPS alone would be misleading |
| [VR-GS](https://arxiv.org/abs/2401.16663) | RTX 4090 + Quest Pro; about 30 min Gaussian and 2.5 h scene preparation; 73.8 FPS example | Interactive playback follows nontrivial offline preparation |
| [Spring-Gaus](https://arxiv.org/abs/2403.09434) | RTX 3090; about 10 min reported preparation/fitting | Moderate per-scene cost; exact workload matters |
| [SimEndoGS](https://arxiv.org/abs/2405.00956) | Xeon W-2223 + RTX 3090; about 20–25 FPS | Domain-specific capture and reconstruction assumptions remain part of feasibility |
| [PhysTwin](https://arxiv.org/abs/2503.17973) | Controlled three-view RGB-D capture plus per-object optimization | Sensor and setup complexity are as important as GPU |
| [D-REX](https://arxiv.org/abs/2603.01151) | RTX 4090; reconstruction ~30–35 min, identification ~5–20 min, inference ~0.5 s | Separate stages reveal the true pipeline cost |
| [GaussTwin](https://arxiv.org/abs/2603.05108) | RTX 4090; about 25 Hz / 40 ms total in reported configuration | One of the more complete latency claims; reproduce the stage boundary |
| [DiffWind](https://arxiv.org/abs/2603.09668) | RTX 4090 24 GB; ~2 h/scene; 12.3–21 GB | Memory and fitting cost constrain scale even if final rendering is fast |
| [NGFF](https://arxiv.org/abs/2602.00148) | 8× H100 80 GB; ~48 h training | Strong learned-physics scale, not a one-GPU training baseline |
| [SoMA](https://arxiv.org/abs/2602.02402) | 4× H200 for ~24 h; one H200 around 12 FPS | Training and inference deployment classes differ |
| [DeferredGS](https://arxiv.org/abs/2404.09412) | RTX 3090 24 GB; ~3–4 h; ~30 FPS at 800² | Physically decomposed rendering retains significant per-scene fitting cost |
| [GaussianShader](https://arxiv.org/abs/2311.17977) | ~0.58 h; ~97 FPS in reported setting | Fast relighting result with a comparatively light optimization report |
| [GIR](https://arxiv.org/abs/2312.05133) | V100 result ~92.58 FPS for ~350k Gaussians; UE5/RTX 3090 result ~78.75 FPS | Engine, Gaussian count, and quality settings must stay attached to FPS |
| [Progressive Radiance Distillation](https://arxiv.org/abs/2408.07595) | i7-13700KF, 32 GB RAM, RTX 4090; ~39+8 min; ~221 FPS | Explicit preprocessing stages precede high-rate rendering |
| [Phys3DGS](https://arxiv.org/abs/2409.10335) | RTX 4090; stages ≤1 h, ≤10 min, ~30 min; ~42.75 FPS at 64 spp | Samples per pixel and stage costs qualify the headline rate |
| [GS³](https://arxiv.org/abs/2410.11419) | dual EPYC, 768 GB RAM, RTX 4090; ~40–70 min; >90 FPS | Host memory/setup should not be collapsed into “single GPU” |
| [GeoSplatting](https://arxiv.org/abs/2410.24204) | RTX 4090 24 GB: 96³ ~15–20 min; 128³ >1 h and >60 GB | Resolution changes both time and memory class |
| [3DGUT](https://arxiv.org/abs/2412.12507) | ~3.77–4.98 ms / ~200–277 FPS in cited configurations | A renderer result; full pipeline and scene settings remain necessary |
| [TransparentGS](https://arxiv.org/abs/2504.18768) | ~1 h; ~31–51 FPS | Refraction-aware fitting remains offline despite interactive rendering |

One dedicated hardware reference can be used as a side box rather than physics-core evidence: **IRIS**, an SoC paper ([IEEE JSSC DOI 10.1109/JSSC.2025.3613707](https://doi.org/10.1109/JSSC.2025.3613707)), reports a 28 nm, 20.25 mm² design and large speed/energy/bandwidth gains over its stated edge-GPU baseline. It demonstrates that Gaussian rendering hardware is evolving, but it should not be used to claim that mechanics, inverse fitting, or transport solvers inherit the same gains.

### Hardware reporting checklist

Every system paper should ideally report:

1. CPU, GPU/accelerator count and model, VRAM and host RAM;
2. input resolution, frame count, Gaussian/particle count, solver grid, ray/sample count;
3. reconstruction/preprocessing, physical fitting, and learned-model training times separately;
4. physical simulation/update rate, renderer rate, and sensor-to-output latency separately;
5. peak memory during fitting and inference, plus stored/streamed model size;
6. sensor count/type, calibration, synchronization, and action/force instrumentation;
7. quality–speed settings and whether timings include data transfer, collision, or policy optimization.

## 9. Cross-Cutting Gaps

### 9.1 The visual state is not the physical state

Splats concentrate on observed appearance, while mechanics needs interiors, connectivity, collision surfaces, and sometimes topology. Particle filling, generative shape completion, mesh binding, and hybrid Gaussian–solver state solve different pieces but introduce priors and failure modes. A central research question is how uncertainty in occluded geometry should propagate into physical prediction.

### 9.2 Contact and topology remain weakly validated

Many demonstrations include collision, grasping, cutting, fluid interaction, or fracture, yet quantitative contact timing, impulse, penetration, and topology consistency are rarely benchmarked. Contact is where small geometry and parameter errors become discontinuous outcome errors. It is also where smooth differentiable simulators are least faithful.

### 9.3 Mechanics and optics are usually separated

Mechanical identification often assumes that the renderer’s appearance model is good enough. Optical inverse rendering often assumes a static scene. In reality, deformation changes normals, self-shadow, specularity, transmission paths, and sensor exposure; optical error can bias the inferred force or stiffness. A joint optomechanical twin should estimate enough image formation to prevent these compensations while remaining computationally tractable.

### 9.4 Uncertainty is almost absent

Most pipelines output one geometry, one material vector, one light, and one rollout. Partial views and weak excitation generally admit multiple explanations. Downstream planning needs a posterior or calibrated ensemble, not a single attractive video. Uncertainty should cover geometry, association, parameters, actions/forces, and model mismatch.

### 9.5 Evaluation datasets reward home disciplines

Relighting datasets rarely contain measured mechanics; physics datasets rarely contain controlled complex illumination; robotics datasets rarely provide ground-truth material/contact parameters; event, thermal, X-ray, and SAR datasets use different formation models. This fragmentation explains the metric heatmap and prevents strong cross-branch claims.

### 9.6 Runtime accounting stops at convenient boundaries

The field needs stage-separated profiles and a standard vocabulary: training, per-scene fitting, physical identification, one simulation step, Gaussian update, render, planning, and total sensor-to-action latency. Model size, peak VRAM, and streaming bandwidth should be included where state persists over time.

## 10. Open Questions and Debates

### What qualifies as physics awareness?

An explicit law is a defensible minimum for C2 evidence, but physically inspired regularization can still be useful at C1. The important practice is to state which variables and laws are represented and avoid equating plausible motion or view dependence with identified physics.

### Should Gaussians be the solver state or the interface?

Dual-use particles offer direct rendering and simple correspondence. A hybrid mesh/MPM/graph state may be more stable and complete. The right answer depends on whether rendering fidelity, continuum accuracy, topology, or online estimation is primary. Research should compare state coupling and failure modes instead of assuming end-to-end Gaussian purity is always desirable.

### Is a recovered parameter meaningful if it is solver-specific?

Spring constants, PBD compliance, MPM constitutive parameters, and real-world material measurements may not correspond one-to-one. Papers should report units, parameterization, discretization dependence, and transfer tests. A parameter can be useful for prediction without being a laboratory material measurement, but the claim must match.

### Is perceptual realism enough?

For animation and telepresence, preference and image quality may be the correct objective. For surgery, robotics, or scientific reconstruction, held-out intervention, parameter, geometry, and uncertainty evidence are necessary. The review should not impose one standard on every use case; it should expose which standard each paper meets.

### Differentiable solver or learned surrogate?

Differentiable solvers embed stronger structure but can be expensive, brittle around contact, and biased by model choice. Learned surrogates are fast and can absorb residual dynamics but may fail out of distribution. Hybrid residual models and uncertainty-aware switching are promising, provided evaluation separates interpolation from unseen causal transfer.

### Can physical optics fix mechanical identification—or make it less identifiable?

A better renderer can reduce systematic residuals from shadows, gloss, blur, or refraction. It also introduces more latent variables that can trade against geometry and motion. Joint estimation therefore needs staged calibration, priors, or experimental design—not merely a larger end-to-end optimizer.

## 11. Ranked Research Agenda

![Scatter plot of research directions by gap confidence and student feasibility.](quartz/content/3dgs%201/assets/physics-aware-3dgs-detailed/06-research-agenda.png)

**Figure 6 — Research-direction frontier.** Scores are a qualitative synthesis for a 6–12 month project, not a performance ranking. The best near-term directions create stronger evidence and reusable evaluation rather than requiring a new solver, renderer, sensor rig, and controller simultaneously.

| Rank | Direction | Why the gap is credible | Minimum credible contribution and evaluation |
|---|---|---|---|
| **1** | **Joint visual–physical benchmark** | Existing datasets and metrics are siloed by branch | Provide controlled scenes with geometry, illumination, force/action, parameters, contacts, and held-out interventions; report appearance, geometry, physical, efficiency, and task metrics with stage accounting |
| **2** | **Uncertainty-aware system identification** | Most inverse pipelines output one parameter vector despite ambiguity | Add posterior/ensemble estimation to a strong Gaussian inverse-physics baseline; test coverage and parameter/rollout error across view count, noise, occlusion, and excitation |
| **3** | **Contact and topology validation** | Collision, fracture, cutting, and tearing are visually shown more often than measured | Build a benchmark with contact timing, penetration, impulse/trajectory, and topology annotations; compare MPM, spring–mass, PBD, and hybrid bindings |
| **4** | **Physics-state compression** | Dynamic codecs budget appearance/motion but not stress, velocity, parameters, or contact | Compress solver and Gaussian state jointly; evaluate bitrate against image, rollout, conservation/contact, and task distortion rather than PSNR alone |
| **5** | **Online optomechanical digital twin** | Mechanics and physical optics remain largely separate and offline | Jointly estimate object/camera state, a limited optical model, and physical parameters online; report observability, sensor-to-rollout latency, and unseen-interaction error |
| **6** | **Calibrated secondary light transport** | Reflection/refraction/GI methods improve visually but physical decomposition remains ambiguous | Use known-material/known-light scenes, held-out transport paths, geometry checks, uncertainty, and complete samples-per-pixel/time/memory accounting |
| **7** | **Sensor-aware physics under adverse capture** | Event, thermal, underwater, X-ray, and SAR work rarely connects to dynamics | Combine one explicit adverse sensor model with system identification; show that modeling formation improves parameter or task evidence, not only reconstructed images |

For the best novelty-to-risk balance, a student project should begin with Rank 1, 2, 3, or 4. Rank 5 is the most integrative and potentially highest impact, but errors in calibration, geometry, optical decomposition, physical identification, and latency compound. Rank 6 is a good graphics-focused direction; Rank 7 is especially compelling when suitable instrumentation or a domain collaborator already exists.

## 12. Suggested Reading Order

1. **Representation and continuum root:** [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079), [PAC-NeRF](https://arxiv.org/abs/2303.05512), [PhysGaussian](https://arxiv.org/abs/2311.12198).
2. **Mechanical identification:** [PhysDreamer](https://arxiv.org/abs/2404.13026), [PhysFlow](https://arxiv.org/abs/2411.14423), [PhysTwin](https://arxiv.org/abs/2503.17973), [PIN-WM](https://arxiv.org/abs/2504.16693).
3. **Inverse rendering:** [R3DG](https://arxiv.org/abs/2311.16043), [GS-IR](https://arxiv.org/abs/2311.16473), [GaussianShader](https://arxiv.org/abs/2311.17977), [DeferredGS](https://arxiv.org/abs/2404.09412).
4. **Secondary transport and cameras:** [3D Gaussian Ray Tracing](https://arxiv.org/abs/2407.07090), [3DGUT](https://arxiv.org/abs/2412.12507), [TransparentGS](https://arxiv.org/abs/2504.18768), [PRTGS](https://arxiv.org/abs/2408.03538).
5. **Media and sensors:** [WaterSplatting](https://arxiv.org/abs/2408.08206), [SpectralGaussians](https://arxiv.org/abs/2408.06975), [Thermal3D-GS](https://arxiv.org/abs/2409.08042), [X-Gaussian](https://arxiv.org/abs/2403.04116), [E2GS](https://arxiv.org/abs/2406.14978).

## 13. Sources, Reproducibility, and Boundaries

The [companion evidence matrix](quartz/content/3dgs%201/physics-aware-3dgs-simulation-identification-optics-evidence-matrix.md) is the authoritative human-readable source index. Every record includes a title, stable primary URL, first-public year, publication/access status, evidence tier, primary branch and subclass, analysis group, novelty, Gaussian role, physical model, inputs, evaluation, limitations, task-dependent causal level, metric families, and hardware fields. The [JSON dataset](quartz/content/3dgs%201/physics-aware-3dgs-simulation-identification-optics-data.json) is the machine-readable source for the matrix and figures, and the [Matplotlib generator](quartz/content/3dgs%201/generate_physics_aware_3dgs_review_assets.py) validates its schema before regenerating every artifact.

Counts in Figures 1–5 are generated from that dataset. Figure 6 is an explicitly qualitative research judgment. Any numerical performance cited in prose should be traceable to E2/E3 primary evidence; E1 publisher-only records should contribute to novelty coverage and be described conservatively. The review is a reproducible scoping review, not a claim of exhaustive systematic coverage.

## Rerun Inputs

```yaml
workflow: firecrawl-research-papers
topic: strict physics-aware 3D Gaussian Splatting
branches:
  - forward mechanical simulation
  - inverse system identification and digital twins
  - learned physics, prediction, and control
  - physics-based image formation and sensing
cutoff: 2026-07-20
access_policy: include novel open and paywalled records; do not bypass paywalls
hardware_policy: record reported constraints independently from novelty; do not infer consumer feasibility
core_count: 115
full_text_count: 90
abstract_or_metadata_count: 25
output: detailed Markdown review, evidence matrix, JSON dataset, and six Matplotlib figures
```
