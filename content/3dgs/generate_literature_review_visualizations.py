#!/usr/bin/env python3
"""Generate evidence-linked visualizations for the physics-aware dynamic 3DGS review.

The corpus and metric figures are parsed from the companion evidence matrix. The
gap figure is an explicitly qualitative synthesis of claims in the review, and
the agenda figure uses the scores in the review's ranked research agenda.
"""

from __future__ import annotations

import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "physics-aware-dynamic-3dgs-evidence-matrix.md"
OUT = ROOT / "assets" / "physics-aware-dynamic-3dgs"

NAVY = "#183153"
BLUE = "#3274A1"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
ORANGE = "#F4A261"
RED = "#D65A4A"
LIGHT = "#EEF3F7"
GRID = "#D6DEE5"
TEXT = "#263238"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT,
            "axes.edgecolor": GRID,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
        }
    )


def wrap(label: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_matrix() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    section = ""
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        if not re.match(r"^\| P\d{3} \|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        records.append(
            {
                "id": cells[0],
                "paper": cells[1],
                "tags": cells[2],
                "setting": cells[3],
                "evaluation": cells[4],
                "code": cells[5],
                "boundary": cells[6],
                "section": section,
            }
        )
    if len(records) != 78:
        raise ValueError(f"Expected 78 evidence records, parsed {len(records)}")
    return records


def corpus_status(records: list[dict[str, str]]) -> None:
    section_names = {
        "Foundational Roots": "Foundational roots",
        "Dynamic and Deformable Gaussian Representations": "Dynamic / deformable",
        "Gaussian SLAM and Dynamic Mapping": "SLAM / mapping",
        "Physics-Aware Gaussians": "Physics-aware",
        "Compression and Deployment Efficiency": "Compression / deployment",
        "Robot, Contact, and Action-Conditioned Interaction": "Robot / action",
        "Human Semantic Querying and Editing": "Human query / editing",
    }
    section_counts = Counter(section_names[r["section"]] for r in records)

    tag_patterns = {
        "Dynamic": r"\bDyn\b",
        "Deformable": r"\bDef\b",
        "SLAM": r"\bSLAM\b",
        "Physics": r"\bPhys\b",
        "Compression": r"\bComp\b",
        "Robot interaction": r"\bR-Int\b",
        "Human interaction": r"\bH-Int\b",
    }
    tag_counts = {
        label: sum(bool(re.search(pattern, r["tags"])) for r in records)
        for label, pattern in tag_patterns.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"wspace": 0.45})
    fig.suptitle("Status quo: a broad corpus with uneven capability coverage", y=1.02, fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.965,
        "78 core works; primary sections are mutually exclusive, capability tags can overlap",
        ha="center",
        color="#52616B",
    )

    left_labels = list(section_counts)
    left_values = [section_counts[x] for x in left_labels]
    order = np.argsort(left_values)
    axes[0].barh(np.array(left_labels)[order], np.array(left_values)[order], color=BLUE)
    axes[0].set_title("Primary strand", loc="left")
    axes[0].set_xlabel("Number of works")
    axes[0].grid(axis="x")
    axes[0].set_axisbelow(True)
    for y, value in enumerate(np.array(left_values)[order]):
        axes[0].text(value + 0.35, y, str(value), va="center", fontweight="bold")
    axes[0].set_xlim(0, max(left_values) + 4)

    right_labels = list(tag_counts)
    right_values = [tag_counts[x] for x in right_labels]
    order = np.argsort(right_values)
    colors = [TEAL if value >= 20 else ORANGE if value >= 10 else RED for value in np.array(right_values)[order]]
    axes[1].barh(np.array(right_labels)[order], np.array(right_values)[order], color=colors)
    axes[1].set_title("Cross-cutting capability tag", loc="left")
    axes[1].set_xlabel("Tagged works (overlap allowed)")
    axes[1].grid(axis="x")
    axes[1].set_axisbelow(True)
    for y, value in enumerate(np.array(right_values)[order]):
        axes[1].text(value + 0.35, y, str(value), va="center", fontweight="bold")
    axes[1].set_xlim(0, max(right_values) + 5)

    fig.text(
        0.01,
        -0.015,
        "Interpretation: dynamic/deformable reconstruction dominates; physics and interaction are smaller, overlapping subsets.",
        fontsize=9,
        color="#52616B",
    )
    save(fig, "01-status-quo-corpus.png")


def metric_landscape(records: list[dict[str, str]]) -> None:
    sections = [
        ("Dynamic and Deformable Gaussian Representations", "Dynamic /\ndeformable"),
        ("Gaussian SLAM and Dynamic Mapping", "SLAM /\nmapping"),
        ("Physics-Aware Gaussians", "Physics-aware"),
        ("Compression and Deployment Efficiency", "Compression /\ndeployment"),
        ("Robot, Contact, and Action-Conditioned Interaction", "Robot /\naction"),
        ("Human Semantic Querying and Editing", "Human query /\nediting"),
    ]
    metric_patterns = {
        "Appearance\nPSNR · SSIM · LPIPS": r"PSNR|SSIM|LPIPS|DSSIM|render(?:ing)? quality|image quality|video (?:error|comparison)",
        "Geometry\ndepth · Chamfer · IoU": r"Chamfer|F-score|depth|geometry|mesh|IoU|surface|reconstruction accuracy",
        "State estimation\nATE · tracking · pose": r"\bATE\b|\bRPE\b|tracking|trajectory|pose|joint-axis|part-motion|parameter (?:error|recovery)",
        "Physical behavior\nrollout · contact · motion": r"physics-consistency|physical plausibility|deformation|lattice|motion realism|perceptual motion|future-video|rollout|simulation|unseen interaction|parameter (?:error|recovery)",
        "Efficiency\nFPS · memory · bitrate": r"FPS|speed|runtime|time|latency|memory|size|bitrate|MB/frame|throughput|efficiency|update cost|GPU|decode|training",
        "Task / human outcome\nsuccess · preference": r"task success|policy success|grasp success|user study|user preference|preference|reasoning metrics",
    }

    heat = np.zeros((len(sections), len(metric_patterns)))
    raw = np.zeros_like(heat, dtype=int)
    totals = []
    for i, (section, _) in enumerate(sections):
        rows = [r for r in records if r["section"] == section]
        totals.append(len(rows))
        for j, pattern in enumerate(metric_patterns.values()):
            raw[i, j] = sum(bool(re.search(pattern, r["evaluation"], re.I)) for r in rows)
            heat[i, j] = 100 * raw[i, j] / len(rows)

    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    image = ax.imshow(heat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    fig.suptitle("Evaluation is siloed: metric coverage by research strand", y=0.985, fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.938,
        "Cell = share of papers in that strand whose evidence field reports the metric family",
        ha="center",
        color="#52616B",
    )
    ax.set_xticks(range(len(metric_patterns)), labels=list(metric_patterns), fontsize=9)
    ax.set_yticks(range(len(sections)), labels=[label for _, label in sections])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    fig.subplots_adjust(top=0.82, bottom=0.11)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            color = "white" if heat[i, j] >= 55 else TEXT
            ax.text(j, i, f"{heat[i, j]:.0f}%\n({raw[i, j]}/{totals[i]})", ha="center", va="center", color=color, fontweight="bold")

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("Share of strand (%)")
    ax.set_xticks(np.arange(-0.5, len(metric_patterns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sections), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(
        0.01,
        0.01,
        "Keyword-coded from the matrix's Evaluation evidence column; families indicate reporting presence, not result quality.",
        fontsize=9,
        color="#52616B",
    )
    save(fig, "02-metric-landscape.png")


def capability_gaps() -> None:
    strands = [
        "Dynamic / deformable\nrendering",
        "Gaussian SLAM /\nmapping",
        "Physics-aware\nGaussians",
        "Dynamic\ncompression",
        "Robot / action\nmodels",
    ]
    capabilities = [
        "Online camera + map estimation",
        "Persistent object identity",
        "General deformable state",
        "Material parameter identification",
        "Explicit contact / collision",
        "Rate / memory accounting",
        "Unseen-intervention prediction",
        "Closed-loop task evaluation",
        "Calibrated uncertainty",
    ]
    # 0 missing, 1 rare, 2 emerging, 3 established. This rubric summarizes the
    # review's strand-level claims; it is not a paper-count statistic.
    scores = np.array(
        [
            [0, 3, 3, 0, 0, 2, 1, 0, 0],
            [3, 2, 1, 0, 0, 2, 1, 0, 0],
            [0, 2, 3, 2, 2, 0, 2, 1, 0],
            [0, 2, 2, 0, 0, 3, 0, 0, 0],
            [1, 3, 2, 1, 2, 1, 2, 3, 1],
        ]
    )
    labels = np.array(["Missing", "Rare", "Emerging", "Established"])
    cmap = ListedColormap(["#F2F2F2", "#E76F51", "#E9C46A", "#2A9D8F"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.imshow(scores, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title("The central gap is integration across capabilities", pad=22)
    ax.text(
        0.5,
        1.025,
        "Qualitative strand-level synthesis: maturity of support in the reviewed literature",
        transform=ax.transAxes,
        ha="center",
        color="#52616B",
    )
    ax.set_xticks(range(len(capabilities)), labels=[wrap(x, 17) for x in capabilities], rotation=34, ha="right")
    ax.set_yticks(range(len(strands)), labels=strands)
    ax.tick_params(length=0)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            ax.text(j, i, labels[scores[i, j]], ha="center", va="center", fontsize=8.5, fontweight="bold", color="white" if scores[i, j] in (1, 3) else TEXT)
    ax.set_xticks(np.arange(-0.5, len(capabilities), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(strands), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.subplots_adjust(bottom=0.31, top=0.89)
    fig.legend(
        handles=[Patch(facecolor=cmap(i), label=labels[i]) for i in range(4)],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.01,
        0.012,
        "Rubric: established = routine capability; emerging = multiple concrete systems; rare = isolated/limited evidence; missing = not demonstrated at strand level.",
        fontsize=9,
        color="#52616B",
    )
    save(fig, "03-capability-gap-map.png")


def research_agenda() -> None:
    directions = [
        (1, "Physics-state-aware\ndynamic compression", 5, 4),
        (2, "Persistent dynamic-object\nPhoto-SLAM", 5, 4),
        (3, "Uncertainty-aware\nmaterial ID", 5, 3),
        (4, "Topology / lifecycle-aware\ndeformable GS", 4, 3),
        (5, "Action-conditioned\nobject-centric GS-SLAM", 5, 2),
        (6, "Physics-aware\nendoscopic twin", 4, 3),
        (7, "Unified cross-strand\nbenchmark", 5, 4),
    ]
    offsets = {1: (-0.39, 0.24), 2: (-0.39, 0.00), 7: (-0.39, -0.24), 3: (-0.39, 0.14), 4: (0.09, 0.14), 6: (0.09, -0.15), 5: (-0.39, 0.10)}
    # Several agenda items have identical integer scores. Apply a small visual
    # jitter so every rank remains visible while preserving its score cell.
    jitter = {1: (-0.045, 0.045), 2: (0.0, 0.0), 7: (0.045, -0.045), 4: (-0.025, 0.035), 6: (0.025, -0.035)}

    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    ax.axhspan(3.5, 5.25, color="#E8F5F1", zorder=0)
    ax.axvspan(4.5, 5.25, color="#EDF4FA", zorder=0)
    plot_x = [d[2] + jitter.get(d[0], (0, 0))[0] for d in directions]
    plot_y = [d[3] + jitter.get(d[0], (0, 0))[1] for d in directions]
    ax.scatter(plot_x, plot_y, s=360, c=[TEAL, TEAL, GOLD, ORANGE, RED, ORANGE, TEAL], edgecolor="white", linewidth=2, zorder=3)
    for rank, label, confidence, feasibility in directions:
        jx, jy = jitter.get(rank, (0, 0))
        ax.text(confidence + jx, feasibility + jy, str(rank), ha="center", va="center", color="white" if rank != 3 else TEXT, fontweight="bold", fontsize=12, zorder=4)
        dx, dy = offsets[rank]
        ax.annotate(label, (confidence, feasibility), xytext=(confidence + dx, feasibility + dy), textcoords="data", fontsize=9, ha="left", va="center")

    ax.set_xlim(3.55, 5.25)
    ax.set_ylim(1.6, 5.15)
    ax.set_xticks([4, 5])
    ax.set_yticks([2, 3, 4, 5])
    ax.set_xlabel("Gap confidence (1–5)", fontweight="bold")
    ax.set_ylabel("Student feasibility (1–5)", fontweight="bold")
    ax.set_title("Research directions: prioritize the high-confidence, feasible frontier", pad=18)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.text(4.93, 5.02, "Best starting region", ha="center", va="top", color=TEAL, fontweight="bold")
    fig.text(
        0.01,
        0.01,
        "Numbers match the ranked agenda. Identical score pairs are slightly jittered for legibility. Assumption: one consumer GPU, 6–12 months.",
        fontsize=9,
        color="#52616B",
    )
    save(fig, "04-research-agenda.png")


def main() -> None:
    set_style()
    records = parse_matrix()
    corpus_status(records)
    metric_landscape(records)
    capability_gaps()
    research_agenda()
    print(f"Generated 4 figures in {OUT}")


if __name__ == "__main__":
    main()
