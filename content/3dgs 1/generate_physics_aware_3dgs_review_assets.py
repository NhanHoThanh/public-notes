#!/usr/bin/env python3
"""Validate the detailed physics-aware 3DGS corpus and generate its matrix/figures."""

from __future__ import annotations

import json
import textwrap
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "physics-aware-3dgs-simulation-identification-optics-data.json"
MATRIX = ROOT / "physics-aware-3dgs-simulation-identification-optics-evidence-matrix.md"
OUT = ROOT / "assets" / "physics-aware-3dgs-detailed"

TEXT = "#263238"
BLUE = "#3274A1"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
ORANGE = "#F4A261"
RED = "#D65A4A"
PURPLE = "#7B61A8"
GRID = "#D6DEE5"
PALETTE = [BLUE, TEAL, GOLD, ORANGE, RED, PURPLE, "#4E8098", "#8AB17D"]

BRANCHES = ["mechanics", "identification", "learned_physics", "physical_optics"]
CAUSAL_LABELS = {
    1: "C1 physically inspired",
    2: "C2 explicit law/state",
    3: "C3 identified variables",
    4: "C4 unseen intervention",
    5: "C5 closed-loop evidence",
}
METRIC_FAMILIES = [
    "appearance",
    "geometry",
    "material_or_light",
    "physical_behavior",
    "parameter_error",
    "efficiency",
    "task_outcome",
]
HARDWARE_FIELDS = [
    "gpu",
    "vram",
    "training_or_fitting_time",
    "simulation_or_update_rate",
    "rendering_rate",
    "end_to_end_latency",
    "memory_or_size",
    "sensor_setup",
]
CAPABILITIES = [
    "explicit_state",
    "parameter_id",
    "contact",
    "unseen_intervention",
    "closed_loop",
    "optical_transport",
]
ANALYSIS_GROUPS = [
    "continuum_particle_mechanics",
    "deformable_articulated_mechanics",
    "fluid_granular_mechanics",
    "identification_digital_twins",
    "learned_dynamics_control",
    "inverse_rendering_relighting",
    "reflection_transmission_transport",
    "sensor_media_radiative_imaging",
]
ANALYSIS_GROUP_LABELS = {
    "continuum_particle_mechanics": "continuum / particle mechanics",
    "deformable_articulated_mechanics": "deformable / articulated mechanics",
    "fluid_granular_mechanics": "fluid / granular mechanics",
    "identification_digital_twins": "identification / digital twins",
    "learned_dynamics_control": "learned dynamics / control",
    "inverse_rendering_relighting": "inverse rendering / relighting",
    "reflection_transmission_transport": "reflection / transmission transport",
    "sensor_media_radiative_imaging": "sensor / media / radiative imaging",
}
GROUP_BRANCHES = {
    "continuum_particle_mechanics": "mechanics",
    "deformable_articulated_mechanics": "mechanics",
    "fluid_granular_mechanics": "mechanics",
    "identification_digital_twins": "identification",
    "learned_dynamics_control": "learned_physics",
    "inverse_rendering_relighting": "physical_optics",
    "reflection_transmission_transport": "physical_optics",
    "sensor_media_radiative_imaging": "physical_optics",
}


def style() -> None:
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
        }
    )


def wrapped(value: str, width: int) -> str:
    return "\n".join(textwrap.wrap(value.replace("_", " "), width=width, break_long_words=False))


def load_and_validate() -> tuple[dict, list[dict]]:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    papers = payload["papers"]
    if not papers:
        raise ValueError("Corpus is empty")
    cutoff = date.fromisoformat(payload["cutoff"])
    required = {
        "id",
        "title",
        "url",
        "year",
        "branch",
        "subclass",
        "analysis_group",
        "publication",
        "access",
        "evidence",
        "novelty",
        "gaussian_role",
        "physical_model",
        "inputs",
        "evaluation",
        "hardware",
        "limitations",
        "causal_level",
        "metric_families",
        "hardware_fields",
        "capabilities",
    }
    ids: set[str] = set()
    urls: set[str] = set()
    titles: set[str] = set()
    for index, paper in enumerate(papers):
        missing = required - paper.keys()
        if missing:
            raise ValueError(f"Record {index} missing keys: {sorted(missing)}")
        if paper["id"] in ids:
            raise ValueError(f"Duplicate id: {paper['id']}")
        ids.add(paper["id"])
        normalized_title = " ".join(paper["title"].casefold().split())
        if normalized_title in titles:
            raise ValueError(f"Duplicate normalized title: {paper['title']}")
        titles.add(normalized_title)
        if paper["url"] in urls:
            raise ValueError(f"Duplicate URL: {paper['url']}")
        urls.add(paper["url"])
        if paper["branch"] not in BRANCHES:
            raise ValueError(f"Unknown branch in {paper['id']}: {paper['branch']}")
        if paper["analysis_group"] not in ANALYSIS_GROUPS:
            raise ValueError(f"Unknown analysis group in {paper['id']}: {paper['analysis_group']}")
        if GROUP_BRANCHES[paper["analysis_group"]] != paper["branch"]:
            raise ValueError(f"Analysis group / branch mismatch in {paper['id']}")
        if not isinstance(paper["year"], int) or isinstance(paper["year"], bool) or paper["year"] > cutoff.year:
            raise ValueError(f"Invalid first-public year in {paper['id']}: {paper['year']}")
        if (
            not isinstance(paper["causal_level"], int)
            or isinstance(paper["causal_level"], bool)
            or paper["causal_level"] not in CAUSAL_LABELS
        ):
            raise ValueError(f"Invalid causal level in {paper['id']}")
        list_fields = ["metric_families", "hardware_fields", "capabilities"]
        if any(not isinstance(paper[field], list) for field in list_fields):
            raise ValueError(f"Coded fields must be lists in {paper['id']}")
        if any(any(not isinstance(item, str) for item in paper[field]) for field in list_fields):
            raise ValueError(f"Coded fields must contain strings in {paper['id']}")
        if any(len(paper[field]) != len(set(paper[field])) for field in list_fields):
            raise ValueError(f"Duplicate coded value in {paper['id']}")
        bad_metrics = set(paper["metric_families"]) - set(METRIC_FAMILIES)
        bad_hardware = set(paper["hardware_fields"]) - set(HARDWARE_FIELDS)
        bad_capabilities = set(paper["capabilities"]) - set(CAPABILITIES)
        if bad_metrics or bad_hardware or bad_capabilities:
            raise ValueError(
                f"Invalid coded fields in {paper['id']}: "
                f"{bad_metrics | bad_hardware | bad_capabilities}"
            )
        if not str(paper["url"]).startswith("https://"):
            raise ValueError(f"Non-HTTPS source in {paper['id']}")
        string_fields = required - {"year", "causal_level", "metric_families", "hardware_fields", "capabilities"}
        if any(not isinstance(paper[field], str) or not paper[field].strip() for field in string_fields):
            raise ValueError(f"Empty or non-string field in {paper['id']}")
        if not paper["evidence"].startswith(("E1 ", "E2 ", "E3 ")):
            raise ValueError(f"Invalid evidence tier in {paper['id']}: {paper['evidence']}")
    missing_branches = set(BRANCHES) - {paper["branch"] for paper in papers}
    if missing_branches:
        raise ValueError(f"Corpus is missing branches: {sorted(missing_branches)}")

    agenda = payload["research_agenda"]
    if sorted(item["rank"] for item in agenda) != list(range(1, len(agenda) + 1)):
        raise ValueError("Research-agenda ranks must be contiguous from 1")
    for item in agenda:
        if not (1 <= item["gap_confidence"] <= 5 and 1 <= item["student_feasibility"] <= 5):
            raise ValueError(f"Invalid agenda score: {item}")
        if not item["evidence_ids"] or not set(item["evidence_ids"]).issubset(ids):
            raise ValueError(f"Invalid agenda evidence IDs: {item}")
    return payload, papers


def esc(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def write_matrix(payload: dict, papers: list[dict]) -> None:
    branch_names = {
        "mechanics": "Forward Mechanical Simulation",
        "identification": "Inverse Identification and Digital Twins",
        "learned_physics": "Learned Physics, Prediction, and Control",
        "physical_optics": "Physics-Based Image Formation",
    }
    lines = [
        "# Physics-Aware 3DGS — Detailed Evidence Matrix",
        "",
        f"> **Frozen {payload['cutoff']}.** {len(papers)} Gaussian-centric technical works. Companion to [the detailed literature review](physics-aware-3dgs-simulation-identification-optics-literature-review.md).",
        "",
        "## Evidence and Coding Rules",
        "",
        "- **E3**: full paper plus supplement or code inspected; **E2**: full paper inspected; **E1**: primary abstract/metadata only. E1 records remain in the novelty census but are not used for detailed performance claims.",
        "- Access and publication status are independent. A paywalled venue version can coexist with a lawful open preprint.",
        "- Hardware is transcribed only when located in inspected primary evidence. **NR** means not reported or not recovered; it is not a zero-cost result.",
        "- The task-dependent evidence ladder runs from **C1** (physically inspired) to **C5** (closed-loop task evidence). C5 is not applicable to every optics problem, and the code is not a paper-quality ranking.",
        "",
    ]
    for branch in BRANCHES:
        subset = [p for p in papers if p["branch"] == branch]
        lines.extend(
            [
                f"## {branch_names[branch]} ({len(subset)})",
                "",
                "| ID | Work, publication, and access | Novelty and physical model | Inputs and evaluation | Reported hardware | Boundary |",
                "|---|---|---|---|---|---|",
            ]
        )
        for p in sorted(subset, key=lambda x: (x["year"], x["id"])):
            work = f"[{p['title']}]({p['url']}) ({p['year']}); {p['publication']}; {p['access']}; {p['evidence']}"
            novelty = f"**{ANALYSIS_GROUP_LABELS[p['analysis_group']]} → {p['subclass']} · {CAUSAL_LABELS[p['causal_level']]}** — {p['novelty']} Model/state: {p['physical_model']} Gaussian role: {p['gaussian_role']}"
            evidence = f"Inputs: {p['inputs']} Evaluation: {p['evaluation']} Codes — metrics: {', '.join(p['metric_families']) or 'none recovered'}; capabilities: {', '.join(p['capabilities']) or 'none recovered'}."
            hardware = f"{p['hardware']} Fields: {', '.join(p['hardware_fields']) or 'none recovered'}."
            lines.append(
                "| " + " | ".join(esc(x) for x in [p["id"], work, novelty, evidence, hardware, p["limitations"]]) + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Rerun Inputs",
            "",
            "```yaml",
            "workflow: firecrawl-research-papers",
            f"cutoff: {payload['cutoff']}",
            f"scope: {payload['scope']}",
            f"core_count: {len(papers)}",
            "access_policy: access-neutral inclusion; no paywall bypass",
            "hardware_policy: transcribe reported setup; do not infer consumer feasibility",
            "query_families:",
            *[f"  - {query}" for query in payload["search_method"]["query_families"]],
            f"deduplication: {payload['search_method']['deduplication']}",
            f"inclusion: {payload['search_method']['inclusion']}",
            f"exclusion: {payload['search_method']['exclusion']}",
            "```",
            "",
        ]
    )
    MATRIX.write_text("\n".join(lines), encoding="utf-8")


def save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def detailed_records(papers: list[dict]) -> list[dict]:
    """Return records whose full text supports detailed coding."""
    return [paper for paper in papers if not paper["evidence"].startswith("E1")]


def figure_timeline(papers: list[dict]) -> None:
    years = sorted({p["year"] for p in papers})
    counts = {branch: [sum(p["year"] == year and p["branch"] == branch for p in papers) for year in years] for branch in BRANCHES}
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.3), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.38})
    fig.suptitle("Included physics-aware 3DGS corpus by first-public year and branch", y=1.01, fontsize=18, fontweight="bold")
    bottom = np.zeros(len(years))
    branch_labels = ["Forward mechanics", "Identification / twins", "Learned physics / control", "Physics-based image formation"]
    for color, branch, label in zip(PALETTE, BRANCHES, branch_labels):
        axes[0].bar(years, counts[branch], bottom=bottom, color=color, label=label)
        bottom += np.array(counts[branch])
    if 2026 in years:
        partial_total = sum(counts[branch][years.index(2026)] for branch in BRANCHES)
        axes[0].bar(2026, partial_total, facecolor="none", edgecolor=TEXT, hatch="///", linewidth=0.8)
    axes[0].set_title("First-public year", loc="left")
    axes[0].set_ylabel("Included works")
    axes[0].set_xticks(years, labels=[f"{year}*" if year == 2026 else str(year) for year in years])
    axes[0].grid(axis="y")
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    bc = Counter(p["branch"] for p in papers)
    labels = ["Image formation", "Learned / control", "Identification", "Mechanics"]
    values = [bc[b] for b in BRANCHES][::-1]
    axes[1].barh(labels, values, color=PALETTE[:4][::-1])
    axes[1].set_title("Corpus by primary contribution", loc="left")
    axes[1].set_xlabel("Works")
    axes[1].grid(axis="x")
    axes[1].set_axisbelow(True)
    for y, value in enumerate(values):
        axes[1].text(value + 0.35, y, str(value), va="center", fontweight="bold")
    axes[1].set_xlim(0, max(values) + 7)
    fig.text(0.01, -0.01, "Counts use first-public date; preprint and venue versions are merged. *2026 is partial through 20 July.", color="#52616B", fontsize=9)
    save(fig, "01-status-quo-timeline.png")


def figure_taxonomy(papers: list[dict]) -> None:
    papers = detailed_records(papers)
    subclasses = [group for group in ANALYSIS_GROUPS if any(p["analysis_group"] == group for p in papers)]
    capabilities = CAPABILITIES
    display = ["Explicit physical\nstate / variables", "Parameter\nidentification", "Contact /\ncollision", "Held-out\nintervention", "Closed-loop\ntask", "Physical light /\nsensor transport"]
    matrix = np.zeros((len(subclasses), len(capabilities)), dtype=int)
    totals = []
    for i, subclass in enumerate(subclasses):
        rows = [p for p in papers if p["analysis_group"] == subclass]
        totals.append(len(rows))
        for j, cap in enumerate(capabilities):
            matrix[i, j] = sum(cap in p["capabilities"] for p in rows)
    pct = np.array([[100 * matrix[i, j] / totals[i] for j in range(len(capabilities))] for i in range(len(subclasses))])
    height = max(7.5, 0.48 * len(subclasses) + 2.5)
    fig, ax = plt.subplots(figsize=(14.8, height))
    im = ax.imshow(pct, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_title("Capability coverage reveals where physics remains descriptive", pad=24)
    ax.set_xticks(range(len(capabilities)), labels=display)
    ax.set_yticks(range(len(subclasses)), labels=[f"{wrapped(ANALYSIS_GROUP_LABELS[s], 29)}  (n={totals[i]})" for i, s in enumerate(subclasses)])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            ax.text(j, i, f"{pct[i,j]:.0f}%", ha="center", va="center", fontsize=8.5, color="white" if pct[i, j] > 55 else TEXT, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Share within analysis group (%)")
    fig.text(0.01, 0.005, "Full-text records only. Capabilities are coded from demonstrated method/evaluation evidence, not inferred from titles.", color="#52616B", fontsize=9)
    save(fig, "02-capability-coverage.png")


def figure_causal(papers: list[dict]) -> None:
    papers = detailed_records(papers)
    counts = np.zeros((len(BRANCHES), len(CAUSAL_LABELS)), dtype=int)
    for p in papers:
        counts[BRANCHES.index(p["branch"]), p["causal_level"] - 1] += 1
    row_totals = counts.sum(axis=1, keepdims=True)
    pct = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float), where=row_totals > 0) * 100
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    im = ax.imshow(pct, cmap="OrRd", vmin=0, vmax=max(50, pct.max()), aspect="auto")
    labels = ["Forward mechanics", "Identification / twins", "Learned physics / control", "Physics-based image formation"]
    ax.set_title("Intervention and task evidence mature unevenly", pad=22)
    ax.set_xticks(range(5), labels=[f"C{i}\n{CAUSAL_LABELS[i].split(' ', 1)[1]}" for i in range(1, 6)])
    ax.set_yticks(range(4), labels=labels)
    ax.tick_params(length=0)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{counts[i,j]}\n({pct[i,j]:.0f}%)", ha="center", va="center", color="white" if pct[i, j] > 35 else TEXT, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025, label="Share within branch (%)")
    fig.text(0.01, 0.005, "Full-text records only. C5 is task-dependent and not expected for all optics work; the scale does not rank paper quality.", color="#52616B", fontsize=9)
    save(fig, "03-causal-evidence.png")


def figure_metrics(papers: list[dict]) -> None:
    papers = detailed_records(papers)
    subclass_groups = [
        ("forward simulation", lambda p: p["branch"] == "mechanics"),
        ("inverse identification", lambda p: p["branch"] == "identification"),
        ("learned dynamics / control", lambda p: p["branch"] == "learned_physics"),
        ("inverse rendering / relighting", lambda p: p["analysis_group"] == "inverse_rendering_relighting"),
        ("reflection / transmission", lambda p: p["analysis_group"] == "reflection_transmission_transport"),
        ("sensor / media / radiative", lambda p: p["analysis_group"] == "sensor_media_radiative_imaging"),
    ]
    matrix = np.zeros((len(subclass_groups), len(METRIC_FAMILIES)), dtype=int)
    totals = []
    for i, (_, predicate) in enumerate(subclass_groups):
        rows = [p for p in papers if predicate(p)]
        totals.append(len(rows))
        for j, metric in enumerate(METRIC_FAMILIES):
            matrix[i, j] = sum(metric in p["metric_families"] for p in rows)
    pct = np.array([[100 * matrix[i, j] / totals[i] if totals[i] else 0 for j in range(len(METRIC_FAMILIES))] for i in range(len(subclass_groups))])
    fig, ax = plt.subplots(figsize=(15.5, 7))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_title("Evaluation remains organized by home discipline", pad=24)
    ax.set_xticks(range(len(METRIC_FAMILIES)), labels=[wrapped(x, 16) for x in METRIC_FAMILIES])
    ax.set_yticks(range(len(subclass_groups)), labels=[f"{wrapped(name, 25)} (n={totals[i]})" for i, (name, _) in enumerate(subclass_groups)])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            ax.text(j, i, f"{pct[i,j]:.0f}%\n{matrix[i,j]}/{totals[i]}", ha="center", va="center", color="white" if pct[i, j] > 55 else TEXT, fontweight="bold", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025, label="Reporting share (%)")
    fig.text(0.01, 0.005, "Full-text records only. Presence indicates a metric family was used; magnitudes are not compared across incompatible workloads.", color="#52616B", fontsize=9)
    save(fig, "04-metric-landscape.png")


def figure_hardware(papers: list[dict]) -> None:
    papers = detailed_records(papers)
    labels = ["Forward\nmechanics", "Identification /\ntwins", "Learned physics /\ncontrol", "Physics-based\nimage formation"]
    matrix = np.zeros((len(BRANCHES), len(HARDWARE_FIELDS)), dtype=int)
    totals = []
    for i, branch in enumerate(BRANCHES):
        rows = [p for p in papers if p["branch"] == branch]
        totals.append(len(rows))
        for j, field in enumerate(HARDWARE_FIELDS):
            matrix[i, j] = sum(field in p["hardware_fields"] for p in rows)
    pct = np.array([[100 * matrix[i, j] / totals[i] for j in range(len(HARDWARE_FIELDS))] for i in range(len(BRANCHES))])
    fig, ax = plt.subplots(figsize=(15.5, 6.3))
    im = ax.imshow(pct, cmap="Purples", vmin=0, vmax=100, aspect="auto")
    ax.set_title("Recovered hardware-field coverage is incomplete and stage-specific", pad=24)
    hardware_labels = ["GPU", "VRAM", "training or\nfitting time", "simulation or\nupdate rate", "rendering rate", "end-to-end\nlatency", "memory or size", "sensor setup"]
    ax.set_xticks(range(len(HARDWARE_FIELDS)), labels=hardware_labels)
    ax.set_yticks(range(len(BRANCHES)), labels=[f"{label} (n={totals[i]})" for i, label in enumerate(labels)])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            ax.text(j, i, f"{pct[i,j]:.0f}%\n{matrix[i,j]}/{totals[i]}", ha="center", va="center", color="white" if pct[i, j] > 52 else TEXT, fontweight="bold", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025, label="Full-text records with recovered field (%)")
    fig.text(0.01, 0.005, "Full-text recovery coverage, not compute magnitude. Renderer FPS is separate from update and end-to-end latency; see the review's exact anchor table.", color="#52616B", fontsize=9)
    save(fig, "05-hardware-reporting.png")


def figure_agenda(agenda: list[dict]) -> None:
    fig, (ax, notes) = plt.subplots(1, 2, figsize=(15.5, 8), gridspec_kw={"width_ratios": [1.05, 1.25], "wspace": 0.22})
    ax.axvspan(3.5, 5.5, color="#EDF4FA")
    ax.axhspan(3.5, 5.5, color="#E8F5F1", alpha=0.8)
    offsets = [(-0.12, 0.12), (0.12, 0.10), (-0.12, -0.12), (0.12, -0.10), (0, 0), (-0.10, 0.10), (0.10, -0.10)]
    for item, (dx, dy) in zip(agenda, offsets * ((len(agenda) // len(offsets)) + 1)):
        confidence = item["gap_confidence"] + dx
        feasibility = item["student_feasibility"] + dy
        color = TEAL if item["student_feasibility"] >= 4 else GOLD if item["student_feasibility"] == 3 else RED
        ax.scatter(confidence, feasibility, s=330, color=color, edgecolor="white", linewidth=2, zorder=3)
        ax.text(confidence, feasibility, str(item["rank"]), ha="center", va="center", fontweight="bold", color="white" if item["student_feasibility"] != 3 else TEXT, fontsize=11)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xlabel("Gap confidence (1–5)", fontweight="bold")
    ax.set_ylabel("6–12 month project feasibility (1–5)", fontweight="bold")
    ax.set_title("Evidence–feasibility map", loc="left", pad=14)
    ax.grid(True)
    ax.set_axisbelow(True)
    notes.axis("off")
    notes.set_title("Ranked interface opportunities", loc="left", pad=14)
    y = 0.96
    for item in agenda:
        notes.text(0.0, y, f"{item['rank']}. {item['label']}", transform=notes.transAxes, fontweight="bold", va="top", fontsize=10.5)
        notes.text(0.04, y - 0.04, wrapped(item["rationale"], 68), transform=notes.transAxes, va="top", fontsize=9, color="#52616B")
        y -= 0.135
    fig.suptitle("Research directions: novelty lies at the interfaces", y=0.99, fontsize=18, fontweight="bold")
    fig.text(0.01, 0.005, "Qualitative synthesis backed by evidence IDs in the structured corpus; scores are judgments, not bibliometric measurements.", color="#52616B", fontsize=9)
    save(fig, "06-research-agenda.png")


def main() -> None:
    style()
    payload, papers = load_and_validate()
    write_matrix(payload, papers)
    figure_timeline(papers)
    figure_taxonomy(papers)
    figure_causal(papers)
    figure_metrics(papers)
    figure_hardware(papers)
    figure_agenda(payload["research_agenda"])
    print(f"Validated {len(papers)} papers; wrote {MATRIX.name} and 6 figures to {OUT}")


if __name__ == "__main__":
    main()
