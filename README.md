# Black Hole Observational Memory

## Observational Memory Horizons of Black Holes: Quantifying the Recoverability of Event Identity and Event Timing from Synthetic and Realistic Observations

<p align="center">
  <img src="figures/black_hole_simulator_preview.png" width="1000">
</p>

<p align="center">
  <em>
  Real-time GPU black-hole simulator used to generate synthetic observations for observational-memory experiments.
  </em>
</p>

## Research Pipeline

<p align="center">
  <img src="figures/overview_pipeline.png" width="1200">
</p>

<p align="center">
  <em>
  End-to-end research workflow spanning synthetic black-hole generation, observational-memory analysis, GRMHD-inspired bridging, synthetic-to-real calibration, robustness testing, and expanded external validation.
  </em>
</p>

## Overview

Can a present day observation of a black hole retain information about events that occurred in its past?

This project introduces the concept of **Observational Memory** and investigates whether observable black-hole structures preserve recoverable information about prior physical events.

Using synthetic simulations, deep learning, morphology analysis, GRMHD-inspired bridge datasets, Event Horizon Telescope (EHT) observations, and external validation, the project studies how information persists and degrades across different observational regimes.



## Main Research Question

> Can black-hole observations retain recoverable information about past events, and what aspects of that information remain observable over time?

The project investigates:

* What happened?
* When did it happen?
* How long does the information remain observable?



## Key Concepts

### Observational Memory

Information about past physical events encoded within a current observation.

### Event Identity

Information describing **what happened**.

Examples:

* Accretion burst
* Jet eruption
* Turbulence spike
* Spin transition

### Event Timing

Information describing **when the event occurred**.

### Memory Persistence

The duration over which event information remains recoverable.

### Memory Horizon

The maximum timescale over which information remains observable.



## Main Results

### Event Identity Remains Recoverable

Across multiple experiments, event identity remained partially recoverable from observations.

### Event Timing Is Much Harder

Event timing consistently proved more difficult to recover than event identity.

### Observation Timing Matters More Than Model Complexity

Event-centered observations substantially improved recoverability, while late-stage observations provided limited additional information.

### Synthetic-to-Real Consistency Exists

Real EHT observations consistently aligned with a GRMHD-inspired synthetic bridge domain.

### Expanded External Validation Passed

Phase 8-A achieved:

```text
PASS
Bridge Consistency Score = 1.0
```

with all major validation layers identifying the same bridge domain.

---

## Research Pipeline

```text
Synthetic Black-Hole Universe
            ↓
Memory Persistence Experiments
            ↓
Temporal Observation Studies
            ↓
Event-Centered Observation
            ↓
GRMHD-Inspired Bridge Domain
            ↓
Real EHT Observations
            ↓
Synthetic-to-Real Calibration
            ↓
Robustness Validation
            ↓
Expanded External Validation
```



## Project Timeline

| Phase       | Description                      |
| ----------- | -------------------------------- |
| Phase 1     | Synthetic Black-Hole Universe    |
| Phase 2     | Observation Reconstruction       |
| Phase 2.2   | Memory-Preserving Reconstruction |
| Phase 4     | Physical Parameter Recovery      |
| Phase 5     | Physics-to-Image Coupling        |
| Phase 6     | Static Memory Persistence        |
| Phase 6-T   | Late Temporal Observation        |
| Phase 6-U   | Event-Centered Observation       |
| Phase 7-A   | Real Observation Consistency     |
| Phase 7-A.2 | GRMHD Image Harvesting           |
| Phase 7-B   | Synthetic-to-Real Calibration    |
| Phase 7-C   | Calibration Robustness           |
| Phase 7-D   | Analogue Memory Mapping          |
| Phase 7-D.1 | Metadata Recovery                |
| Phase 8-A   | Expanded Real & GRMHD Validation |
| Final       | Project Synthesis                |



## Final Scientific Finding

The strongest supported conclusion of the project is:

> Real EHT observations consistently occupy regions of morphology and latent space most closely associated with a GRMHD-inspired synthetic bridge domain. This relationship survives calibration, robustness testing, analogue mapping, metadata reconstruction, and expanded external validation.



## What This Project Does NOT Claim

The project does **not** claim:

* Recovery of true black-hole histories
* Recovery of real accretion histories
* Reliable real event-age reconstruction
* Resolution of the black-hole information paradox

Instead, the project provides a framework for:

* Observational memory analysis
* Event recoverability measurement
* Memory persistence estimation
* Synthetic-to-real analogue mapping



## Installation

```bash
git clone https://github.com/Malbasahi/Black-Hole-Observational-Memory.git

cd Black-Hole-Observational-Memory

pip install -r requirements.txt
```



## Repository Structure

```text
README.md
requirements.txt

docs/
figures/
notebooks/
src/

Final_Project_Synthesis.ipynb
```



## Citation

Marwa Albasahi

**Observational Memory Horizons of Black Holes: Quantifying the Recoverability of Event Identity and Event Timing from Synthetic and Realistic Observations**

