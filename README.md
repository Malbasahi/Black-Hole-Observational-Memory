# Observational Memory Horizons of Black Holes

## Quantifying the Recoverability of Event Identity and Event Timing from Synthetic and Realistic Observations

<p align="center">
  <img src="figures/black_hole_simulator_preview.png" width="900">
</p>

<p align="center">
  <em>Real-time GPU black-hole simulator used as the foundation for synthetic observational memory experiments.</em>
</p>


---

## Overview

This project investigates a fundamental question:

> Can observations of a black hole retain recoverable information about events that happened in its past?

Black holes are traditionally viewed as systems that conceal information behind an event horizon. However, the observable environment surrounding a black hole, its accretion disk, photon ring, jet structures, and emitted radiation, may preserve detectable signatures of past physical activity.

This project introduces the concept of **Observational Memory**, a framework for studying how much information about previous events remains encoded in black-hole observations and for how long that information can be recovered.

Using synthetic simulations, machine learning models, morphology analysis, temporal observations, and comparisons with realistic astrophysical data products, this research explores the limits of recovering historical information from black-hole imagery.

---

## Core Research Question

The project investigates two complementary questions:

### Event Identity

**What happened?**

Examples:

* Accretion burst
* Jet eruption
* Turbulence spike
* Spin transition

### Event Timing

**When did it happen?**

Can an observation determine how long ago a physical event occurred?

---

## Key Concepts

### Observational Memory

The persistence of detectable signatures from past physical events within a current observation.

### Memory Persistence

The duration over which information about a past event remains recoverable.

### Memory Horizon

The maximum observational timescale over which information can be reliably recovered.

### Event Recoverability

The ability of an analysis method to reconstruct information about past events from observational data.

---

## Scientific Motivation

Modern observatories such as the Event Horizon Telescope (EHT) provide unprecedented views of black-hole environments.

While current observations capture the present state of a system, an open question remains:

> Do these observations contain measurable traces of past activity?

Understanding observational memory could help improve:

* Interpretation of black-hole observations
* Observation scheduling strategies
* Analysis of transient astrophysical phenomena
* Synthetic-to-real data transfer methods
* Future machine-learning approaches in astrophysics

---

# Project Structure

## Phase 1–5: Synthetic Black Hole Generation

Development of a controllable black-hole simulation framework capable of generating:

* Accretion disks
* Photon rings
* Jet-like structures
* Turbulent features
* Variable physical parameters

These phases establish the synthetic environment used throughout the project.

---

## Phase 2 & 2.2: Reconstruction Models

Physics-aware U-Net models are trained to reconstruct clean observations from corrupted inputs.

Goals:

* Denoising
* Structure preservation
* Memory-preserving reconstruction

---

## Phase 4 Series: Parameter Recovery

Machine-learning models are trained to recover hidden physical parameters from synthetic observations.

Examples:

* Black-hole mass
* Spin
* Accretion rate
* Jet power
* Turbulence strength

---

## Phase 5 Series: Memory-Preserving Dataset Design

The synthetic generator is improved to increase coupling between:

* Physical parameters
* Observable morphology
* Temporal history

These phases strengthen the relationship between hidden physics and visible structure.

---

## Phase 6: Static Memory Persistence

Question:

> Can a single final image recover information about a past event?

This phase introduces event histories and measures:

* Event identity recoverability
* Event timing recoverability
* Memory persistence curves
* Memory half-life

---

## Phase 6-T: Temporal Observation

Question:

> Does observing a short sequence improve recoverability?

Temporal clips near the final observation are analyzed.

---

## Phase 6-U: Event-Centered Observation

Question:

> Does observing the event directly improve recoverability?

This phase centers observations around the event itself and studies how recoverability changes.

---

## Phase 7-A: Real Observation Consistency Study

Question:

> Do synthetic memory signatures resemble realistic black-hole observations?

This phase compares:

* Synthetic observations
* EHT-derived products
* External black-hole imagery
* GRMHD-inspired morphology references

using morphology-space and latent-space analysis.

---

## Phase 7-A.1: External Dataset Acquisition

Automated acquisition and organization of:

* Public EHT repositories
* External observation products
* FITS-based data products

---

## Phase 7-A.2: GRMHD Image Harvesting

Creation of a GRMHD-inspired morphology reference library that serves as a bridge between:

```text
Synthetic Simulations
        ↓
GRMHD-Inspired Morphologies
        ↓
Real Observations
```

---

## Final Audit & Synthesis

The project concludes with a comprehensive audit that:

* Aggregates results across all phases
* Generates publication-ready figures
* Produces thesis-style summaries
* Quantifies memory persistence metrics
* Compares observation strategies

---

# Main Findings

The project demonstrates several important results:

### 1. Event Identity Is Recoverable

Black-hole observations can retain detectable information about the type of event that occurred.

### 2. Event Timing Is Much Harder

Determining exactly when an event occurred is significantly more difficult than identifying the event itself.

### 3. Observation Timing Matters

The timing of an observation strongly influences recoverability.

### 4. Event-Centered Observations Improve Recovery

Observations made near the event dramatically improve event identification.

### 5. Information Availability Dominates Model Complexity

Providing relevant observational information is often more important than increasing model complexity.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Malbasahi/Black-Hole-Observational-Memory.git
cd Black-Hole-Observational-Memory-Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Recommended Environment

Python 3.10+

GPU recommended but not required.

Tested with:

* Ubuntu Linux
* PyTorch
* Jupyter Notebook

---

# Running the Project

Launch Jupyter:

```bash
jupyter notebook
```

Suggested execution order:

### Core Study

1. Phase 5 Dataset Generation
2. Phase 2.2 Reconstruction
3. Phase 6 Static Memory Persistence
4. Phase 6-T Temporal Observation
5. Phase 6-U Event-Centered Observation

### External Consistency Study

6. Phase 7-A.1 External Dataset Acquisition
7. Phase 7-A.2 GRMHD Image Harvesting
8. Phase 7-A Real Observation Consistency Study

### Final Analysis

9. Final Audit & Synthesis

---

# Repository Contents

Included:

* Source code
* Notebooks
* Simulation framework
* Analysis pipelines
* Documentation

---

## Project Status

Active Independent Research Project

Current Focus:

* Observational Memory
* Event Recoverability
* Memory Horizons
* Synthetic-to-Real Consistency
* Black-Hole Morphology Analysis
