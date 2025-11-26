# Gale-Shapley Matching Algorithm Implementation

A Python project implementing the Gale-Shapley stable matching algorithm with advanced scoring metrics for matching students to universities based on preferences and capacity constraints.

## Project Overview

This project explores **stable matching problems** in the context of student-university assignments. It implements multiple variants of the Gale-Shapley algorithm and computes satisfaction/optimality metrics to evaluate matching quality.

### Key Features
- **Two Gale-Shapley variants**: Student-optimal and university-optimal algorithms
- **Stability verification**: Checks if matchings are stable (no blocking pairs)
- **Multi-metric scoring system**: Evaluates matchings using Top-K, cross-satisfaction, frustration, and regret metrics
- **Interactive CLI**: User-friendly command-line interface with color-coded output
- **Results visualization**: Heatmaps, scatter plots, and component analysis
- **Stable matching enumeration**: Finds all possible stable matchings using rotation poset theory

## Project Structure

```
.
├── src/
│   ├── main.py                 # CLI entry point - generates preferences and runs algorithms
│   ├── algorithms.py           # Gale-Shapley implementations
│   ├── score.py                # Final composite scoring function
│   ├── satisfaction.py         # Satisfaction-based metrics
│   ├── frustration.py          # Frustration-based metrics
│   ├── regret.py               # Regret-based optimality metrics
│   ├── top_k.py                # Top-K satisfaction scoring
│   ├── swine.py                # Swine transformation
│   ├── visualisation.py        # Plotting and analysis functions
│   └── utils.py                # Utility functions for JSON loading
├── data/
│   ├── etudiants.json          # Student names database
│   └── etablissements.json     # University/institution names database
├── results_rapport/            # Output directory for results (auto-generated)
└── README.md
```

## Core Algorithms

### Gale-Shapley Student-Optimal (`gale_shapley_etudiant_optimal`)
Students propose to universities in order of preference. A university accepts if:
- It has capacity available, OR
- The proposing student ranks higher than a currently matched student

**Guarantee**: Results in a stable matching that maximizes student welfare.

### Gale-Shapley University-Optimal (`gale_shapley_university_optimal`)
Universities propose to students in order of their preference list. A student accepts if:
- They are not yet matched, OR
- The proposing university ranks higher than their current match

**Guarantee**: Results in a stable matching that maximizes university/institution welfare.

## Scoring Metrics

The composite score combines four metrics with configurable weights (α, β, γ):

### 1. **Top-K Score** (`top_k.py`)
Measures the percentage of matches within each participant's top-K preferences.
- For students: Are universities in their top-K?
- For universities: Are students in their top-K?
- Weight: **α** (default 0.4)

### 2. **Cross-Satisfaction** (`satisfaction.py`)
Harmonic mean of individual satisfaction scores using the formula:
```
S = (n - rank - 1) / (n - 1)
```
Captures mutual satisfaction in each pair matching.
- Weight: **β** (default 0.3)

### 3. **Frustration** (`frustration.py`)
Measures frustration ratio for both students and institutions:
- **Student frustration**: Proportion of preferred universities unable to accept them
- **Institution frustration**: Proportion of preferred students who chose elsewhere
- Returns stability measure F = 1 - frustration_ratio

### 4. **Regret/Optimality** (`regret.py`)
Compares current matching against ideal matchings (student-optimal and institution-optimal):
- **Optimality = 1 - regret_global**
- Weight: **γ** (default 0.3, where γ = 1 - α - β)

**Final Score Formula**:
```
score_final = α × top_k_global + β × satisfaction_croisee + γ × optimality
```

## Usage

### Installation
```bash
pip install typer matplotlib numpy pandas seaborn
```

### Running the Main Algorithm

Navigate to the `src/` directory and run:

```bash
cd src
python -m typer main run
```

The CLI will prompt you for:
1. **Number of students** (max available in database)
2. **Number of universities** (max available in database)
3. **Capacity per institution** (how many students each can accept)

### Example Session
```bash
$ python src/main.py
Number of students to use (max 100): 11
Number of universities to use (max 84): 11
Capacity of each institution: 1

--- Generated preferences preview ---
[Shows first 1 students and institutions with their preference lists]



--- Score Calculation ---
Score (student-optimal): {...}
Score (university-optimal): {...}

Results saved to: ./results_rapport/20251119_171111/preference_et_resultats.json

$ python src/visualisation.py
[Graphs made from the results repertory will pop on the screen ]



$ python src/swing.py
[To transform unstable matching into a stable one with better satisfaction]
```

### Output Files

Each run creates a timestamped JSON file containing:
```json
{
  "preferences": {
    "etudiants": {...},
    "etablissements": {...},
    "capacites": {...}
  },
  "matchings": {
    "etudiant_optimal": {...},
    "universite_optimal": {...}
  },
  "scores": {
    "student_optimal": {...},
    "university_optimal": {...}
  }
}
```

## Visualization Functions (`visualisation.py`)

The visualization module provides analysis tools for understanding matching quality:

### 1. **Satisfaction vs Score Visualization**
```python
plot_satisfaction_vs_scorefinal(results)
```
Scatter plot showing relationship between average satisfaction and final score across different matchings.

### 2. **Score vs Weights Heatmap**
```python
plot_score_vs_weights_recompute(
    score_fn,
    matching,
    matching_ideal_etu,
    matching_ideal_uni,
    prefs_etus,
    prefs_unis,
    k=3,
    step=0.1
)
```
Shows how the final score varies as you adjust weight parameters (α, β, γ).
- X-axis: α (Top-K weight)
- Y-axis: β (Satisfaction weight)
- Color: Score value (viridis colormap)
- Note: γ = 1 - α - β

### 3. **Matching Components Comparison**
```python
plot_matching_components(results)
```
Bar chart comparing four components (Top-K, Satisfaction, Optimality, Final Score) across multiple matching results.


## Data Format

### Input Data (`data/etudiants.json`, `data/etablissements.json`)

**Students JSON**:
```json
{
  "nom": ["Student1", "Student2", ...]
}
```

**Universities JSON**:
```json
{
  "etablissements_superieurs_francais_complet": [
    {"nom": "University1"},
    {"nom": "University2"},
    ...
  ]
}
```

### Preference Data Format (Internal)
```python
prefs_etudiants = {
    "Student1": ["Uni3", "Uni1", "Uni2", ...],
    "Student2": ["Uni2", "Uni3", "Uni1", ...],
    ...
}

prefs_unis = {
    "Uni1": ["Student2", "Student5", "Student1", ...],
    "Uni2": ["Student1", "Student3", "Student2", ...],
    ...
}
```

## Configuration Parameters

Adjust in `src/main.py` (lines 82-86):

```python
score_etud_opt = score_final(
    ...,
    k=3,              # Top-K threshold
    alpha=0.4,        # Top-K weight
    beta=0.3,         # Satisfaction weight
    gamma=0.3         # Optimality weight (auto-set: 1 - α - β)
)
```


## Key Insights

- **No perfect match**: Different algorithms optimize for different parties (students vs. institutions)
- **Stability ≠ Optimality**: A stable matching can be suboptimal for overall satisfaction
- **Weight sensitivity**: Final scores vary significantly with α, β, γ choices
- **Solution space**: Multiple stable matchings exist between extremes
- **Tradeoffs**: Improving student satisfaction often worsens institution satisfaction

## Dependencies

- `typer`: CLI framework with color output
- `matplotlib`: Plotting
- `numpy`: Numerical computations
- `pandas`: Data analysis
- `seaborn`: Statistical visualization
- Standard library: `json`, `random`, `pathlib`, `datetime`, `copy`


