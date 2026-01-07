# MSc Thesis Code

# Offshore Wind Capability-Based CfD – Thesis Code Repository

This repository contains the Python scripts developed and used for the analysis presented in the Master thesis:

"Mitigation of Financial Risks in Offshore Wind Farm Projects through Contracts for Difference (CfDs)"

The code implements a modular and parameter-driven analysis pipeline for offshore wind projects, allowing application to different electricity price scenarios, support mechanisms, and case studies by modifying input data and script parameters.

---

## Repository structure

The `code/` directory contains standalone Python scripts grouped by analytical function:

- Electricity price scenario generation under alternative volatility and correlation assumptions
- Offshore wind capability and generation estimation using hourly wind data and wake effects
- Implementation of support mechanisms including capability-based CfDs, Feed-in Premiums, and SDE-type schemes
- Financial performance metrics including revenue volatility, downside risk, NPV, DSCR, and consumer cost

Scripts are not case-study specific. Different markets, scenarios, and projects are analysed by updating input data paths and key parameters defined at the top of each script.

---

## Suggested execution order

A typical analysis workflow follows these steps:

1. Electricity price scenario generation
2. Capability and generation estimation
3. Support mechanism settlement calculations
4. Financial and risk metric evaluation
5. Strike price sensitivity analysis
6. Market isolation effect to compare support mechanism (DK)

Intermediate results are stored as CSV files and used as inputs for subsequent steps.

---

## Requirements

The analysis was performed using Python 3.9+ and common scientific libraries such as NumPy, pandas, matplotlib, and PyWake.  
Exact library versions may be adapted depending on the execution environment.

---

## Reproducibility note

This repository is provided for documentation and reproducibility purposes.  
Users intending to replicate or extend the analysis must supply their own input datasets and adjust file paths and parameters accordingly.
