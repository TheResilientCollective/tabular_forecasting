# Tabular Foundation Models for Epidemiology: Outbreak Forecasting and $R_t$ Estimation

## Overview

Tabular foundation models (TFMs) such as **TabPFN**, **TabPFN-TS**, **TabICL**, and **TabFM** perform zero-shot or in-context predictions on structured tables without task-specific training loops or heavy hyperparameter tuning. In epidemiology, they are emerging as tools for low-data infectious disease forecasting, policy scenario evaluation, and small-cohort risk modeling.

For outbreak forecasting and $R_t$ estimation, **tabular foundation models excel as fast, zero-shot short-term forecasters of observed incidence, while Bayesian renewal models remain necessary for estimating latent $R_t$**. Because $R_t$ is an unobservable, latent epidemiological parameter governed by generation intervals and reporting delays, off-the-shelf TFMs cannot infer it directly from raw case counts without mechanistic assumptions. Instead, the two approaches work best in distinct roles or combined in a hybrid pipeline.

---

## Core Capabilities in Epidemiology

* **Low-data and irregular surveillance forecasting:** Models like TabPFN-TS reframe time-series forecasting as tabular regression using historical sequences and covariates, showing strong short-term accuracy and calibration on sparse or irregular epidemic data (such as influenza, RSV, chickenpox, and dengue).
* **Intervention and policy evaluation:** TFMs have been used to model counterfactual or policy shifts, including estimating the effect of tighter restrictions during COVID-19 surges or evaluating immunization campaigns.
* **Small-cohort and multimodal clinical/genetic tables:** Because epidemiological cohorts often have limited sample sizes or rare outcomes alongside demographic, lab, EHR, and genetic variables, TFMs offer fast baselines without extensive feature engineering.

---

## Limitations & Epidemiological Pitfalls

* **Lack of mechanistic transmission priors:** Generic TFMs pre-trained on synthetic structural causal models or broad time series do not inherently encode compartmental epidemiological dynamics (such as transmission rates, immunity, or waning), which can limit long-horizon extrapolation or causal identifiability.
* **Population and ancestry non-stationarity:** In genetic epidemiology, allele effect sizes can shift across ancestries; off-the-shelf TFMs can degrade when in-context exemplars have high ancestry dispersion or underrepresent target groups.
* **Scale, compute, and calibration trade-offs:** While effective on small-to-moderate tables, larger datasets (e.g., >100,000 rows) or high-dimensional inputs often favor gradient-boosted trees on cost and throughput.

---

## Outbreak Forecasting vs. $R_t$ Estimation

### 1. Outbreak Forecasting: Where TFMs Excel
* **Short-term incidence (1–4 weeks ahead):** Benchmarks on Influenza-Like Illness (ILI), RSV, and dengue show that reframing time series as tabular regression with models like TabPFN-TS produces accurate, calibrated point and quantile forecasts without task-specific retraining.
* **Extreme data scarcity:** Early in an outbreak—when classical deep learning models (LSTMs, Transformers) overfit or fail—TFMs leverage synthetic causal priors to outperform individual statistical and deep learning baselines.
* **Exogenous covariates:** TFMs natively incorporate known-future and historical covariates (e.g., day-of-week reporting artifacts, weather, mobility, or school calendars) into the feature table alongside lagged case counts.

### 2. $R_t$ Estimation: Why Mechanistic Models Still Lead
* **Latent infection reconstruction:** Dedicated tools like EpiNow2 and epinowcast deconvolve observed cases back to infection dates using incubation periods, reporting delays, and right-truncation adjustments. A generic TFM treats case counts purely as statistical patterns and ignores these biological lags.
* **Generation interval sensitivity:** $R_t$ relies directly on the pathogen's generation time distribution. Without a renewal equation linking past infections to current secondary cases, a TFM cannot separate a true surge in transmissibility ($R_t > 1$) from a sudden clearing of testing backlogs.

---

## Hybrid Workflows for Outbreak & $R_t$ Modeling

To combine the speed and covariate handling of TFMs with epidemiological rigor, consider these three patterns:

| Workflow Pattern | How It Works | Best Used When |
| :--- | :--- | :--- |
| **Mechanistic → TFM** | Estimate historical $R_t$ and latent infections first using EpiNow2 or EpiEstim, then pass the smoothed $R_t$ trajectory + covariates (mobility, weather, policy) into a TFM to forecast future $R_t$. | You need short-term $R_t$ projections driven by external covariates without building a custom Bayesian hierarchical model. |
| **TFM Nowcasting → Renewal** | Use a TFM to impute missing or right-truncated recent surveillance counts using auxiliary signals (e.g., wastewater, syndromic ED visits), then feed the completed series into a renewal model. | Reporting delays are severe or irregular, and MCMC nowcasting is too slow across dozens of regions. |
| **TFM Surrogate** | Simulate thousands of synthetic outbreaks across varying $R_t$, immunity, and contact parameters using an SEIR or renewal model, and train or condition a TFM to map observed curves to $R_t$. | You need near-instantaneous $R_t$ inference across hundreds of jurisdictions where full Stan MCMC sampling is a bottleneck. |

---

*This is for informational purposes only. For medical advice or diagnosis, consult a professional. AI responses may include mistakes.*
