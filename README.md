# BUARO

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24221770.svg)](https://doi.org/10.5281/zenodo.24221770)

**Bayesian Urban Adaptation and Resilience Optimization, built on real European data.**

BUARO turns the Bayesian Urban Adaptation and Resilience Optimization Model from Dimitrios Sargiotis, *AI for Planetary Resilience* (Springer, 2027, Section 1.5.2, Table 1.29, [link.springer.com/book/9783032355393](https://link.springer.com/book/9783032355393)) into open software that runs on real data.

The book's model scores every urban zone for climate resilience and picks the adaptation measure that gives the most resilience for the money:

```math
R_z=\mathbb{E}_{\theta}\left[\sum_{h\in\mathcal{H}}P(h\mid\theta)\cdot\left(1-V(h,z)\cdot\frac{E(z)}{A(z)}\right)\right]
```

```math
s^{*}=\arg\max_{s\in\mathcal{S}}\left\lbrace U(s)=\sum_{z}w_{z}\,R_{z}(s)-C(s)\right\rbrace
```

BUARO extends it so that every term is learned from public data, and so that the climate scenarios are the ones set by the proposed European Climate Resilience Law (COM(2026) 730): 1.5°C by 2030, 2°C by 2050 and 3°C by 2100.

## Why Attica first

Heat is the deadliest climate hazard in Europe. In summer 2024, Greece had the highest heat-related death rate in Europe, 574 deaths per million people (ISGlobal, Nature Medicine, 2025). Attica is home to 36.4% of the country's population (2021 census). If the method works here, it works where it matters most.

## Step 1.1: how vulnerable is each part of Attica to heat, and is it adapting?

Step 1 learns the vulnerability term V(h, z) for the heat hazard, for the seven NUTS 3 areas of Attica, and then tests it honestly. Step 1.1 adds a second, more robust way to set the normal level of deaths, and a test of whether Attica is adapting.

**What it does, in plain words**

1. Downloads weekly deaths for each area from Eurostat, and population by age.
2. Downloads daily temperature for each area (ERA5-Land reanalysis, two points per area).
3. Sets the normal level of deaths in two ways and keeps the one that forecasts unseen summers better (the rule is fixed in advance):
   - **All-weeks baseline**: every week of the year, a long-term trend and three seasonal waves.
   - **EuroMOMO-style baseline**: only the calm weeks of spring (16 to 25) and autumn (37 to 44), with a separate level for each year, following the method of the European mortality monitoring network [EuroMOMO](https://www.euromomo.eu/how-it-works/methods). A hard winter or a hot summer cannot pull it up or down.
4. Learns, for each area, how much deaths rise for each degree above a local heat threshold. The threshold itself is learned from the data.
5. Combines the seven areas with empirical Bayes shrinkage, so that a small area with noisy data borrows strength from the others.
6. **The adaptation test.** Lets the heat effect change from year to year, to see whether each degree of heat is becoming less deadly.
7. **The forecast test.** For each summer from 2023 onwards, the model is trained only on data up to 31 May and asked to forecast June to September. It is compared with an identical model that knows nothing about heat. If knowing about heat does not make the forecast better, the result says so.

**The all-weeks model for each area r and week t**

```math
\log \mathbb{E}[D_{rt}]=\log P_{rt}+\alpha_r+\gamma_r t+\sum_{k=1}^{3}\left(a_{rk}\sin\frac{2\pi k d_t}{365.25}+b_{rk}\cos\frac{2\pi k d_t}{365.25}\right)+\sum_{l=0}^{1}c_{rl}\,(T^{\text{cold}}_r-\bar{T}_{r,t-l})_{+}+\sum_{l=0}^{1}\beta_{rl}\,(T^{\max}_{r,t-l}-\tau_r)_{+}
```

**The EuroMOMO-style model**, in two stages: a baseline fitted on calm weeks only, then the heat effect on weeks 16 to 44 measured against it

```math
\log B_{rt}=\log P_{rt}+\alpha_{r,y(t)}+a_r\sin\frac{2\pi d_t}{365.25}+b_r\cos\frac{2\pi d_t}{365.25},\qquad t\in\text{weeks }16\text{--}25,\ 37\text{--}44
```

```math
\log \mathbb{E}[D_{rt}]=\log B_{rt}+\kappa_r+\sum_{l=0}^{1}\beta_{rl}\,(T^{\max}_{r,t-l}-\tau_r)_{+}
```

**Pooling across areas and the adaptation test**

```math
\beta_r=\beta_{r0}+\beta_{r1},\qquad \beta_r\sim\mathcal{N}(\bar{\beta},\,\tau^{2}),\qquad \beta_r(t)=\beta_r+\delta_r\,(\text{year}_t-2019)
```

D is weekly deaths, P is population, d is day of year, y(t) is the year of week t, and τ_r is the learned heat threshold. A negative pooled δ means heat is becoming less deadly per degree. Weeks from 2020-W10 to 2022-W26 are left out of training, because COVID-19 deaths would distort the heat signal.

## Interactive results page

Every run also rebuilds an interactive page with charts: [uomoathens.github.io/buaro](https://uomoathens.github.io/buaro/). It lets you explore any summer and area week by week, see how vulnerable each area is, check each summer's response to heat without any model, follow the adaptation test, and compare the forecasts. The page is self-contained, works in light and dark mode, and every chart can also be shown as a table.

## Results

Everything the pipeline produces is in [`results/`](results/):

- [`results/SUMMARY.md`](results/SUMMARY.md): the findings in plain language, with charts
- `results/buaro_attica_step1.xlsx`: every result, by area, by week and by baseline, plus the adaptation test, the area features, the warming levels and the measures
- `results/data_panel.xlsx`: every weekly number the model used (deaths, temperature, population), so anyone can check the results
- `results/vulnerability.png`, `results/backtest_YYYY.png`, `results/skill.png`, `results/baseline_comparison.png`: the charts
- `results/run_metadata.json`: what was run, on which data, and when

## It runs by itself

A GitHub Actions workflow ([`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml)) runs the tests, downloads fresh data, refits the model and commits new results. It runs on the 4th of every month (Eurostat adds new weeks of deaths data regularly), on every change to the code, and on demand from the Actions tab. No API keys are needed for step 1.

## Data sources

BUARO uses only official statistics, official Earth observation services and peer-reviewed research.


| Data | Source | Licence |
|---|---|---|
| Weekly deaths by NUTS 3 area | Eurostat `demo_r_mwk3_t` (fallback `demo_r_mwk3_ts`) | Eurostat reuse policy, free with attribution |
| Population by age and NUTS 3 area | Eurostat `demo_r_pjangrp3` | Eurostat reuse policy, free with attribution |
| Daily temperature | ERA5-Land (Copernicus Climate Change Service), served by the Open-Meteo archive API | Copernicus data licence; Open-Meteo CC BY 4.0 |
| Residents and residents aged 65+ per 1 km cell | Eurostat Census 2021 population grid | Eurostat reuse policy, free with attribution |
| Land use, sealing classes and street tree cover | Copernicus Land Monitoring Service, Urban Atlas 2021 and Street Tree Layer 2021 | Copernicus data policy, free and open |
| Summer land surface temperature, 100 m | USGS Landsat 8 and 9 Collection 2 Level-2, via Microsoft Planetary Computer | No copyright restrictions (USGS) |
| Elevation | Copernicus DEM GLO-30 | Licence for the use of the Copernicus WorldDEM-30, free |
| Summer warming at 1.5°C, 2°C and 3°C | C3S Gridded dataset underpinning the Copernicus Interactive Climate Atlas (CMIP6), IPCC AR6 WGI Atlas warming levels | CC BY 4.0 |
| Air cooling relative to surface cooling of urban trees | Du et al. 2024, Environmental Research Letters 19:044037 | CC BY 4.0 |

Contains modified Copernicus Climate Change Service information. Neither the European Commission nor ECMWF is responsible for any use of this information. Weather data by [Open-Meteo.com](https://open-meteo.com/).

Contains Copernicus Land Monitoring Service information (Urban Atlas 2021 and Street Tree Layer 2021), adapted and modified: summarised per 1 km census cell, per 100 m cell and per NUTS 3 area. This work is not endorsed by the European Union.

Elevation produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30.

## Honest limits

- Deaths are weekly and by NUTS 3 area, not daily or by neighbourhood. In a comparison across 147 European regions, weekly data underestimated heat-related mortality by about 10% to 22% compared with daily data (Ballester et al. 2023, The Lancet Regional Health Europe, [doi.org/10.1016/j.lanepe.2023.100779](https://doi.org/10.1016/j.lanepe.2023.100779)).
- Temperature is reanalysis at about 11 km (0.1°), not street level.
- Shrinkage is empirical Bayes, a first step towards the fully Bayesian layer of the framework.
- The adaptation test assumes a straight-line change over time and has few post-pandemic summers, so it is a first signal, not proof.
- With seven areas, the link between heat risk and tree cover, sealing or age is an association between areas, not proof of cause. Dense, highly sealed fabric means Urban Atlas classes 11100 (sealing above 80%) and 11210 (50% to 80%).
- The comparison with published estimates is an order-of-magnitude check, not a validation: published studies use daily data and different models.

## Roadmap

| Step | Book term | What it adds | Data |
|---|---|---|---|
| 1 and 1.1 | V(h, z) | Heat vulnerability by NUTS 3 area, adaptation test, tested on unseen summers | Eurostat, ERA5-Land |
| 2 | E(z), A(z) | 1 km exposure (residents and residents aged 65+) and adaptive capacity (street tree cover, green urban areas, sealed urban fabric), weighted by where people live, and a random-effects meta-regression of each area's heat effect on these features | Eurostat Census Grid 2021, Copernicus Urban Atlas 2021 and Street Tree Layer 2021 |
| 3.0 | LST(c, t) | Summer land surface temperature on a 100 m grid from every clear Landsat 8 and 9 scene since 2014 (USGS Collection 2 Level-2, band ST_B10, cloud and water masked with QA_PIXEL), set against 100 m tree cover, green urban areas, sealed fabric and elevation, compared only within each 1 km cell so that coast, hills and wind cannot pose as cooling, trained on 2014 to 2022 and tested on every unseen summer from 2023 onwards | USGS Landsat Collection 2 via Microsoft Planetary Computer, Copernicus DEM GLO-30, Urban Atlas 2021, Street Tree Layer 2021 |
| 3.2 | P(h \| θ) | Heat deaths per summer at the 1.5°C, 2°C and 3°C reference levels of the proposed European Climate Resilience Law (COM(2026) 730): observed summers shifted by the CMIP6 monthly change in daily maximum temperature at each level (delta method), today's population and vulnerability | C3S Atlas dataset (CMIP6), IPCC AR6 Atlas warming levels |
| 4 (this release) | s*, C(s) | Heat deaths avoided by street trees and green urban areas, per area and at each warming level, from two independent lines of evidence: cooling measured from space turned into air cooling with the published European air-to-surface ratio, and the difference in heat risk between areas, kept inside the range Attica already shows | Steps 1 to 3.2, Du et al. 2024 (Environmental Research Letters 19:044037) |

## Run it locally

```bash
pip install -r requirements.txt
python -m pytest -q
python -m buaro.run --config config/attica.yaml --out results --cache data/raw
```

## Credits

Framework: Dimitrios Sargiotis, *AI for Planetary Resilience*, Springer Nature Switzerland, 2027.

## Licence and citation

Copyright © 2026 Dimitrios Sargiotis, National Technical University of Athens.

Licensed under the EUPL v1.2, the open source licence of the European Union. Anyone may use, study and adapt BUARO; whoever distributes it or offers it as an online service, changed or unchanged, must do so under the same licence, keep this copyright notice and make the source code available. The full text is in [LICENSE](LICENSE).

The method develops the Bayesian Urban Adaptation and Resilience Optimization Model of the book *AI for Planetary Resilience* (D. Sargiotis, Springer Nature Switzerland, 2027). If you use BUARO, please cite it as given in [CITATION.cff](CITATION.cff), or with the "Cite this repository" button on GitHub:

> Sargiotis, D. (2026). *BUARO: Bayesian Urban Adaptation and Resilience Optimization* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24221770

The data keep their own licences, listed in the table of sources above.

