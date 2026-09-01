# Deep Hedging Under Realistic Market Frictions

**Paper:** [arXiv:2608.29025](https://arxiv.org/abs/2608.29025)

A regime-conditional empirical study comparing classical option-hedging strategies (Black-Scholes delta, Leland, Whalley-Wilmott) against deep hedging (LSTM and feedforward neural networks) on five years of real BTC options data from Deribit.

## Key finding

Under realistic transaction costs, the Whalley-Wilmott no-trade band strategy significantly reduces trading costs (~88% fewer trades) compared to continuous delta-hedging, with directionally better risk-adjusted returns. Three different deep hedging configurations, tested across two architectures and a 20x range of turnover penalties, all underperformed the classical benchmarks and never learned to trade less often.

## Repository structure

- `step1-2` — Data acquisition (Deribit/Tardis.dev) and cleaning
- `step3` — Classical hedging benchmarks (Black-Scholes delta, Leland, Whalley-Wilmott)
- `step4` — Evaluation framework (CVaR, block bootstrap significance testing)
- `step5` — Deep hedging models (LSTM and feedforward, CVaR loss with turnover penalties)
- `step6` — Policy interpretation and visualization
- `paper/` — Final paper (PDF)

## Data

Historical BTC options data sourced freely from [Tardis.dev](https://tardis.dev) (Deribit exchange), covering January 2020 to December 2024.

## Citation

If you reference this work, please cite:

```
Kumar, S. (2026). Deep Hedging Under Realistic Market Frictions: A Regime-Conditional
Empirical Study of Dynamic Option Hedging on Bitcoin Options. arXiv:2608.29025.
```