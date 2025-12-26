# Australian Economic Data Charts

A collection of Jupyter notebooks that fetch the latest economic data and generate charts covering key Australian social and economic statistics.

## Data Sources

- **Australian Bureau of Statistics (ABS)** - Primary source for Australian economic data
- **Reserve Bank of Australia (RBA)** - Monetary policy and financial data
- **OECD** - International comparisons
- **Bank for International Settlements (BIS)** - Central bank policy rates
- **FRED** - US Federal Reserve economic data
- **World Bank** - Commodity prices
- **DB.nomics** - International GDP data

## Notebooks

### ABS Monthly Series
| Notebook | ABS Cat. | Description |
|----------|----------|-------------|
| ABS Monthly Labour Force 6202 | 6202.0 | Employment, unemployment, participation rate |
| ABS Monthly Consumer Price Index Indicator 6484 | 6484.0 | Monthly CPI indicator |
| ABS Monthly Building Approvals 8731 | 8731.0 | Dwelling and building approvals |
| ABS Monthly International Trade in Goods 5368 | 5368.0 | Imports, exports, trade balance |
| ABS Monthly Arrivals Departures 3401 | 3401.0 | International travel movements |
| ABS Monthly Business Turnover Indicator | - | Business activity indicator |

### ABS Quarterly Series
| Notebook | ABS Cat. | Description |
|----------|----------|-------------|
| ABS Quarterly National Accounts 5206 | 5206.0 | GDP, economic growth |
| ABS Quarterly Consumer Price Index 6401 | 6401.0 | Official CPI, inflation |
| ABS Quarterly Wage Price Index 6345 | 6345.0 | Wage growth |
| ABS Quarterly Labour Account 6150 | 6150.0 | Hours worked, labour costs |
| ABS Quarterly Job Vacancies 6354 | 6354.0 | Job vacancies by industry |
| ABS Quarterly Business Indicators 5676 | 5676.0 | Company profits, wages |
| ABS Quarterly Producer Price Index 6427 | 6427.0 | Producer prices |
| ABS Quarterly Living Cost Index 6467 | 6467.0 | Cost of living by household type |
| ABS Quarterly Building Activity 8752 | 8752.0 | Construction work done |
| ABS Quarterly Financial Accounts 5232 | 5232.0 | Financial flows |
| ABS Quarterly Dwelling Stock 6432 | 6432.0 | Housing stock |
| ABS Quarterly Estimated Resident Population 3101 | 3101.0 | Population estimates |
| ABS Quarterly International Trade 5302 | 5302.0 | Balance of payments |
| ABS Quarterly Lending 5601 | 5601.0 | Lending indicators |

### ABS Combined Monthly/Quarterly Series
| Notebook | ABS Cat. | Description |
|----------|----------|-------------|
| ABS Monthly+Quarterly Retail Trade 8501 | 8501.0 | Retail turnover |
| ABS Monthly+Quarterly Detailed Labour Force 6291 | 6291.0 | Detailed labour statistics |
| ABS Monthly+Quarterly Household Spending | - | Household spending indicator |

### ABS Annual/Other Series
| Notebook | Description |
|----------|-------------|
| ABS Bi-annual Average Weekly Earnings 6302 | Earnings data |
| ABS Yearly National Accounts | Annual GDP and components |
| ABS Yearly State Accounts | State-level economic data |
| ABS Yearly Marriages and Divorces 3310 | Social statistics |
| ABS Population Growth Measures | Population growth analysis |
| ABS Census - Ad Hoc | Census data analysis |

### Inflation Analysis
| Notebook | Description |
|----------|-------------|
| ABS Inflation multi-measure | Multiple inflation measures compared |
| ABS Inflation model | Inflation modelling |
| ABS !Economy | Economic overview charts |

### RBA Data
| Notebook | Description |
|----------|-------------|
| RBA Selected Tables | Key RBA statistical tables |
| RBA SOMP Forecasts | Statement on Monetary Policy forecasts |

### International Comparisons
| Notebook | Description |
|----------|-------------|
| OECD - UR GDP CPI | Unemployment, GDP, CPI comparisons |
| BIS - CB policy rates | Central bank policy rate comparisons |
| FRED GDP International | International GDP from FRED |
| FRED Commodity Prices | Commodity price indices |
| World Bank Commodity Prices | World Bank commodity data |
| DB.nomics GDP International | International GDP via DB.nomics |

### Other Australian Data
| Notebook | Description |
|----------|-------------|
| AFSA Personal Insolvency | Personal insolvency statistics |
| ASIC Corporate Insolvency | Corporate insolvency data |
| ANGG-Quarterly-Greenhouse-Gas | Greenhouse gas emissions |

## Project Structure

```
├── notebooks/          # Jupyter notebooks
│   └── CHARTS/         # Generated chart output
└── .venv/              # Python virtual environment
```

## Setup

Python environment is managed with `uv`. To run:

```bash
source .venv/bin/activate
jupyter notebook notebooks/<notebook-name>.ipynb
```

## Notes

- Each notebook is self-contained and fetches the latest data when run
- Charts are output to `notebooks/CHARTS/`
- Directory structure designed to work on iPad using the Carnets app
