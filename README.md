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
- **Yahoo Finance** - Daily commodity futures, metals, energy and equity indices
- **Other Australian agencies** - AIP (petrol prices), DCCEEW (petroleum), AFSA/ASIC (insolvency), Home Affairs (visas)

## Notebooks

### ABS Monthly Series
| Notebook | ABS Cat. | Description |
|----------|----------|-------------|
| ABS Monthly Labour Force 6202 | 6202.0 | Employment, unemployment, participation rate |
| ABS Monthly Building Approvals 8731 | 8731.0 | Dwelling and building approvals |
| ABS Monthly International Trade in Goods 5368 | 5368.0 | Imports, exports, trade balance |

### ABS Quarterly Series
| Notebook | ABS Cat. | Description |
|----------|----------|-------------|
| ABS Quarterly National Accounts 5206 | 5206.0 | GDP, economic growth |
| ABS Quarterly National Accounts 5206 No 2 | 5206.0 | Productivity analysis from the national accounts |
| ABS Quarterly Wage Price Index 6345 | 6345.0 | Wage growth |
| ABS Quarterly Labour Account 6150 | 6150.0 | Hours worked, labour costs |
| ABS Quarterly Job Vacancies 6354 | 6354.0 | Job vacancies by industry |
| ABS Quarterly Business Indicators 5676 | 5676.0 | Company profits, wages |
| ABS Quarterly Producer Price Index 6427 | 6427.0 | Producer prices |
| ABS Quarterly Living Cost Index 6467 | 6467.0 | Cost of living by household type |
| ABS Quarterly Building Activity 8752 | 8752.0 | Construction work done |
| ABS Quarterly Financial Accounts 5232 | 5232.0 | Financial flows |
| ABS Quarterly Dwelling Stock 6432 | 6432.0 | Housing stock |
| ABS Quarterly International Trade 5302 | 5302.0 | Balance of payments |
| ABS Quarterly Lending 5601 | 5601.0 | Lending indicators |
| ABS Quarterly Capital Expenditure 5625 | 5625.0 | Private new capital expenditure by asset type and industry |

### ABS Combined Monthly/Quarterly Series
| Notebook | ABS Cat. | Description |
|----------|----------|-------------|
| ABS Monthly+Quarterly Retail Trade 8501 | 8501.0 | Retail turnover |
| ABS Monthly+Quarterly Detailed Labour Force 6291 | 6291.0 | Detailed labour statistics |
| ABS Monthly+Quarterly Household Spending | - | Household spending indicator |

### ABS SDMX API Series
| Notebook | Description |
|----------|-------------|
| ABS-SDMX-Inflation-multi_measure | Inflation measures via the ABS SDMX API |
| ABS-SDMX-Monthly-Labour-Force-6202 | Labour force via the ABS SDMX API |
| ABS-SDMX-Monthly-Household-Spending-Indicator-5682 | Household spending indicator via the ABS SDMX API |

### ABS Annual/Other Series
| Notebook | Description |
|----------|-------------|
| ABS Bi-annual Average Weekly Earnings 6302 | Earnings data |
| ABS Yearly National Accounts | Annual GDP and components |
| ABS Yearly State Accounts | State-level economic data |
| ABS Yearly Government Finance Statistics 5512 | Government revenue and expenditure |
| ABS Yearly Taxation Revenue 5506 | Taxation revenue |
| ABS Yearly Marriages and Divorces 3310 | Social statistics |
| ABS Business Entries and Exits 8165 | Business entry and exit rates |
| ABS Earnings by Education 6337 | Earnings by educational attainment |
| ABS Personal Income by Remoteness 6524 | Personal income by remoteness area |
| ABS LFS - Household dynamics | Household dynamics from the detailed Labour Force Survey (6291.0) |
| ABS Population | Population: overseas arrivals/departures (3401.0) and resident population (3101.0) |
| ABS Census - Ad Hoc | Census data analysis |

### Inflation and Economy Analysis
| Notebook | Description |
|----------|-------------|
| ABS Inflation multi-measure | Multiple inflation measures compared |
| ABS Recession | Recession indicators and analysis |

### RBA Data
| Notebook | Description |
|----------|-------------|
| RBA Selected Tables | Key RBA statistical tables |
| RBA SOMP Forecasts | Statement on Monetary Policy forecasts |

### International Comparisons
| Notebook | Description |
|----------|-------------|
| OECD - UR CPI | Unemployment and CPI comparisons |
| OECD Global Savings Glut | Savings glut analysis: bond yields, investment and land values (OECD) |
| BIS - CB policy rates | Central bank policy rate comparisons |
| FRED GDP International | International GDP from FRED |
| FRED Stagflation | Stagflation-era international comparison, 1970-1995 (FRED) |
| DB.nomics GDP International | International GDP via DB.nomics |
| Productivity AU vs US | Labour productivity, unit labour costs and rates: Australia vs US |
| World Bank Global Savings Glut | Global savings glut: current account balances (World Bank) |

### Commodities and Energy
| Notebook | Description |
|----------|-------------|
| YAHOO_daily_commodities | Daily commodity futures, energy, metals and ASX indices (Yahoo Finance) |
| FRED Commodity Prices | Commodity price indices (FRED) |
| World Bank Commodity Prices | World Bank commodity data |
| AIP_petrol_prices | Australian petrol terminal gate prices (AIP) |
| DCCEEW Petroleum Statistics | Australian petroleum consumption and import cover |

### Other Australian Data
| Notebook | Description |
|----------|-------------|
| AFSA Personal Insolvency | Personal insolvency statistics |
| ASIC Corporate Insolvency | Corporate insolvency data |
| ANGG-Quarterly-Greenhouse-Gas | Greenhouse gas emissions |
| Domestic and Family Violence | Domestic and family violence statistics |
| Home Affairs Temporary Visa Workforce | Temporary visa holders in the workforce |

### Development/Test
| Notebook | Description |
|----------|-------------|
| Test mgplot | mgplot library test/scratch notebook |
| Test readabs | readabs library test/scratch notebook |

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
