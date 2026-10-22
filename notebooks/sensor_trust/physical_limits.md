# Physical limits

Derived from `Particulars.xlsx` and the towing-tank table (`reference/towing_with_psv.csv`). These are **proposals for the team to agree**. The same values are in `src/trust.py` (`LIMITS`), so the notebooks use them directly.

- **Normal:** the usual range.
- **Check:** possible but rare. Look at the context.
- **Hard limit:** physically impossible, almost certainly a sensor error.

## Ship data used

| Item | Value |
|---|---|
| Engine SMCR (max continuous) | 9,500 kW at 105 rpm |
| Engine CSR (normal service) | 8,075 kW at 99.5 rpm |
| Propeller | D = 6.8 m, P/D = 0.6742 → pitch 4.58 m, 4 blades |
| Lpp / B | 217 m / 32.25 m |

| Condition | Draft fwd / aft | Volume | ≈ Mass (×1.025 t/m³) |
|---|---|---|---|
| Ballast | 5.0 / 7.8 m | 36,134 m³ | 37,000 t |
| Heavy ballast | 4.4 / 9.4 m | 39,024 m³ | 40,000 t |
| Design | 12.2 / 12.2 m | 73,346 m³ | 75,200 t |
| Scantling | 14.25 / 14.25 m | 86,941 m³ | 89,100 t |

## Limits

| Signal | Normal | Check | Hard limit | Based on |
|---|---|---|---|---|
| `shaft_rpm` ahead | 0–100 | > 105 | > 110 | SMCR 105 rpm + small overspeed margin |
| `shaft_rpm` astern | 0 to −70 | < −90 | < −110 | astern usually limited below ahead rpm |
| `shaft_kw` | 0–8,075 | > 8,075 | > 10,000 | CSR 8,075 / SMCR 9,500 kW |
| `shaft_torque_nm` (really kN·m) | 0–750 | > 864 | > 950 | torque at SMCR = 9,500 / (2π·105/60) = 864 kN·m |
| `fo_kgh` | 300–1,600 sailing | > 1,750 | > 2,000 | 9,500 kW × ~0.18 kg/kWh |
| `me_load_pct` | 0–90 | > 100 | > 110 | % of engine max |
| `stw_kn` | 0–15 | > 16 | > 18 | towing tank ends at 16 kn; max seen 14.9 |
| `sog_kn` | 0–17 | > 18 | > 21 | STW + up to ~3 kn current |
| Draft (noon) | 4.4–14.25 m | outside | < 3 or > 14.5 | heavy ballast fwd / scantling |
| Trim, aft − fwd (noon) | −1 to +5 m | > +5.5 | > +7 or < −2 | heavy ballast 5 m by the stern |
| Displacement (noon) | 37,000–89,100 t | outside | < 30,000 or > 92,000 | ballast / scantling × 1.025 |

## Physics relations used as checks

| Relation | Expected | What a mismatch means |
|---|---|---|
| Power = 2π · rpm/60 · torque | ratio 1.00 | torquemeter internal fault or units |
| `shaft_rpm` = `me_rpm` | equal (direct drive) | rpm dropout or timing mismatch |
| SFOC = fuel / power | ~165–190 g/kWh, flat over time | step change → power or fuel sensor drift |
| Apparent slip = 1 − STW / (4.58 m · rpm/60) | ~0–20 % | negative → speed log too high; rising over months → fouling or speed log drift |
| Thrust from physics | ~700–900 kN at 5.7 MW, 12.8 kn | logged −9,172 "kN" → unknown unit / calibration |

The SFOC range is a typical value for this engine type, not from the particulars.
