<img src="../../../Images/brand-kit/logo/logo-horizontal-light.png" alt="Mobius Tensors" width="200" align="right" />

# Histor pricing

*Internal: not for clients or partners yet. Draft, 2026-10-13.*

## Software (one-time, per site)

| | Price |
| --- | --- |
| Site, up to 8 cameras | **$6,000** |
| Each extra camera | **$400** |

## Hardware (at cost plus a small margin)

Every quote includes the hardware, sized from the camera count; untick what the customer already has (often the cameras).

| Item | Price | How many |
| --- | --- | --- |
| Server (CPU) | $1,800 | 1 per 8 cameras |
| PoE switch, 16 ports | $300 | 1 per 15 cameras |
| IP camera | $150 | 1 per camera |
| NVIDIA GPU (optional) | $700 | only when needed |

**These hardware prices and the cameras-per-server rule are placeholders: replace them with our supplier prices and the pilot's measured capacity.** A standard CPU server is the default: a GPU is added only when a site needs it (more cameras per server, heavier models), so the customer doesn't pay for one they don't need.

Example, a full 8-camera system: $6,000 software + $3,300 hardware (1 server, 1 switch, 8 cameras) = $9,300, then $1,440 a year.

## Yearly subscription (updates, fixes, support)

**$15 per camera per month**, billed yearly ($180 per camera per year), with a **minimum of $600 a year** per site. No cap tied to the software price: support work grows with the cameras. Decided 2026-10-13.

| Cameras | Software | Yearly |
| --- | --- | --- |
| 2 | $6,000 | $600 (minimum) |
| 4 | $6,000 | $720 |
| 8 | $6,000 | $1,440 |
| 12 | $7,600 | $2,160 |
| 16 | $9,200 | $2,880 |
| 32 | $15,600 | $5,760 |

The minimum applies up to 3 cameras. In a pitch: "$15 per camera per month", or "less than 50 cents per camera per day".

## Everything is part of the system

The site price includes every module Histor has: memory and plain-language search, areas and lines, crowds, bags left behind, wrong way, plate reading, person down and activity, fire and smoke (once our model is trained), sound alerts, face search (with the client's written approval), camera health, alert checking, dashboard and Telegram. Final, 2026-10-13: no paid modules.

## Extra modules (only if needed)

Only a detection Histor doesn't have yet costs extra: **$5,000 per extra module**, built for the site, one-time, no yearly fee.

## Partner commissions on these prices

Rates from the [Partner Program study](<Partner Program — Study.md>): on the software and extra modules only, never on hardware or the yearly subscription.

| Sale | Software | Silver (12%) | Gold (25%) | Platinum (40% off) |
| --- | --- | --- | --- | --- |
| 8 cameras | $6,000 | $720 | $1,500 | buys at $3,600 |
| 16 cameras | $9,200 | $1,104 | $2,300 | buys at $5,520 |
| One extra module | $5,000 | $600 | $1,250 | buys at $3,000 |

The calculator for quotes and estimates is on the website, unlisted: `internal/pricing.html` (prices in `internal/prices.js`).

## Still open

- Pilot: free, or paid and credited against the purchase.
- Currency and tax on invoices (USD; VAT where it applies).
