# Hardware: a real follow-me drone on a student budget

Goal: the cheapest drone that can still run the **same code** as the simulator and fly the defense demo
safely. Prices are rough AliExpress-level estimates in USD (late 2026); check local shops in Kazakhstan, they
can be cheaper for batteries and shipping is free.

## Shopping list

| Part | Choice | Approx. price | Why / cheaper option |
|---|---|---|---|
| Frame | **3D-printed** on the uni printer (F450-style arms) | ~$5 filament | F450 clone kit ~$15–20 if printing fails |
| Motors | 4 × 2212 920KV | ~$25–35 | Buy as a "motor + ESC + props" combo kit to save ~20 % |
| ESCs | 4 × 30A BLHeli (or one 4-in-1 ESC) | ~$20–30 | A flight-controller + 4-in-1 ESC "stack" saves wiring |
| Propellers | 1045, buy 3–4 sets | ~$5–8 | They break — spares are part of the budget |
| Flight controller | SpeedyBee F405 V4 running **ArduPilot** | ~$40 (FC) / ~$70 (stack with ESC) | Cheapest ArduPilot-capable F4 boards ~$20–25, but SpeedyBee is better documented |
| Altitude + drift | **MicoAir MTF-01** (optical flow + 8 m laser) | ~$30–40 | Essential: holds height and position indoors without GPS |
| Obstacles | VL53L5CX 8×8 distance sensor (front) | ~$15–25 | Read by the Pi over I2C |
| Companion computer | Raspberry Pi Zero 2 W | ~$15–20 | Only streams video + forwards MAVLink; the laptop does the thinking |
| Camera | Pi camera (OV5647, 5 MP) | ~$8–15 | Wide-angle version helps keep you in frame |
| Radio (safety!) | RadioMaster Pocket ELRS + RP1 receiver | ~$60 + ~$12 | Do not skip: manual override is a safety rule |
| Battery | 2 × 3S 2200 mAh LiPo | ~$40–50 | Bought locally |
| Charger | Budget balance charger | ~$20–35 | Must balance-charge LiPos |
| Small parts | XT60, wires, M3 screws, zip ties, 3D-printed prop guards | ~$10–15 | |
| GPS + compass | BN-880 | ~$15–20 | **Optional** — only for outdoor flights / return-to-launch. Indoors the MTF-01 does the job |

**Total: roughly $300–380**, or **~$280–360 without GPS** for an indoor-first demo.

Already owned (costs nothing extra): **laptop** (vision, gestures, voice, RL policy), **phone or headset** (the
microphone for voice commands — on the drone, propeller noise would drown out your voice), uni **3D printer**.

## Where the money is saved
- The laptop does all heavy computing, so the drone only needs a $15 Pi Zero instead of a $100+ Jetson.
- No depth camera ($150+): the 8×8 laser sensor and the downward MTF-01 are much cheaper and enough for a demo.
- 3D-printed frame and prop guards instead of a bought frame.
- GPS skipped for the first (indoor) demo.
- ArduPilot is free and already does stabilisation, altitude hold and failsafes.

## Where **not** to save
- **The radio transmitter** (manual override) — a safety rule and usually required by the examiners.
- **Batteries and charger** — cheap no-name LiPos and chargers are a fire risk.
- **Spare propellers** — you will break some while learning.

## What the simulator assumes vs. this hardware
| Simulator | Real drone | Gap to handle |
|---|---|---|
| 64×48 depth camera, 10 m | 8×8 laser grid, ~4 m | Train / test the pilot with the reduced 8×8, 4 m input (planned) |
| Perfect altitude hold | ArduPilot + MTF-01 | Add altitude noise / drift in simulation |
| No wind | Wind, gusts, air pressure changes | Domain randomisation (wind, noise, latency) + ArduPilot SITL with wind |
| Instant commands | Wi-Fi latency 100–300 ms | Random command delay in simulation |

Before buying: register the drone (Kazakhstan rules), and first flights indoors or in an open field, low and
slow, with prop guards and nobody underneath.
