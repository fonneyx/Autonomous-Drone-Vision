# Documentation

| Document | Contents |
|---|---|
| [Architecture](architecture.md) | Components, design rules, tap to track, avoidance, vehicles, measurements |
| [Requirements](requirements.md) | Purpose, users, functional requirements R1-R8 |
| [Analysis](analysis/README.md) | Data flow diagrams (context, level 0, level 1), activity and sequence diagrams |
| [Database](database/README.md) | ER diagram, `schema.sql` with test data and queries, query results |
| [Interface](ui/README.md) | Navigation, wireframes, traceability, colours, usability evaluation |
| [Hardware](hardware.md) | Low-cost ArduPilot drone parts list and test plan |

The diagrams and wireframes are generated from code, so they always match:

```bash
python scripts/docs/build_diagrams.py     # analysis + database diagrams
python scripts/docs/build_ui.py           # navigation + wireframes
```
