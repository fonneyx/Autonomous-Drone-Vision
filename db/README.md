# Database (Phase 2)

PostgreSQL + PostGIS (TimescaleDB optional for telemetry). Alembic migrations go in `db/migrations/`;
the ER model is `docs/uml/schema.dbml`. Media files stay on disk; tables keep their path and SHA-256.
