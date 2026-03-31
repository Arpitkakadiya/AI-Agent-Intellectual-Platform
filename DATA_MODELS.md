# Compliance Agent — Data Models

## Supabase Tables

### jurisdictions
| Column      | Type    | Notes                              |
|-------------|---------|-------------------------------------|
| id          | SERIAL  | Primary key                         |
| type        | TEXT    | federal / state / county / city     |
| name        | TEXT    | e.g. "California", "Los Angeles"    |
| parent_id   | INT     | FK → jurisdictions.id (nullable)    |
| state_code  | CHAR(2) | e.g. "CA", "TX"                     |
| fips_code   | TEXT    | For geocoding lookups (nullable)    |

### regulations
| Column          | Type        | Notes                           |
|-----------------|-------------|----------------------------------|
| id              | SERIAL      | Primary key                      |
| jurisdiction_id | INT         | FK → jurisdictions.id            |
| domain          | TEXT        | housing / pet / insurance        |
| category        | TEXT        | e.g. "ESA", "Rent Control"       |
| source_name     | TEXT        |                                  |
| url             | TEXT        |                                  |
| content         | TEXT        | Full scraped text                |
| content_hash    | TEXT        | SHA256 for change detection      |
| version         | INT         | Increments on each change        |
| is_current      | BOOL        | Only latest version = true       |
| effective_date  | DATE        | Nullable                         |
| created_at      | TIMESTAMPTZ | auto                             |

### regulation_embeddings
| Column        | Type          | Notes                    |
|---------------|---------------|--------------------------|
| id            | SERIAL        | Primary key              |
| regulation_id | INT           | FK → regulations.id      |
| embedding     | vector(3072)  | pgvector (Gemini dims)   |
| chunk_text    | TEXT          | 800-char chunk           |

### email_subscriptions
| Column        | Type        | Notes                         |
|---------------|-------------|-------------------------------|
| id            | SERIAL      | Primary key                   |
| email         | TEXT        |                               |
| jurisdiction_id | INT       | FK → jurisdictions.id         |
| subscribed_at | TIMESTAMPTZ | auto                          |
| is_active     | BOOL        | default true                  |

### regulation_updates
| Column               | Type        | Notes                      |
|----------------------|-------------|----------------------------|
| id                   | SERIAL      | Primary key                |
| regulation_id        | INT         | FK → regulations.id        |
| update_summary       | TEXT        |                            |
| affected_jurisdictions | JSONB     | list of jurisdiction ids   |
| detected_at          | TIMESTAMPTZ | auto                       |

### pet_policies
| Column                  | Type    | Notes                           |
|-------------------------|---------|----------------------------------|
| id                      | SERIAL  | Primary key                      |
| jurisdiction_id         | INT     | FK → jurisdictions.id            |
| esa_deposit_allowed     | BOOL    | Always false at federal level    |
| service_animal_fee      | BOOL    | Always false at federal level    |
| breed_restrictions      | JSONB   | list of restricted breed names   |
| max_pet_deposit_amount  | NUMERIC | Nullable — state sets this       |
| source_regulation_id    | INT     | FK → regulations.id              |

### insurance_requirements
| Column                   | Type    | Notes                     |
|--------------------------|---------|---------------------------|
| id                       | SERIAL  | Primary key               |
| jurisdiction_id          | INT     | FK → jurisdictions.id     |
| landlord_can_require     | BOOL    |                           |
| min_liability_coverage   | NUMERIC | Nullable                  |
| tenant_must_show_proof   | BOOL    |                           |
| notes                    | TEXT    | Nullable                  |
| source_regulation_id     | INT     | FK → regulations.id       |

### app_settings
| Column     | Type        | Notes                                  |
|------------|-------------|----------------------------------------|
| key        | TEXT        | Primary key (e.g. "use_db_source_registry") |
| value      | TEXT        | Feature flag value ("true"/"false")    |
| updated_at | TIMESTAMPTZ | auto                                   |

### regulation_sources
| Column          | Type        | Notes                                  |
|-----------------|-------------|----------------------------------------|
| id              | SERIAL      | Primary key                            |
| jurisdiction_id | INT         | FK → jurisdictions.id                  |
| source_name     | TEXT        | Human-readable name                    |
| url             | TEXT        | Unique — the scrape target             |
| domain          | TEXT        | Default "housing"                      |
| category        | TEXT        | Default "General"                      |
| state_code      | CHAR(2)     | Nullable                               |
| is_active       | BOOL        | Per-source toggle (default true)       |
| last_scraped_at | TIMESTAMPTZ | Updated after each scrape attempt      |
| last_error      | TEXT        | Nullable — last scrape error message   |
| created_at      | TIMESTAMPTZ | auto                                   |

## Pydantic Models (db/models.py)
- Jurisdiction, Regulation, RegulationEmbedding
- EmailSubscription, RegulationUpdate
- PetPolicy, InsuranceRequirement
- **RegulationSource** — maps to `regulation_sources`
- **AppSetting** — maps to `app_settings`
- ComplianceResult, ComplianceIssue (core/compliance/checker.py)
- UpdateResult (core/regulations/update_checker.py)
- SearchResult (core/rag/vector_store.py)