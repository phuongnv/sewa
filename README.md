## Streamlit + PostgreSQL Starter (Dev Container + Cloud)

This project scaffolds a Streamlit app with PostgreSQL. It includes a Dev Container setup for local development and is ready to deploy to Streamlit Community Cloud.

### Prerequisites
- Docker + Docker Compose (for Dev Container)
- VS Code + Dev Containers extension (or GitHub Codespaces)
- GitHub account (for Streamlit Community Cloud)

### Run locally in Dev Container
1. Open this folder in VS Code.
2. Use the command palette: “Dev Containers: Reopen in Container”.
3. Post-create will install Python deps in `.venv`.
4. Start the app:
   ```bash
   . .venv/bin/activate
   streamlit run app.py --server.address=0.0.0.0 --server.port=8501
   ```
5. Visit http://localhost:8501

The container includes a `postgres:16` service exposed on `localhost:5432`. The app reads DB config from environment variables (already set in the Dev Container):
```
POSTGRES_HOST=db
POSTGRES_PORT=5432
POSTGRES_DB=appdb
POSTGRES_USER=appuser
POSTGRES_PASSWORD=apppassword
```

### Deploy to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and create a new app pointing to your repo (`app.py` is the entry).
3. In the app’s “Secrets” settings, add a TOML block like (Neon example):
   ```toml
   [postgres]
   host = "ep-xxx.aws.neon.tech"  # from Neon dashboard
   port = 5432
   dbname = "YOUR_DB_NAME"
   user = "YOUR_DB_USER"
   password = "YOUR_DB_PASSWORD"
   sslmode = "require"           # Neon requires TLS
   ```
4. Deploy. Streamlit Cloud will install from `requirements.txt` and inject `st.secrets`.

### Files
- `app.py` – Streamlit app; auto-creates a `notes` table and lets you insert/list notes.
- `requirements.txt` – Python dependencies.
- `.streamlit/secrets.toml.example` – Example for Streamlit Cloud secrets.
- `.devcontainer/devcontainer.json` & `.devcontainer/docker-compose.yml` – Dev Container + Postgres.
- `.gitignore` – Common ignores.

### Notes
- For production databases, use managed Postgres (e.g., RDS, Neon, Supabase, etc.) and configure credentials via Streamlit secrets.
- Locally, data persists in the Docker volume `pgdata`.

#### Using Neon (managed Postgres)
- Create a database and user in Neon, then copy the connection parameters.
- This app automatically URL-encodes credentials and sets `sslmode=require` for Neon hosts (or when specified in secrets/env).
- If you want to use Neon during local dev instead of the Docker Postgres, set these env vars in your shell or Dev Container settings:
  ```bash
  export POSTGRES_HOST=ep-xxx.aws.neon.tech
  export POSTGRES_PORT=5432
  export POSTGRES_DB=YOUR_DB_NAME
  export POSTGRES_USER=YOUR_DB_USER
  export POSTGRES_PASSWORD=YOUR_DB_PASSWORD
  export POSTGRES_SSLMODE=require
  ```

### Streamlit
Learn more about Streamlit, deployment options, and components on the official site: [streamlit.io](https://streamlit.io/).


docker compose build
docker compose up

### rebuild after the code change
docker compose up --build -d


### Build devcontainer
devcontainer up --workspace-folder . --remove-existing-container