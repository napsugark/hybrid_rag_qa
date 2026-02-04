# Langfuse Self-Hosted Setup

This folder contains the Docker infrastructure for running Langfuse locally.

## Quick Start

```bash
# 1. Download the official docker-compose.yml
curl -o docker-compose.yml https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml

# 2. Start Langfuse (from this langfuse/ folder)
docker-compose up -d

# 3. Access UI
# Open http://localhost:3000 in your browser

# 4. Create account and get API keys
# - Sign up in the UI
# - Create a project
# - Go to Settings → API Keys
# - Copy the public and secret keys

# 5. Add keys to main .env file (in parent app_v2/ folder)
# Edit ../env and add:
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

## Management Commands

```bash
# Start Langfuse
docker-compose up -d

# View logs
docker-compose logs -f

# Stop Langfuse
docker-compose down

# Stop and remove all data (fresh start)
docker-compose down -v

# Check status
docker-compose ps
```

## Default Access

- **URL:** http://localhost:3000
- **PostgreSQL:** localhost:5432
- **Redis:** localhost:6379

## Data Persistence

Data is stored in Docker volumes:
- `postgres-data/` - Database files
- `redis-data/` - Cache files

These folders are git-ignored to keep your traces private.

## Troubleshooting

**Port conflicts:**
- If port 3000 is in use, edit `docker-compose.yml` and change the port mapping
- Change `3000:3000` to `3001:3000` (or any available port)

**Can't connect:**
- Ensure Docker is running
- Check logs: `docker-compose logs`
- Verify ports aren't blocked by firewall

**Reset everything:**
```bash
docker-compose down -v
rm -rf postgres-data redis-data
docker-compose up -d
```

## Security Notes

- This setup is for **local development only**
- Default credentials are in docker-compose.yml
- Don't expose port 3000 to the internet
- For production, use Langfuse Cloud or properly secured deployment
