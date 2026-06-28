# Deployment

Status: deployed behind the main portfolio gateway.

## Public URLs

| Purpose | URL |
| --- | --- |
| Interactive Flask demo | `https://neojustin.dothost.net/projects/information-retrieval/` |
| Live API stats | `https://neojustin.dothost.net/projects/information-retrieval/api/stats` |
| GitHub Pages portfolio case study | `https://justin21523.github.io/zh-TW/projects/information-retrieval/` |
| Portfolio media root | `https://justin21523.github.io/portfolio/projects/information-retrieval/` |

The old `/p/information-retrieval/` route is not the CNIRS app route. The
portfolio nginx gateway proxies CNIRS at `/projects/information-retrieval/`.

## Runtime

- Docker Compose service: `information-retrieval`
- Server checkout path: `/home/neojustin/justin-portfolio/projects/information-retrieval`
- Main portfolio gateway config: `/home/neojustin/justin-portfolio/docker/nginx.conf`
- Demo-safe profile: `IR_ENABLE_HEAVY_MODELS=false`, `IR_TOKENIZER_ENGINE=jieba`

## Update After Code Changes

```bash
cd /home/neojustin/justin-portfolio
git pull
docker-compose up -d --build information-retrieval
docker-compose up -d --build web
```

## Verification

```bash
curl -I https://neojustin.dothost.net/projects/information-retrieval/
curl -I https://neojustin.dothost.net/projects/information-retrieval/static/css/style.css
curl -s https://neojustin.dothost.net/projects/information-retrieval/api/stats | python -m json.tool | head
python scripts/smoke_demo.py --base-url https://neojustin.dothost.net/projects/information-retrieval
```

Reference workflow:

- SSH connection note: `~/SSH_LIVE_DOTHOST_NET.local.md`
- Portfolio update runbook: `/home/justin/web-projects/justin-portfolio/docs/deployment/update-workflow.md`
