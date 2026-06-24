# Deployment

Status: Deployed behind the portfolio gateway.

- URL: `https://neojustin.dothost.net/p/information-retrieval/`
- docker-compose service name: `information-retrieval`
- Server checkout path: `/home/neojustin/justin-portfolio/projects/information-retrieval`

## Update after code changes
```bash
cd /home/neojustin/justin-portfolio
docker-compose up -d --build information-retrieval
```

Reference workflow:
- SSH 連線資訊（本機私有）：`~/SSH_LIVE_DOTHOST_NET.local.md`
- `/home/justin/web-projects/justin-portfolio/docs/deployment/update-workflow.md`
