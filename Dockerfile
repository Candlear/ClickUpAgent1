# Python 3.11; Node optional for CLICKUP_MCP_MODE=community_stdio (npx)
FROM python:3.11-slim-bookworm

ARG INSTALL_NODE=true

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && if [ "$INSTALL_NODE" = "true" ]; then \
        apt-get install -y --no-install-recommends curl \
        && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
        && apt-get install -y --no-install-recommends nodejs; \
    fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py mcp_helper.py mcp_session.py oauth_storage.py tool_policy.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Railway sets PORT; hosted-only images can use INSTALL_NODE=false at build time
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
