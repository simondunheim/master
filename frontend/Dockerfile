# ./frontend/Dockerfile
FROM node:20-slim

WORKDIR /app

# Optional build tooling; safe to keep
RUN apt-get update && apt-get install -y python3 make g++ git \
  && rm -rf /var/lib/apt/lists/*

# Copy manifests first to leverage cache
# If you have a package-lock.json, include it here as well
COPY package.json package-lock.json* ./

# ---- INSTALL DEPS (pick ONE of the following) ----
# If you have package-lock.json (recommended):
# RUN npm ci --no-audit --no-fund

# If you do NOT have a lockfile:
RUN npm ci --legacy-peer-deps --no-audit --no-fund
# --------------------------------------------------

# Copy the rest
COPY . .

# (Compose already mounts an anonymous volume to /app/node_modules,
# so you don't need a VOLUME line here.)

ENV NODE_ENV=development
ENV VITE_API_BASE_URL=http://localhost:8000

EXPOSE 5173

# Run local Vite from node_modules/.bin via npm script
CMD ["npm","run","dev","--","--host","0.0.0.0","--port","5173","--force"]
