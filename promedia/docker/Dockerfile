# ProMedia scaffold — hub image (build context = monorepo root `promedia/`)
# docker compose -f docker/compose.yml up --build
FROM node:22-bookworm-slim

RUN corepack enable && corepack prepare pnpm@10.33.4 --activate

WORKDIR /app

COPY package.json pnpm-workspace.yaml tsconfig.base.json .npmrc ./
COPY packages ./packages
COPY apps ./apps

RUN pnpm install --frozen-lockfile=false \
  && pnpm build

ENV NODE_ENV=production
ENV PORT=8787

EXPOSE 8787

CMD ["pnpm", "--filter", "@promedia/hub", "start"]
