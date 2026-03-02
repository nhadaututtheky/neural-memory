# NeuralMemory Dashboard

React 19 + TypeScript + Vite 7 dashboard for NeuralMemory.

## Tech Stack

- **Framework**: React 19 + TypeScript
- **Build**: Vite 7
- **Styling**: TailwindCSS 4 + shadcn/ui
- **State**: TanStack Query 5 + Zustand 5
- **Charts**: Recharts 3
- **Graph**: Sigma.js 3 + graphology (WebGL)
- **Animation**: Framer Motion 11
- **Icons**: Lucide React
- **Theme**: Warm cream light (dark mode secondary)

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Overview | `/dashboard` | KPI cards, brain list, file info |
| Health | `/dashboard/health` | Radar chart, warnings, recommendations |
| Graph | `/dashboard/graph` | Sigma.js neural graph explorer |
| Timeline | `/dashboard/timeline` | Date-filtered memory timeline |
| Evolution | `/dashboard/evolution` | Brain maturation & stage distribution |
| Diagrams | `/dashboard/diagrams` | Fiber subgraph viewer |
| Settings | `/dashboard/settings` | Brain config, Telegram backup, about |

## Development

```bash
# Install dependencies
npm install

# Start dev server (proxies API to localhost:8000)
npm run dev
# Opens at http://localhost:5174

# In another terminal, start the NeuralMemory server
nmem serve
```

## Production Build

```bash
npm run build
# Output: ../src/neural_memory/server/static/dist/
```

The built SPA is served by FastAPI at `/ui` and `/dashboard`.

## Project Structure

```
src/
├── main.tsx                  # Entry point
├── App.tsx                   # Router + QueryClient + Toaster
├── index.css                 # Warm cream palette + dark mode
├── lib/utils.ts              # cn() utility
├── components/
│   ├── ui/                   # shadcn/ui primitives
│   └── layout/               # AppShell, Sidebar, TopBar
├── api/
│   ├── client.ts             # Fetch wrapper
│   ├── types.ts              # TypeScript interfaces
│   └── hooks/                # TanStack Query hooks
├── features/
│   ├── overview/             # Overview page + KPI cards
│   ├── health/               # Health radar + warnings
│   ├── graph/                # Sigma.js graph explorer
│   ├── timeline/             # Timeline page
│   ├── evolution/            # Evolution metrics
│   ├── diagrams/             # Fiber diagrams
│   └── settings/             # Settings + Telegram backup
└── stores/                   # Zustand stores
```
