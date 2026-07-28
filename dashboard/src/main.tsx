import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Toaster } from "sonner"
import App from "./App"
import "./i18n"
import "./index.css"

/**
 * Router basename, derived from where the bundle is actually mounted.
 *
 * FastAPI serves the built dashboard under /ui (and /dashboard), but the Vite
 * dev server serves it at the root. The previous logic defaulted to "/ui"
 * whenever the path was not /dashboard, so `npm run dev` rendered a blank page:
 * basename "/ui" never matched location "/". Falling back to "" fixes dev while
 * leaving both mounted paths behaving exactly as before.
 */
function routerBasename(): string {
  const path = window.location.pathname
  if (path === "/dashboard" || path.startsWith("/dashboard/")) return "/dashboard"
  if (path === "/ui" || path.startsWith("/ui/")) return "/ui"
  return ""
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename={routerBasename()}>
        <App />
        <Toaster
          position="bottom-right"
          toastOptions={{
            className: "bg-card text-card-foreground border-border",
          }}
        />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)
