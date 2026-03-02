import { Outlet } from "react-router-dom"
import { Sidebar } from "./Sidebar"
import { TopBar } from "./TopBar"
import { useLayoutStore } from "@/stores/useLayoutStore"
import { cn } from "@/lib/utils"

export function AppShell() {
  const sidebarOpen = useLayoutStore((s) => s.sidebarOpen)

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <div
        className={cn(
          "flex flex-col transition-all duration-[var(--transition-normal)]",
          sidebarOpen ? "ml-56" : "ml-16",
        )}
      >
        <TopBar />
        <main className="flex-1">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
