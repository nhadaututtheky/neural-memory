import { Outlet } from "react-router-dom"
import { Sidebar } from "./Sidebar"
import { TopBar } from "./TopBar"
import { useLayoutStore } from "@/stores/useLayoutStore"
import { cn } from "@/lib/utils"
import { UpgradeModal } from "@/components/common/UpgradeModal"

export function AppShell() {
  const sidebarOpen = useLayoutStore((s) => s.sidebarOpen)

  return (
    <div className="h-screen bg-background overflow-hidden">
      <Sidebar />
      <div
        className={cn(
          "flex flex-col h-full transition-all duration-[var(--transition-normal)]",
          sidebarOpen ? "ml-56" : "ml-16",
        )}
      >
        <TopBar />
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
      <UpgradeModal />
    </div>
  )
}
