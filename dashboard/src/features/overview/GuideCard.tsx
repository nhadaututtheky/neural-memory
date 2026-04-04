import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useStats } from "@/api/hooks/useDashboard"
import { BookOpen, X, ArrowSquareOut } from "@phosphor-icons/react"

const GUIDE_URL =
  "https://nhadaututtheky.github.io/neural-memory/guides/quickstart-guide/"
const STORAGE_KEY = "nmem-guide-card-dismissed"
const NEW_USER_THRESHOLD = 50

export default function GuideCard() {
  const { data: stats, isLoading } = useStats()
  const [dismissed, setDismissed] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) === "1"
    } catch {
      return false
    }
  })

  // Hide while loading to prevent flash for returning users
  if (isLoading) return null

  // Only show for new users (<50 neurons) who haven't dismissed
  const neuronCount = stats?.total_neurons ?? 0
  if (dismissed || neuronCount >= NEW_USER_THRESHOLD) {
    return null
  }

  const handleDismiss = () => {
    localStorage.setItem(STORAGE_KEY, "1")
    setDismissed(true)
  }

  return (
    <Card className="relative overflow-hidden border-primary/30 bg-gradient-to-r from-primary/5 to-primary/10">
      <CardContent className="flex items-center gap-4 p-4 sm:p-5">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/15">
          <BookOpen className="size-5 text-primary" aria-hidden="true" />
        </div>

        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold">Quickstart Guide</p>
          <p className="text-xs text-muted-foreground">
            Learn setup, recall, cognitive tools &amp; more
          </p>
        </div>

        <Button
          variant="outline"
          size="sm"
          className="shrink-0 gap-1.5"
          asChild
        >
          <a href={GUIDE_URL} target="_blank" rel="noopener noreferrer">
            Open Guide
            <ArrowSquareOut className="size-3.5" aria-hidden="true" />
          </a>
        </Button>

        <Button
          variant="ghost"
          size="icon"
          className="absolute right-2 top-2 size-7 text-muted-foreground hover:text-foreground"
          onClick={handleDismiss}
          aria-label="Dismiss guide card"
        >
          <X className="size-3.5" aria-hidden="true" />
        </Button>
      </CardContent>
    </Card>
  )
}
