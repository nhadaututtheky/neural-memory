import { useTimeline } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"

export default function TimelinePage() {
  const { data: timeline, isLoading } = useTimeline(100, 0)

  return (
    <div className="space-y-6 p-6">
      <h1 className="font-display text-2xl font-bold">Timeline</h1>

      <Card>
        <CardHeader>
          <CardTitle>
            Memory Timeline
            {timeline && (
              <span className="ml-2 text-sm font-normal text-muted-foreground">
                {timeline.total.toLocaleString()} entries
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : timeline?.entries && timeline.entries.length > 0 ? (
            <div className="space-y-3">
              {timeline.entries.map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-start gap-3 rounded-lg border border-border/50 p-3 transition-colors hover:bg-accent/30"
                >
                  <Badge variant="outline" className="mt-0.5 shrink-0">
                    {entry.neuron_type}
                  </Badge>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm leading-relaxed line-clamp-2">
                      {entry.content}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {entry.created_at ? new Date(entry.created_at).toLocaleString() : "-"}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No timeline entries.</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
