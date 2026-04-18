import { Button } from "@/components/ui/button"
import { useTranslation } from "react-i18next"

export type GraphMode = "2d" | "3d"

interface ModeToggleProps {
  mode: GraphMode
  onChange: (mode: GraphMode) => void
}

export function ModeToggle({ mode, onChange }: ModeToggleProps) {
  const { t } = useTranslation()
  return (
    <div
      role="group"
      aria-label={t("livingBrain.modeGroup")}
      className="inline-flex rounded-lg border border-border bg-muted/40 p-0.5"
    >
      <Button
        variant={mode === "2d" ? "default" : "ghost"}
        size="sm"
        onClick={() => onChange("2d")}
        aria-pressed={mode === "2d"}
      >
        {t("graph.view2D")}
      </Button>
      <Button
        variant={mode === "3d" ? "default" : "ghost"}
        size="sm"
        onClick={() => onChange("3d")}
        aria-pressed={mode === "3d"}
      >
        {t("graph.view3D")}
      </Button>
    </div>
  )
}
