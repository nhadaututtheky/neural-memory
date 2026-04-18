import { useEffect, useRef } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { useTranslation } from "react-i18next"
import { X } from "@phosphor-icons/react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { BrainLayout, PositionedNeuron } from "../engine/types"
import { useLivingBrainStore } from "../stores/livingBrainStore"

interface NodeDetailPanelProps {
  layout: BrainLayout
  selected: PositionedNeuron | null
  onClose: () => void
  onSelectNeighbor: (id: string) => void
}

export function NodeDetailPanel({
  layout,
  selected,
  onClose,
  onSelectNeighbor,
}: NodeDetailPanelProps) {
  const { t } = useTranslation()
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const neighbors = selected ? (layout.neighbors.get(selected.id) ?? []) : []
  const topNeighbors = neighbors.slice(0, 8)

  // Auto-focus the close button when a node is first opened so keyboard users
  // know the panel has focus. Does not refocus when swapping between nodes.
  // Returns focus to the previous activeElement on close.
  const prevFocusRef = useRef<HTMLElement | null>(null)
  useEffect(() => {
    if (!selected) return
    prevFocusRef.current = (document.activeElement as HTMLElement | null) ?? null
    closeButtonRef.current?.focus()
    return () => {
      prevFocusRef.current?.focus?.()
    }
  }, [selected?.id, selected])

  return (
    <AnimatePresence>
      {selected && (
        <motion.aside
          key={selected.id}
          initial={{ x: 360, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 360, opacity: 0 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          className="pointer-events-auto absolute right-3 top-3 bottom-3 w-[360px] max-w-[90vw] overflow-hidden rounded-lg border border-border/60 bg-card/95 shadow-lg backdrop-blur-md"
          role="dialog"
          aria-modal="true"
          aria-label={t("livingBrain.detailPanel")}
          onPointerEnter={() => useLivingBrainStore.getState().setHovered(null)}
        >
          <div className="flex items-start justify-between border-b border-border/50 p-3">
            <div className="flex items-center gap-2">
              <Badge
                variant="secondary"
                style={{ backgroundColor: `${selected.color}22`, color: selected.color }}
              >
                {selected.type}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {neighbors.length} {t("livingBrain.connections")}
              </span>
            </div>
            <Button
              ref={closeButtonRef}
              variant="ghost"
              size="icon"
              onClick={onClose}
              aria-label={t("common.close")}
              className="h-7 w-7"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex h-full flex-col gap-3 overflow-y-auto p-3 pb-16">
            <section>
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {t("livingBrain.content")}
              </h3>
              <p className="whitespace-pre-wrap text-sm leading-relaxed">
                {selected.content}
              </p>
            </section>

            <section>
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                ID
              </h3>
              <p className="break-all font-mono text-[11px] text-muted-foreground">
                {selected.id}
              </p>
            </section>

            {topNeighbors.length > 0 && (
              <section>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  {t("livingBrain.neighbors")}
                </h3>
                <ul className="space-y-1">
                  {topNeighbors.map((nid) => {
                    const idx = layout.indexById.get(nid)
                    if (idx === undefined) return null
                    const n = layout.neurons[idx]
                    return (
                      <li key={nid}>
                        <button
                          type="button"
                          onClick={() => onSelectNeighbor(nid)}
                          className="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs transition-colors hover:bg-accent"
                        >
                          <span
                            className="h-2 w-2 shrink-0 rounded-full"
                            style={{ backgroundColor: n.color }}
                            aria-hidden
                          />
                          <span className="truncate">{n.content}</span>
                        </button>
                      </li>
                    )
                  })}
                </ul>
              </section>
            )}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}
