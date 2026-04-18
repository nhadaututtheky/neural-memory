import { useEffect, useRef } from "react"
import type { BrainLayout } from "../engine/types"

interface UseKeyboardNavOptions {
  layout: BrainLayout
  selectedId: string | null
  hoveredId: string | null
  onSelect: (id: string | null) => void
}

/**
 * Arrow keys walk the graph along synapse edges (sorted by weight desc).
 * A one-slot history ref avoids ping-ponging back to the previous node.
 *
 * - Esc: clear selection
 * - Enter (when hovered + no selection): open hovered node
 * - ArrowRight: strongest-weight neighbor (skipping last-visited)
 * - ArrowLeft: weakest-weight neighbor (skipping last-visited)
 *
 * Inputs/textareas/contenteditable targets are ignored so search boxes work.
 * Selection/hover are read via refs so the keydown listener registers once
 * per layout change, not on every state transition.
 */
export function useKeyboardNav({
  layout,
  selectedId,
  hoveredId,
  onSelect,
}: UseKeyboardNavOptions): void {
  const prevRef = useRef<string | null>(null)
  const selectedRef = useRef(selectedId)
  const hoveredRef = useRef(hoveredId)
  const onSelectRef = useRef(onSelect)

  useEffect(() => {
    selectedRef.current = selectedId
  }, [selectedId])
  useEffect(() => {
    hoveredRef.current = hoveredId
  }, [hoveredId])
  useEffect(() => {
    onSelectRef.current = onSelect
  }, [onSelect])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      const tag = target?.tagName?.toLowerCase()
      if (tag === "input" || tag === "textarea" || target?.isContentEditable) {
        return
      }

      const selected = selectedRef.current
      const hovered = hoveredRef.current
      const apply = onSelectRef.current

      if (e.key === "Escape") {
        if (selected) {
          e.preventDefault()
          prevRef.current = null
          apply(null)
        }
        return
      }

      if (e.key === "Enter" && hovered && !selected) {
        e.preventDefault()
        apply(hovered)
        return
      }

      if (!selected) return
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return

      const neighbors = layout.neighbors.get(selected)
      if (!neighbors || neighbors.length === 0) return

      e.preventDefault()
      const prev = prevRef.current
      const filtered = prev ? neighbors.filter((n) => n !== prev) : [...neighbors]
      const pool = filtered.length > 0 ? filtered : [...neighbors]
      const next = e.key === "ArrowRight" ? pool[0] : pool[pool.length - 1]
      prevRef.current = selected
      apply(next)
    }

    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [layout])
}
