import { useEffect } from "react"
import { useSearchParams } from "react-router-dom"

const FOCUS_PARAM = "focus"

interface UseFocusDeepLinkOptions {
  knownIds: ReadonlySet<string>
  selectedId: string | null
  onApply: (id: string) => void
}

/**
 * Two-way binding between `?focus=<neuronId>` and the current selection.
 * - On mount / URL change: if `?focus=` matches a known node, select it.
 * - On selection change: mirror to URL via `replaceState` (no history spam).
 */
export function useFocusDeepLink({
  knownIds,
  selectedId,
  onApply,
}: UseFocusDeepLinkOptions): void {
  const [params, setParams] = useSearchParams()
  const urlFocus = params.get(FOCUS_PARAM)

  // URL → state: apply `?focus=` once per URL change, only if the id exists.
  // If the id is invalid AND the layout is loaded (knownIds non-empty), strip
  // the stale param so the URL stays clean and browser history doesn't carry
  // a dead focus across sessions.
  useEffect(() => {
    if (!urlFocus) return
    if (urlFocus === selectedId) return
    if (knownIds.has(urlFocus)) {
      onApply(urlFocus)
      return
    }
    if (knownIds.size === 0) return
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        next.delete(FOCUS_PARAM)
        return next
      },
      { replace: true },
    )
    // Intentionally depend only on urlFocus + the known set; selectedId echo
    // is handled by the write-back effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlFocus, knownIds])

  // State → URL: mirror current selection via functional updater so we don't
  // have to close over `params` (stale on browser back/forward nav).
  useEffect(() => {
    if (selectedId === urlFocus) return
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (selectedId) {
          next.set(FOCUS_PARAM, selectedId)
        } else {
          next.delete(FOCUS_PARAM)
        }
        return next
      },
      { replace: true },
    )
  }, [selectedId, urlFocus, setParams])
}
