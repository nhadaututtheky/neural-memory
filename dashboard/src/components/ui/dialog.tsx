import { useEffect, useRef } from "react"
import { cn } from "@/lib/utils"

interface DialogProps {
  open: boolean
  onClose: () => void
  children: React.ReactNode
  className?: string
}

export function Dialog({ open, onClose, children, className }: DialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    if (open && !dialog.open) dialog.showModal()
    if (!open && dialog.open) dialog.close()
  }, [open])

  if (!open) return null

  return (
    <dialog
      ref={dialogRef}
      className={cn(
        "fixed inset-0 z-50 m-auto max-w-2xl w-[90vw] rounded-xl border border-border bg-card p-0 shadow-lg backdrop:bg-black/50",
        className,
      )}
      onClose={onClose}
      onClick={(e) => {
        if (e.target === dialogRef.current) onClose()
      }}
    >
      <div className="max-h-[85vh] overflow-y-auto p-6">{children}</div>
    </dialog>
  )
}

interface DialogHeaderProps {
  children: React.ReactNode
  className?: string
}

export function DialogHeader({ children, className }: DialogHeaderProps) {
  return (
    <div className={cn("mb-4", className)}>
      {children}
    </div>
  )
}

interface DialogTitleProps {
  children: React.ReactNode
  className?: string
}

export function DialogTitle({ children, className }: DialogTitleProps) {
  return (
    <h2 className={cn("font-display text-lg font-bold text-card-foreground", className)}>
      {children}
    </h2>
  )
}
