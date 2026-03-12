import { useState, useEffect, useRef } from "react"
import type { OracleCard } from "../engine/types"
import { CardFace } from "./CardFace"
import { CardBack } from "./CardBack"

interface FlipCardProps {
  card: OracleCard
  autoFlipDelay?: number
  onFlip?: () => void
  className?: string
}

export function FlipCard({ card, autoFlipDelay, onFlip, className = "" }: FlipCardProps) {
  const [isFlipped, setIsFlipped] = useState(false)
  const onFlipRef = useRef(onFlip)
  onFlipRef.current = onFlip

  useEffect(() => {
    if (autoFlipDelay !== undefined && autoFlipDelay >= 0) {
      const timer = setTimeout(() => {
        setIsFlipped(true)
        onFlipRef.current?.()
      }, autoFlipDelay)
      return () => clearTimeout(timer)
    }
  }, [autoFlipDelay])

  const handleFlip = () => {
    const next = !isFlipped
    setIsFlipped(next)
    if (next) onFlipRef.current?.()
  }

  return (
    <div
      className={`cursor-pointer ${className}`}
      style={{ perspective: "1000px" }}
      onClick={handleFlip}
      role="button"
      tabIndex={0}
      aria-label={isFlipped ? `Card: ${card.title} — ${card.content}` : "Tap to reveal card"}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          handleFlip()
        }
      }}
    >
      <div
        className="relative h-full w-full"
        style={{
          transformStyle: "preserve-3d",
          transform: isFlipped ? "rotateY(180deg)" : "rotateY(0deg)",
          transitionTimingFunction: "cubic-bezier(0.4, 0, 0.2, 1)",
          transitionDuration: "600ms",
        }}
      >
        {/* Back face (default visible) */}
        <div className="absolute inset-0" style={{ backfaceVisibility: "hidden" }}>
          <CardBack className="h-full w-full" />
        </div>

        {/* Front face (visible when flipped) */}
        <div
          className="absolute inset-0"
          style={{
            backfaceVisibility: "hidden",
            transform: "rotateY(180deg)",
          }}
        >
          <CardFace card={card} className="h-full w-full" />
        </div>
      </div>
    </div>
  )
}
