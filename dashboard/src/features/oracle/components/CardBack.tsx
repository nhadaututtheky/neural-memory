interface CardBackProps {
  className?: string
}

export function CardBack({ className = "" }: CardBackProps) {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-white/10 bg-[#16140f] ${className}`}
      style={{ backfaceVisibility: "hidden" }}
    >
      {/* Mandala pattern */}
      <div className="absolute inset-0 opacity-30">
        <div
          className="absolute inset-0"
          style={{
            background:
              "repeating-conic-gradient(from 0deg at 50% 50%, #818cf8 0deg 30deg, transparent 30deg 60deg)",
            opacity: 0.15,
          }}
        />
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(circle at 50% 50%, transparent 30%, #818cf820 50%, transparent 70%)",
          }}
        />
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(circle at 50% 50%, #818cf815 0%, transparent 40%)",
          }}
        />
      </div>

      {/* Center symbol */}
      <div className="absolute inset-0 flex items-center justify-center" aria-hidden="true">
        <div className="relative">
          <div
            className="text-5xl font-bold opacity-20"
            style={{ color: "#818cf8" }}
          >
            ✦
          </div>
          <div className="absolute inset-0 flex items-center justify-center text-2xl font-bold text-white/40">
            ?
          </div>
        </div>
      </div>

      {/* Border glow */}
      <div className="absolute inset-0 rounded-2xl ring-1 ring-inset ring-white/5" />
    </div>
  )
}
