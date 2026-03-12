import type { OracleCard, DailyReading } from "../engine/types"

const WIDTH = 800
const HEIGHT = 420
const CARD_W = 180
const CARD_H = 260
const CARD_GAP = 24

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + r)
  ctx.lineTo(x + w, y + h - r)
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
  ctx.lineTo(x + r, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - r)
  ctx.lineTo(x, y + r)
  ctx.quadraticCurveTo(x, y, x + r, y)
  ctx.closePath()
}

function drawCard(
  ctx: CanvasRenderingContext2D,
  card: OracleCard,
  x: number,
  y: number,
  label: string,
) {
  // Card background
  drawRoundedRect(ctx, x, y, CARD_W, CARD_H, 12)
  ctx.fillStyle = "#1a1814"
  ctx.fill()
  ctx.strokeStyle = card.suit.color + "40"
  ctx.lineWidth = 1.5
  ctx.stroke()

  // Label above card
  ctx.fillStyle = "#a0a0a0"
  ctx.font = "bold 10px system-ui, sans-serif"
  ctx.textAlign = "center"
  ctx.fillText(label.toUpperCase(), x + CARD_W / 2, y - 8)

  // Suit symbol
  ctx.fillStyle = card.suit.color
  ctx.font = "36px system-ui, sans-serif"
  ctx.fillText(card.suit.symbol, x + CARD_W / 2, y + 50)

  // Suit name
  ctx.fillStyle = card.suit.color
  ctx.font = "bold 11px system-ui, sans-serif"
  ctx.fillText(card.title, x + CARD_W / 2, y + 72)

  // Divider
  ctx.strokeStyle = card.suit.color + "30"
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(x + 30, y + 82)
  ctx.lineTo(x + CARD_W - 30, y + 82)
  ctx.stroke()

  // Content (wrap text)
  ctx.fillStyle = "#e0e0e0"
  ctx.font = "12px system-ui, sans-serif"
  const maxWidth = CARD_W - 28
  const words = card.content.split(" ")
  let line = ""
  let lineY = y + 100
  const maxLines = 5

  let lineCount = 0
  for (const word of words) {
    const test = line + (line ? " " : "") + word
    if (ctx.measureText(test).width > maxWidth && line) {
      ctx.fillText(line, x + CARD_W / 2, lineY)
      line = word
      lineY += 16
      lineCount++
      if (lineCount >= maxLines) {
        ctx.fillText(line + "...", x + CARD_W / 2, lineY)
        break
      }
    } else {
      line = test
    }
  }
  if (lineCount < maxLines && line) {
    ctx.fillText(line, x + CARD_W / 2, lineY)
  }

  // Stats row
  ctx.fillStyle = "#808080"
  ctx.font = "10px system-ui, sans-serif"
  ctx.textAlign = "left"
  ctx.fillText(`⚡${card.activation}`, x + 14, y + CARD_H - 16)
  ctx.textAlign = "center"
  ctx.fillText(`🔗${card.connectionCount}`, x + CARD_W / 2, y + CARD_H - 16)
  ctx.textAlign = "right"
  ctx.fillText(`📅${card.age}`, x + CARD_W - 14, y + CARD_H - 16)

  // Suit badge
  ctx.textAlign = "center"
  ctx.fillStyle = card.suit.color + "80"
  ctx.font = "bold 9px system-ui, sans-serif"
  ctx.fillText(
    `${card.suit.symbol} ${card.suitKey}`,
    x + CARD_W / 2,
    y + CARD_H - 4,
  )
}

export function generateShareImage(reading: DailyReading): Promise<Blob> {
  const canvas = document.createElement("canvas")
  canvas.width = WIDTH
  canvas.height = HEIGHT
  const ctx = canvas.getContext("2d")!

  // Background
  const bg = ctx.createLinearGradient(0, 0, WIDTH, HEIGHT)
  bg.addColorStop(0, "#0c0b09")
  bg.addColorStop(1, "#16140f")
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, WIDTH, HEIGHT)

  // Title
  ctx.fillStyle = "#ffffff"
  ctx.font = "bold 22px system-ui, sans-serif"
  ctx.textAlign = "center"
  ctx.fillText("✦ Brain Oracle ✦", WIDTH / 2, 36)

  // Subtitle
  ctx.fillStyle = "#818cf8"
  ctx.font = "12px system-ui, sans-serif"
  ctx.fillText(
    `${reading.brainName} — ${reading.date}`,
    WIDTH / 2,
    54,
  )

  // Draw 3 cards
  const cards = [reading.past, reading.present, reading.future]
  const labels = ["Past", "Present", "Future"]
  const totalW = CARD_W * 3 + CARD_GAP * 2
  const startX = (WIDTH - totalW) / 2
  const cardY = 80

  cards.forEach((card, i) => {
    drawCard(ctx, card, startX + i * (CARD_W + CARD_GAP), cardY, labels[i])
  })

  // Footer
  ctx.fillStyle = "#606060"
  ctx.font = "10px system-ui, sans-serif"
  ctx.textAlign = "center"
  ctx.fillText("Neural Memory — neuralmemory.dev", WIDTH / 2, HEIGHT - 10)

  return new Promise((resolve) => {
    canvas.toBlob(
      (blob) => resolve(blob!),
      "image/png",
    )
  })
}

export async function copyShareImage(reading: DailyReading): Promise<boolean> {
  try {
    const blob = await generateShareImage(reading)
    await navigator.clipboard.write([
      new ClipboardItem({ "image/png": blob }),
    ])
    return true
  } catch {
    return false
  }
}

export async function downloadShareImage(reading: DailyReading): Promise<void> {
  const blob = await generateShareImage(reading)
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `oracle-${reading.brainName}-${reading.date}.png`
  a.click()
  URL.revokeObjectURL(url)
}
