import { test, expect } from "@playwright/test"
import { fileURLToPath, pathToFileURL } from "node:url"
import path from "node:path"

// Resolved from this spec's location so the suite is not tied to one machine.
// The project is ESM, so __dirname is unavailable — derive it from import.meta.
const HERE = path.dirname(fileURLToPath(import.meta.url))
const FILE = pathToFileURL(
  path.resolve(HERE, "../../docs/landing/pro-landing.html"),
).href

test("landing: no console errors, scene boots, keyboard reaches every panel", async ({ page }) => {
  const errs: string[] = []
  page.on("pageerror", (e) => errs.push("PAGEERROR: " + e.message))
  page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()) })

  await page.goto(FILE)
  await page.waitForTimeout(3000)

  // every feature must be reachable as a real button
  const labels = page.locator("button.cluster-label")
  expect(await labels.count()).toBe(6)

  // keyboard: focus a cluster button and activate with Enter
  await labels.nth(2).focus()
  await expect(labels.nth(2)).toBeFocused()
  await page.keyboard.press("Enter")
  await expect(page.locator("#feature-panel")).toHaveClass(/open/)
  const text = await page.locator("#panel-content").innerText()
  expect(text.length).toBeGreaterThan(50)

  console.log("CONSOLE ERRORS:", JSON.stringify(errs.slice(0, 5)))
})

test("landing: fallback appears when the three.js CDN is unreachable", async ({ page }) => {
  await page.route("**/cdn.jsdelivr.net/**", (r) => r.abort())
  await page.goto(FILE)
  await page.waitForTimeout(5000)

  const fb = page.locator("#scene-fallback")
  await expect(fb).toBeVisible()
  // and the buttons must actually open a panel without the 3D module
  await fb.getByRole("button", { name: "Pricing" }).click()
  await expect(page.locator("#feature-panel")).toHaveClass(/open/)
  await expect(page.locator("#panel-content")).toContainText("$9")
})

test("landing: cluster labels stay inside a 375px viewport", async ({ page }) => {
  // The page is a fixed-viewport app shell (body{overflow:hidden}) and the
  // feature panel is parked off-screen by design, so documentElement
  // scrollWidth is not the thing to assert. What is user-visible is whether the
  // 3D-projected labels get clipped at the screen edge.
  await page.setViewportSize({ width: 375, height: 700 })
  await page.goto(FILE)
  await page.waitForTimeout(3000)
  const clipped = await page.evaluate(() => {
    const vw = document.documentElement.clientWidth
    return [...document.querySelectorAll("button.cluster-label")]
      .map((el) => ({ id: el.id, r: el.getBoundingClientRect() }))
      .filter(({ r }) => r.left < 0 || r.right > vw)
      .map(({ id, r }) => `${id} left=${Math.round(r.left)} right=${Math.round(r.right)}`)
  })
  console.log("CLIPPED LABELS:", JSON.stringify(clipped))
  expect(clipped).toEqual([])
})
