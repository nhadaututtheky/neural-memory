import { test, expect } from "@playwright/test"

test.describe("Dashboard Smoke Tests", () => {
  test("loads the overview page", async ({ page }) => {
    await page.goto("/")
    await expect(page.locator("body")).toBeVisible()
    // App shell: sidebar nav should render
    await expect(page.getByRole("navigation")).toBeVisible()
  })

  test("sidebar navigation links are present", async ({ page }) => {
    await page.goto("/")
    const nav = page.getByRole("navigation")
    await expect(nav).toBeVisible()
    // At minimum, overview and health links should exist
    expect(await nav.getByRole("link").count()).toBeGreaterThan(0)
  })

  test("health page loads", async ({ page }) => {
    // Canonical route. The legacy "/health" alias cannot be hard-loaded in dev:
    // vite.config.ts proxies /health to the API on :8000, so the request never
    // reaches the SPA. In production the dashboard sits under a basename, so
    // the real path is /ui/health and there is no collision.
    await page.goto("/insights?tab=health")
    await expect(page.locator("body")).toBeVisible()
    await expect(page.getByRole("navigation")).toBeVisible()
  })

  test("settings page loads", async ({ page }) => {
    await page.goto("/settings")
    await expect(page.locator("body")).toBeVisible()
    await expect(page.getByRole("navigation")).toBeVisible()
  })

  test("oracle page loads", async ({ page }) => {
    await page.goto("/oracle")
    await expect(page.locator("body")).toBeVisible()
    // Oracle should render mode selector or heading
    await expect(page.getByRole("navigation")).toBeVisible()
  })

  test("no console errors on overview page", async ({ page }) => {
    const errors: string[] = []
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text())
      }
    })

    await page.goto("/")
    await page.waitForLoadState("networkidle")

    // Filter out expected errors (API calls that fail without backend)
    const unexpected = errors.filter(
      (e) => !e.includes("fetch") && !e.includes("ERR_CONNECTION") && !e.includes("net::"),
    )
    expect(unexpected).toHaveLength(0)
  })

  test("theme toggle works", async ({ page }) => {
    await page.goto("/")
    const themeBtn = page.getByTestId("theme-toggle")
    await expect(themeBtn).toBeVisible()

    // The cycle is light -> dark -> system and starts at "system". Asserting
    // that ONE click flips the `dark` class is wrong: with a light OS,
    // system -> light leaves the class absent both before and after. Assert on
    // the persisted choice, then walk the cycle far enough to prove the class
    // really does get applied.
    const storedBefore = await page.evaluate(() => localStorage.getItem("nm-theme"))
    await themeBtn.click()
    const storedAfter = await page.evaluate(() => localStorage.getItem("nm-theme"))
    expect(storedAfter).not.toBe(storedBefore)

    let sawDark = await page.locator("html.dark").count()
    for (let i = 0; i < 3 && sawDark === 0; i++) {
      await themeBtn.click()
      sawDark = await page.locator("html.dark").count()
    }
    expect(sawDark).toBeGreaterThan(0)
  })

  test("Phosphor icons render (no broken SVGs)", async ({ page }) => {
    await page.goto("/")
    // Phosphor icons render as SVG elements
    const svgs = page.locator("svg")
    const count = await svgs.count()
    // Sidebar alone has 5+ nav icons
    expect(count).toBeGreaterThanOrEqual(5)

    // Verify first few SVGs have valid dimensions (not 0x0)
    for (let i = 0; i < Math.min(3, count); i++) {
      const box = await svgs.nth(i).boundingBox()
      if (box) {
        expect(box.width).toBeGreaterThan(0)
        expect(box.height).toBeGreaterThan(0)
      }
    }
  })
})
