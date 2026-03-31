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
    await page.goto("/health")
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

    const htmlBefore = await page.locator("html").getAttribute("class")
    await themeBtn.click()
    const htmlAfter = await page.locator("html").getAttribute("class")
    // Class should change (dark ↔ light ↔ system)
    expect(htmlAfter).not.toBe(htmlBefore)
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
