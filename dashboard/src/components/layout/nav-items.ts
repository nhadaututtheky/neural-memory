import {
  Brain,
  ChartLine,
  Cloud,
  Gauge,
  Gear,
  Graph,
  HardDrive,
  Lightbulb,
  ShareNetwork,
  Sparkle,
  SquaresFour,
  Storefront,
} from "@phosphor-icons/react"
import type { Icon } from "@phosphor-icons/react"

/**
 * The navigation tree — shared by the sidebar and the command palette.
 *
 * Both used to keep their own copy of this list, so an entry added to one
 * silently went missing from the other. `/living-brain` was the proof: it
 * existed as a route and in the palette, but never in the sidebar, leaving the
 * Pro flagship reachable only through a 2D/3D toggle buried in /graph.
 */

export type NavGroup = "core" | "visualize" | "data" | "analyze" | "settings"

export interface NavItem {
  to: string
  icon: Icon
  labelKey: string
  group: NavGroup
  /**
   * True only when the ENTIRE page is behind the license gate, so a free user
   * gets nothing from opening it. Pages where merely a card is gated
   * (Storage's migration, Settings' embedding config, Sync's delta mode) are
   * deliberately false — badging those would wrongly read as "locked".
   */
  pro?: boolean
}

export const NAV_GROUP_ORDER: readonly NavGroup[] = [
  "core",
  "visualize",
  "data",
  "analyze",
  "settings",
]

export const NAV_GROUP_LABEL_KEYS: Record<NavGroup, string> = {
  core: "nav.groups.core",
  visualize: "nav.groups.visualize",
  data: "nav.groups.data",
  analyze: "nav.groups.analyze",
  settings: "nav.groups.settings",
}

export const navItems: NavItem[] = [
  { to: "/", icon: SquaresFour, labelKey: "nav.overview", group: "core" },
  { to: "/insights", icon: Lightbulb, labelKey: "nav.insights", group: "core" },

  { to: "/graph", icon: Graph, labelKey: "nav.graph", group: "visualize" },
  { to: "/diagrams", icon: ShareNetwork, labelKey: "nav.mindmap", group: "visualize" },
  {
    to: "/living-brain",
    icon: Brain,
    labelKey: "nav.livingBrain",
    group: "visualize",
    pro: true,
  },

  { to: "/sync", icon: Cloud, labelKey: "nav.sync", group: "data" },
  { to: "/store", icon: Storefront, labelKey: "nav.store", group: "data" },
  { to: "/storage", icon: HardDrive, labelKey: "nav.storage", group: "data" },
  { to: "/tier-analytics", icon: Gauge, labelKey: "nav.tierAnalytics", group: "data" },

  { to: "/visualize", icon: ChartLine, labelKey: "nav.visualize", group: "analyze", pro: true },
  { to: "/oracle", icon: Sparkle, labelKey: "nav.oracle", group: "analyze" },

  { to: "/settings", icon: Gear, labelKey: "nav.settings", group: "settings" },
]

/** Items of one group, in declaration order. */
export function navItemsByGroup(group: NavGroup): NavItem[] {
  return navItems.filter((item) => item.group === group)
}
