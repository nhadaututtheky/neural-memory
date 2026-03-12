// Daily reading interpretations — {past}, {present}, {future}, {pastContent}, {presentContent}, {futureContent}
export const DAILY_INTERPRETATIONS = [
  "Your past is marked by {past} — {pastContent}. Today, {present} guides your path — {presentContent}. Tomorrow, {future} awaits — {futureContent}.",
  "{past} shaped what came before: {pastContent}. Now {present} holds the key — {presentContent}. The future whispers of {future}: {futureContent}.",
  "From the shadow of {past} ({pastContent}), through the lens of {present} ({presentContent}), your path leads to {future} — {futureContent}.",
  "The memory of {past} echoes: {pastContent}. {present} illuminates the now: {presentContent}. {future} beckons ahead: {futureContent}.",
  "Once, {past} taught you: {pastContent}. Today, {present} reveals: {presentContent}. Soon, {future} will show: {futureContent}.",
  "{past} planted seeds — {pastContent}. {present} tends the garden — {presentContent}. {future} promises the harvest — {futureContent}.",
] as const

// What-if scenario templates — {cardA}, {cardB}, {suitA}, {suitB}, {wildcard}, {wildcardSuit}
export const WHAT_IF_TEMPLATES = [
  "What if {suitA} and {suitB} collided? Imagine: \"{cardA}\" meets \"{cardB}\" — and {wildcardSuit} throws in a twist: \"{wildcard}\"",
  "In another timeline, {suitA} chose differently: \"{cardA}\". Meanwhile {suitB} discovered: \"{cardB}\". The catalyst? {wildcardSuit}: \"{wildcard}\"",
  "Picture this: {suitA}'s memory (\"{cardA}\") suddenly merges with {suitB}'s truth (\"{cardB}\"). {wildcardSuit} watches from the shadows: \"{wildcard}\"",
  "Two paths diverged — {suitA} whispered \"{cardA}\" while {suitB} insisted \"{cardB}\". {wildcardSuit} appeared: \"{wildcard}\"",
  "What if you had followed {suitA} (\"{cardA}\") instead of {suitB} (\"{cardB}\")? {wildcardSuit} holds the answer: \"{wildcard}\"",
  "Rewind. {suitA} says: \"{cardA}\". Fast forward. {suitB} replies: \"{cardB}\". Plot twist by {wildcardSuit}: \"{wildcard}\"",
] as const

// Matchup comparison prompts — {suitA}, {suitB}, {contentA}, {contentB}
export const MATCHUP_PROMPTS = [
  "Which memory is stronger? {suitA}: \"{contentA}\" vs {suitB}: \"{contentB}\"",
  "{suitA} challenges {suitB}! \"{contentA}\" faces off against \"{contentB}\" — who wins?",
  "Battle of memories: {suitA} brings \"{contentA}\", {suitB} counters with \"{contentB}\"",
  "Your brain asks: is \"{contentA}\" ({suitA}) more important than \"{contentB}\" ({suitB})?",
  "Memory arena: {suitA} (\"{contentA}\") vs {suitB} (\"{contentB}\") — choose wisely!",
] as const
