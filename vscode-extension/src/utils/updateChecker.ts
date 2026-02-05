import * as vscode from "vscode";

const GITHUB_OWNER = "nhadaututtheky";
const GITHUB_REPO = "neural-memory";
const GITHUB_API_URL = `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/releases/latest`;
const CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000; // 24 hours
const STATE_KEY_LAST_CHECK = "neuralmemory.lastUpdateCheck";
const STATE_KEY_DISMISSED = "neuralmemory.dismissedVersion";

interface GitHubRelease {
  readonly tag_name: string;
  readonly html_url: string;
  readonly name: string;
  readonly body: string;
}

/**
 * Check for newer versions on GitHub Releases.
 * Runs at most once per 24h, non-blocking, does not interrupt startup.
 */
export async function checkForUpdates(
  context: vscode.ExtensionContext,
): Promise<void> {
  // Throttle: skip if checked within last 24h
  const lastCheck = context.globalState.get<number>(STATE_KEY_LAST_CHECK, 0);
  if (Date.now() - lastCheck < CHECK_INTERVAL_MS) {
    return;
  }

  try {
    const currentVersion = getLocalVersion();
    const release = await fetchLatestRelease();
    if (!release) {
      return;
    }

    await context.globalState.update(STATE_KEY_LAST_CHECK, Date.now());

    const remoteVersion = normalizeVersion(release.tag_name);
    if (!remoteVersion) {
      return;
    }

    // Already dismissed this version?
    const dismissed = context.globalState.get<string>(STATE_KEY_DISMISSED);
    if (dismissed === remoteVersion) {
      return;
    }

    // Compare versions
    if (!isNewer(remoteVersion, currentVersion)) {
      return;
    }

    // Show notification
    const releaseName = release.name || `v${remoteVersion}`;
    const action = await vscode.window.showInformationMessage(
      `NeuralMemory ${releaseName} is available (current: v${currentVersion}).`,
      "View Release",
      "Dismiss",
    );

    if (action === "View Release") {
      vscode.env.openExternal(vscode.Uri.parse(release.html_url));
    } else if (action === "Dismiss") {
      await context.globalState.update(STATE_KEY_DISMISSED, remoteVersion);
    }
  } catch {
    // Non-critical — silently ignore network/API errors
  }
}

function getLocalVersion(): string {
  const ext = vscode.extensions.getExtension("neuralmemory.neuralmemory");
  if (ext) {
    return ext.packageJSON.version as string;
  }
  // Fallback: read from package.json at compile time
  return "0.1.0";
}

async function fetchLatestRelease(): Promise<GitHubRelease | null> {
  try {
    const resp = await fetch(GITHUB_API_URL, {
      headers: {
        Accept: "application/vnd.github+json",
        "User-Agent": "neuralmemory-vscode",
      },
      signal: AbortSignal.timeout(5_000),
    });

    if (!resp.ok) {
      return null;
    }

    return (await resp.json()) as GitHubRelease;
  } catch {
    return null;
  }
}

/**
 * Strip leading 'v' and validate semver-like format.
 */
function normalizeVersion(tag: string): string | null {
  const cleaned = tag.replace(/^v/, "");
  if (/^\d+\.\d+\.\d+/.test(cleaned)) {
    return cleaned;
  }
  return null;
}

/**
 * Simple semver comparison: is `remote` newer than `local`?
 */
function isNewer(remote: string, local: string): boolean {
  const rParts = remote.split(".").map(Number);
  const lParts = local.split(".").map(Number);

  for (let i = 0; i < 3; i++) {
    const r = rParts[i] ?? 0;
    const l = lParts[i] ?? 0;
    if (r > l) {
      return true;
    }
    if (r < l) {
      return false;
    }
  }

  return false;
}
