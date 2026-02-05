/**
 * Build unit test files with esbuild.
 *
 * Bundles each .test.ts into out/test/unit/*.test.js with all
 * dependencies resolved (including src/ imports).
 * External: vscode (not needed for unit tests).
 */

const esbuild = require("esbuild");
const path = require("path");
const glob = require("glob");

async function main() {
  const testFiles = glob.sync("test/unit/**/*.test.ts", {
    cwd: path.resolve(__dirname, "../.."),
  });

  if (testFiles.length === 0) {
    console.log("No test files found.");
    return;
  }

  await esbuild.build({
    entryPoints: testFiles.map((f) =>
      path.resolve(__dirname, "../..", f),
    ),
    bundle: true,
    format: "cjs",
    platform: "node",
    outdir: path.resolve(__dirname, "../../out/test/unit"),
    external: ["vscode", "mocha"],
    sourcemap: true,
    logLevel: "info",
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
