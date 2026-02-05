const esbuild = require("esbuild");
const fs = require("fs");
const path = require("path");

const production = process.argv.includes("--production");
const watch = process.argv.includes("--watch");

/**
 * Copy vendored libraries needed by webviews into dist/lib/.
 * These can't be bundled by esbuild because webviews load them via script tags.
 */
function copyWebviewLibs() {
  const libDir = path.join(__dirname, "dist", "lib");
  fs.mkdirSync(libDir, { recursive: true });

  const libs = [
    {
      src: path.join(__dirname, "node_modules", "cytoscape", "dist", "cytoscape.min.js"),
      dest: path.join(libDir, "cytoscape.min.js"),
    },
  ];

  for (const { src, dest } of libs) {
    fs.copyFileSync(src, dest);
    console.log(`[copy] ${path.basename(src)} → dist/lib/`);
  }
}

async function main() {
  const ctx = await esbuild.context({
    entryPoints: ["src/extension.ts"],
    bundle: true,
    format: "cjs",
    minify: production,
    sourcemap: !production,
    sourcesContent: false,
    platform: "node",
    outfile: "dist/extension.js",
    external: ["vscode"],
    logLevel: "info",
    plugins: [esbuildProblemMatcherPlugin],
  });

  if (watch) {
    await ctx.watch();
    console.log("[watch] build started");
  } else {
    await ctx.rebuild();
    await ctx.dispose();
  }

  copyWebviewLibs();
}

/** @type {import('esbuild').Plugin} */
const esbuildProblemMatcherPlugin = {
  name: "esbuild-problem-matcher",
  setup(build) {
    build.onStart(() => {
      console.log("[watch] build started");
    });
    build.onEnd((result) => {
      for (const { text, location } of result.errors) {
        console.error(`> ${location.file}:${location.line}:${location.column}: error: ${text}`);
      }
      console.log("[watch] build finished");
    });
  },
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
