/**
 * VS Code integration test launcher.
 *
 * Downloads VS Code, starts an Extension Development Host,
 * and runs the test suite inside it.
 */

import * as path from "path";
import { runTests } from "@vscode/test-electron";

async function main(): Promise<void> {
  const extensionDevelopmentPath = path.resolve(__dirname, "../../");
  const extensionTestsPath = path.resolve(__dirname, "./suite/index");

  await runTests({
    extensionDevelopmentPath,
    extensionTestsPath,
    launchArgs: [
      "--disable-extensions",
      "--disable-gpu",
    ],
  });
}

main().catch((err) => {
  console.error("Failed to run tests:", err);
  process.exit(1);
});
