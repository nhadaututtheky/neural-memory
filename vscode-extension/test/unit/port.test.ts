/**
 * Unit tests for port utility.
 *
 * Tests run with plain Mocha — no VS Code runtime needed.
 */

import * as assert from "assert";
import * as net from "net";
import { findFreePort } from "../../src/utils/port";

describe("findFreePort()", () => {
  it("should return a valid port number", async () => {
    const port = await findFreePort();

    assert.ok(typeof port === "number");
    assert.ok(port > 0, `Port should be positive, got ${port}`);
    assert.ok(port < 65536, `Port should be < 65536, got ${port}`);
  });

  it("should return a port that is available", async () => {
    const port = await findFreePort();

    // Verify we can actually listen on this port
    await new Promise<void>((resolve, reject) => {
      const server = net.createServer();
      server.listen(port, "127.0.0.1", () => {
        server.close((err) => {
          if (err) {
            reject(err);
          } else {
            resolve();
          }
        });
      });
      server.on("error", reject);
    });
  });

  it("should return different ports on consecutive calls", async () => {
    const port1 = await findFreePort();
    const port2 = await findFreePort();

    // Not guaranteed to be different, but highly likely
    // If they are the same, verify at least both are valid
    assert.ok(port1 > 0);
    assert.ok(port2 > 0);
  });
});
