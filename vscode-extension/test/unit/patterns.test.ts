/**
 * Unit tests for symbol extraction and comment trigger patterns.
 *
 * Tests the regex patterns used by MemoryCodeLensProvider and
 * CommentTriggerWatcher. Patterns are duplicated here because
 * the source files import vscode and can't be loaded standalone.
 */

import * as assert from "assert";

// --- Symbol patterns (from MemoryCodeLensProvider.ts) ---

const SYMBOL_PATTERNS: readonly RegExp[] = [
  /^\s*(?:async\s+)?def\s+(\w+)\s*\(/,
  /^\s*class\s+(\w+)/,
  /^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)/,
  /^\s*(?:export\s+)?class\s+(\w+)/,
  /^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\(|function)/,
  /^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(/,
  /^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)/,
  /^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?(?:\w+\s+)+(\w+)\s*\(/,
];

function extractSymbolName(lineText: string): string | undefined {
  for (const pattern of SYMBOL_PATTERNS) {
    const match = lineText.match(pattern);
    if (match?.[1]) {
      const name = match[1];
      if (name.startsWith("_") && name !== "__init__") {
        return undefined;
      }
      if (name.length < 3) {
        return undefined;
      }
      return name;
    }
  }
  return undefined;
}

// --- Trigger pattern helpers (from CommentTriggerWatcher.ts) ---

interface TriggerPattern {
  readonly regex: RegExp;
}

function buildTriggerPatterns(triggers: readonly string[]): readonly TriggerPattern[] {
  if (triggers.length === 0) {
    return [];
  }
  const patterns: TriggerPattern[] = [];
  for (const trigger of triggers) {
    const escaped = trigger.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    patterns.push({
      regex: new RegExp(
        `^\\s*(?:\\/\\/+|#+)\\s*${escaped}\\s*(.+)$`,
        "i",
      ),
    });
  }
  return patterns;
}

function matchTrigger(
  lineText: string,
  patterns: readonly TriggerPattern[],
): string | undefined {
  for (const { regex } of patterns) {
    const match = lineText.match(regex);
    if (match?.[1]) {
      return match[1].trim();
    }
  }
  return undefined;
}

// ==================== Tests ====================

describe("Symbol Extraction (CodeLens patterns)", () => {
  describe("Python", () => {
    it("should match def statements", () => {
      assert.strictEqual(extractSymbolName("def authenticate(user):"), "authenticate");
      assert.strictEqual(extractSymbolName("  def process_data(items):"), "process_data");
    });

    it("should match async def statements", () => {
      assert.strictEqual(extractSymbolName("async def fetch_data():"), "fetch_data");
      assert.strictEqual(extractSymbolName("  async def handle_request(req):"), "handle_request");
    });

    it("should match class statements", () => {
      assert.strictEqual(extractSymbolName("class UserService:"), "UserService");
      assert.strictEqual(extractSymbolName("  class DatabaseManager(Base):"), "DatabaseManager");
    });

    it("should match __init__", () => {
      assert.strictEqual(extractSymbolName("  def __init__(self):"), "__init__");
    });

    it("should skip private names (leading _)", () => {
      assert.strictEqual(extractSymbolName("def _helper():"), undefined);
      assert.strictEqual(extractSymbolName("def __private():"), undefined);
    });

    it("should skip short names (< 3 chars)", () => {
      assert.strictEqual(extractSymbolName("def fn():"), undefined);
      assert.strictEqual(extractSymbolName("def go():"), undefined);
    });
  });

  describe("JavaScript/TypeScript", () => {
    it("should match function declarations", () => {
      assert.strictEqual(extractSymbolName("function handleSubmit() {"), "handleSubmit");
      assert.strictEqual(extractSymbolName("export function createUser() {"), "createUser");
    });

    it("should match async function declarations", () => {
      assert.strictEqual(extractSymbolName("async function fetchData() {"), "fetchData");
      assert.strictEqual(extractSymbolName("export async function loadConfig() {"), "loadConfig");
    });

    it("should match class declarations", () => {
      assert.strictEqual(extractSymbolName("class UserService {"), "UserService");
      assert.strictEqual(extractSymbolName("export class ApiClient {"), "ApiClient");
    });

    it("should match arrow function assignments", () => {
      assert.strictEqual(extractSymbolName("const handleClick = () => {"), "handleClick");
      assert.strictEqual(extractSymbolName("export const processData = (items) => {"), "processData");
    });

    it("should match function expression assignments", () => {
      assert.strictEqual(extractSymbolName("const validate = function(input) {"), "validate");
    });
  });

  describe("Go", () => {
    it("should match func declarations", () => {
      assert.strictEqual(extractSymbolName("func HandleRequest(w http.ResponseWriter) {"), "HandleRequest");
    });

    it("should match method declarations", () => {
      assert.strictEqual(extractSymbolName("func (s *Server) Start(port int) {"), "Start");
    });
  });

  describe("Rust", () => {
    it("should match fn declarations", () => {
      assert.strictEqual(extractSymbolName("fn process_request(req: Request) -> Response {"), "process_request");
      assert.strictEqual(extractSymbolName("pub fn new(config: Config) -> Self {"), "new");
      assert.strictEqual(extractSymbolName("pub async fn fetch(url: &str) -> Result<()> {"), "fetch");
    });
  });

  describe("Java/C#", () => {
    it("should match method declarations", () => {
      assert.strictEqual(extractSymbolName("  public void handleRequest(Request req) {"), "handleRequest");
      assert.strictEqual(extractSymbolName("  private static int calculateSum(int[] nums) {"), "calculateSum");
    });
  });

  describe("edge cases", () => {
    it("should return undefined for non-matching lines", () => {
      assert.strictEqual(extractSymbolName("// this is a comment"), undefined);
      assert.strictEqual(extractSymbolName("x = 42"), undefined);
      assert.strictEqual(extractSymbolName("  return result"), undefined);
      assert.strictEqual(extractSymbolName(""), undefined);
    });
  });
});

describe("Comment Trigger Matching", () => {
  const DEFAULT_TRIGGERS = ["remember:", "note:", "decision:", "todo:"];

  describe("buildTriggerPatterns()", () => {
    it("should create patterns for each trigger", () => {
      const patterns = buildTriggerPatterns(DEFAULT_TRIGGERS);
      assert.strictEqual(patterns.length, 4);
    });

    it("should return empty for empty triggers", () => {
      const patterns = buildTriggerPatterns([]);
      assert.strictEqual(patterns.length, 0);
    });
  });

  describe("matchTrigger()", () => {
    const patterns = buildTriggerPatterns(DEFAULT_TRIGGERS);

    it("should match // remember: comments", () => {
      const result = matchTrigger("// remember: JWT was chosen for auth", patterns);
      assert.strictEqual(result, "JWT was chosen for auth");
    });

    it("should match # remember: comments", () => {
      const result = matchTrigger("# remember: use PostgreSQL for storage", patterns);
      assert.strictEqual(result, "use PostgreSQL for storage");
    });

    it("should match // note: comments", () => {
      const result = matchTrigger("// note: this API is rate-limited", patterns);
      assert.strictEqual(result, "this API is rate-limited");
    });

    it("should match # decision: comments", () => {
      const result = matchTrigger("# decision: switch to Redis for caching", patterns);
      assert.strictEqual(result, "switch to Redis for caching");
    });

    it("should match // todo: comments", () => {
      const result = matchTrigger("// todo: add input validation", patterns);
      assert.strictEqual(result, "add input validation");
    });

    it("should be case-insensitive", () => {
      const result = matchTrigger("// REMEMBER: uppercase trigger", patterns);
      assert.strictEqual(result, "uppercase trigger");

      const result2 = matchTrigger("# Note: mixed case", patterns);
      assert.strictEqual(result2, "mixed case");
    });

    it("should handle leading whitespace", () => {
      const result = matchTrigger("    // remember: indented comment", patterns);
      assert.strictEqual(result, "indented comment");

      const result2 = matchTrigger("  # note: also indented", patterns);
      assert.strictEqual(result2, "also indented");
    });

    it("should handle multiple slashes or hashes", () => {
      const result = matchTrigger("/// remember: triple slash", patterns);
      assert.strictEqual(result, "triple slash");

      const result2 = matchTrigger("## note: double hash", patterns);
      assert.strictEqual(result2, "double hash");
    });

    it("should return undefined for non-matching lines", () => {
      assert.strictEqual(matchTrigger("// regular comment", patterns), undefined);
      assert.strictEqual(matchTrigger("const x = 42;", patterns), undefined);
      assert.strictEqual(matchTrigger("", patterns), undefined);
      assert.strictEqual(matchTrigger("# just a heading", patterns), undefined);
    });

    it("should return undefined if trigger has no content after it", () => {
      // The regex requires at least one char of content after trigger
      assert.strictEqual(matchTrigger("// remember:", patterns), undefined);
    });
  });

  describe("special characters in triggers", () => {
    it("should handle triggers with regex special chars", () => {
      const patterns = buildTriggerPatterns(["bug(fix):"]);
      const result = matchTrigger("// bug(fix): fixed the null pointer", patterns);
      assert.strictEqual(result, "fixed the null pointer");
    });
  });
});
