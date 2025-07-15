import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as url from "node:url";

const mdPath = url.fileURLToPath(
  new URL("./add_translation_js_ref.md", import.meta.url)
);

const extractedDir = url.fileURLToPath(
  new URL(
    "../../../oap-langgraphjs-tools-agent/src/add_transaction_js",
    import.meta.url
  )
);

await fs.mkdir(extractedDir, { recursive: true });

const md = (await fs.readFile(mdPath, { encoding: "utf-8" })).split("\n");

const chunks = [];
let current = [];

for (let line of md) {
  if (line.trimStart().startsWith("```")) {
    if (current.length > 0) {
      chunks.push(current.join("\n"));
      current = [];
    } else {
      current.push("// " + line.trimStart());
    }
  } else if (current.length > 0) {
    current.push(line);
  }
}

if (current.length > 0) {
  chunks.push(current.join("\n"));
}

for (let i = 0; i < chunks.length; i += 1) {
  await fs.writeFile(path.resolve(extractedDir, `${i}.mts`), chunks[i], {
    encoding: "utf-8",
  });
}

console.log("finished");
