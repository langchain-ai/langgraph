import * as path from "node:path";
import * as fs from "node:fs/promises";
import * as url from "node:url";

const mdPath = url.fileURLToPath(
  new URL("./add_translation_js_ref_updated.md", import.meta.url)
);

const extractedDir = url.fileURLToPath(
  new URL(
    "../../../oap-langgraphjs-tools-agent/src/add_transaction_js",
    import.meta.url
  )
);

const files = (await fs.readdir(extractedDir, { withFileTypes: true })).sort(
  (a, b) => {
    const aInt = Number.parseInt(a.name.split(".")[0], 10);
    const bInt = Number.parseInt(b.name.split(".")[0], 10);
    return aInt - bInt;
  }
);

let count = 0;

let lines = [];

for (let file of files) {
  if (file.isDirectory() || !file.name.endsWith(".mts")) continue;
  count += 1;

  const content = await fs.readFile(path.resolve(extractedDir, file.name), {
    encoding: "utf-8",
  });

  lines = lines.concat(
    content
      .split("\n")
      .reduce((acc, line) => {
        if (line.trimStart().startsWith("// ```")) acc.push([]);
        acc.at(-1)?.push(line);
        return acc;
      }, [])
      .map((i) => {
        const tag = i[0].trimStart().slice("// ```".length);
        return ["```" + tag, ...i.slice(1), "```"].join("\n");
      })
  );
}

await fs.writeFile(mdPath, lines.join("\n\n"));
