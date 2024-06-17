import { resolve, dirname, parse, format } from "node:path";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

function abs(relativePath) {
  return resolve(dirname(fileURLToPath(import.meta.url)), relativePath);
}

async function moveAndRename(source, dest) {
  for (const file of await readdir(abs(source), { withFileTypes: true })) {
    if (file.isDirectory()) {
      await moveAndRename(`${source}/${file.name}`, `${dest}/${file.name}`);
    } else if (file.isFile()) {
      const parsed = parse(file.name);

      // Ignore anything that's not a .js file
      if (parsed.ext !== ".js") {
        continue;
      }

      // Rewrite any require statements to use .cjs
      const content = await readFile(abs(`${source}/${file.name}`), "utf8");
      const rewritten = content.replace(/require\("(\..+?).js"\)/g, (_, p1) => {
        return `require("${p1}.cjs")`;
      });

      // Rename the file to .cjs
      const renamed = format({ name: parsed.name, ext: ".cjs" });

      await writeFile(abs(`${dest}/${renamed}`), rewritten, "utf8");
    }
  }
}

moveAndRename("../dist-cjs", "../dist").catch((err) => {
  console.error(err);
  process.exit(1);
});
