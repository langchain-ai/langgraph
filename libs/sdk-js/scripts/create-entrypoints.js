import * as fs from "fs";
import * as path from "path";

// This lists all the entrypoints for the library. Each key corresponds to an
// importable path, eg. `import { Foo } from "langgraph-sdk/client"`.
// The value is the path to the file in `src/` that exports the entrypoint.
// This is used to generate the `exports` field in package.json.
// Order is not important.
const entrypoints = { client: "client" };

const updateJsonFile = (relativePath, updateFunction) => {
  const contents = fs.readFileSync(relativePath).toString();
  const res = updateFunction(JSON.parse(contents));
  fs.writeFileSync(relativePath, JSON.stringify(res, null, 2) + "\n");
};

const generateFiles = () => {
  const files = [...Object.entries(entrypoints), ["index", "index"]].flatMap(
    ([key, value]) => {
      const nrOfDots = key.split("/").length - 1;
      const relativePath = "../".repeat(nrOfDots) || "./";
      const compiledPath = `${relativePath}dist/${value}`;
      return [
        [`${key}.cjs`, `module.exports = require('${compiledPath}.cjs');`],
        [`${key}.js`, `export * from '${compiledPath}.js'`],
        [`${key}.d.ts`, `export * from '${compiledPath}.js'`],
        [`${key}.d.cts`, `export * from '${compiledPath}.cjs'`],
      ];
    },
  );

  return Object.fromEntries(files);
};

const updateConfig = () => {
  // Update tsconfig.json `typedocOptions.entryPoints` field
  updateJsonFile("./tsconfig.json", (json) => ({
    ...json,
    typedocOptions: {
      ...json.typedocOptions,
      entryPoints: [...Object.keys(entrypoints)].map((key) => {
        const basePath = `src/${entrypoints[key]}`;
        if (fs.existsSync(`${basePath}.mts`)) {
          return `${basePath}.mts`;
        } else if (fs.existsSync(`${basePath}.ts`)) {
          return `${basePath}.ts`;
        } else {
          console.warn(
            `Warning: Neither ${basePath}.mts nor ${basePath}.ts found for entrypoint ${key}`,
          );
          return `${basePath}.ts`; // Default to .ts if neither exists
        }
      }),
    },
  }));

  const generatedFiles = generateFiles();
  const filenames = Object.keys(generatedFiles);

  // Update package.json `exports` and `files` fields
  updateJsonFile("./package.json", (json) => ({
    ...json,
    exports: Object.assign(
      Object.fromEntries(
        ["index", ...Object.keys(entrypoints)].map((key) => {
          let entryPoint = {
            types: {
              import: `./${key}.d.ts`,
              require: `./${key}.d.cts`,
              default: `./${key}.d.ts`,
            },
            import: `./${key}.js`,
            require: `./${key}.cjs`,
          };

          return [key === "index" ? "." : `./${key}`, entryPoint];
        }),
      ),
      {
        "./package.json": "./package.json",
      },
    ),
    files: ["dist/", ...filenames],
  }));

  // Write generated files
  Object.entries(generatedFiles).forEach(([filename, content]) => {
    fs.mkdirSync(path.dirname(filename), {
      recursive: true,
    });
    fs.writeFileSync(filename, content);
  });

  const gitignore = fs.readFileSync("./.gitignore").toString();
  const lines = gitignore.split("\n");
  const startMarker = "## GENERATED create-entrypoints.js";
  const endMarker = "## END GENERATED create-entrypoints.js";
  const startIdx = lines.findIndex((line) => line.includes(startMarker));
  const endIdx = lines.findIndex((line) => line.includes(endMarker));
  const newLines = lines.slice(0, startIdx + 1);
  if (startIdx === -1) {
    newLines.push(startMarker);
  }
  newLines.push(...filenames.map((fname) => `/${fname}`));
  if (endIdx === -1) {
    newLines.push(endMarker);
  }
  newLines.push(...lines.slice(endIdx));
  fs.writeFileSync("./.gitignore", newLines.join("\n"));
};

const cleanGenerated = () => {
  const filenames = Object.keys(generateFiles());
  filenames.forEach((fname) => {
    try {
      fs.unlinkSync(fname);
    } catch {
      // ignore error
    }
  });
};

const command = process.argv[2];

if (command === "clean") {
  cleanGenerated();
} else {
  updateConfig();
}
