import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * @param {string} relativePath
 * @returns {string}
 */
function abs(relativePath) {
  return resolve(dirname(fileURLToPath(import.meta.url)), relativePath);
}

export const config = {
  internals: [/react/],
  entrypoints: {
    index: "index",
    client: "client",
    react: "react/index",
    "react-ui": "react-ui/index",
    "react-ui/server": "react-ui/server",
    "react-ui/types": "react-ui/types",
  },
  tsConfigPath: resolve("./tsconfig.json"),
  cjsSource: "./dist-cjs",
  cjsDestination: "./dist",
  additionalGitignorePaths: ["docs"],
  abs,
};
