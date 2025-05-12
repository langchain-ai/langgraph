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
    auth: "auth/index",
    react: "react/index",
    "react-ui": "react-ui/index",
    "react-ui/server": "react-ui/server/index",
  },
  tsConfigPath: resolve("./tsconfig.json"),
  cjsSource: "./dist-cjs",
  cjsDestination: "./dist",
  additionalGitignorePaths: ["docs"],
  abs,
};
