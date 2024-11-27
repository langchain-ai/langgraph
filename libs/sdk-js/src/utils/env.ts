export function getEnvironmentVariable(name: string): string | undefined {
  // Certain setups (Deno, frontend) will throw an error if you try to access environment variables
  try {
    return typeof process !== "undefined"
      ? // eslint-disable-next-line no-process-env
        process.env?.[name]
      : undefined;
  } catch (e) {
    return undefined;
  }
}
