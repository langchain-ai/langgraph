export function mergeSignals(...signals: (AbortSignal | null | undefined)[]) {
  const nonZeroSignals = signals.filter(
    (signal): signal is AbortSignal => signal != null,
  );

  if (nonZeroSignals.length === 0) return undefined;
  if (nonZeroSignals.length === 1) return nonZeroSignals[0];

  const controller = new AbortController();
  for (const signal of signals) {
    if (signal?.aborted) {
      controller.abort(signal.reason);
      return controller.signal;
    }

    signal?.addEventListener("abort", () => controller.abort(signal.reason), {
      once: true,
    });
  }

  return controller.signal;
}
