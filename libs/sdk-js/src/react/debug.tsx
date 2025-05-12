import { ThreadState } from "../schema.js";

interface Node<StateType = any> {
  type: "node";
  value: ThreadState<StateType>;
  path: string[];
}

interface Fork<StateType = any> {
  type: "fork";
  items: Array<Sequence<StateType>>;
}

interface Sequence<StateType = any> {
  type: "sequence";
  items: Array<Node<StateType> | Fork<StateType>>;
}

interface ValidFork<StateType = any> {
  type: "fork";
  items: Array<ValidSequence<StateType>>;
}

interface ValidSequence<StateType = any> {
  type: "sequence";
  items: [Node<StateType>, ...(Node<StateType> | ValidFork<StateType>)[]];
}

// forks
export type CheckpointBranchPath = string[];

export type MessageBranch = {
  current: CheckpointBranchPath;
  options: CheckpointBranchPath[];
};

export function DebugSegmentsView(props: {
  sequence: ValidSequence<ThreadState>;
}) {
  const concatContent = (value: ThreadState<any>) => {
    let content;
    try {
      content = value.values?.messages?.at(-1)?.content ?? "";
    } catch {
      content = JSON.stringify(value.values);
    }

    content = content.replace(/(\n|\r\n)/g, "");
    if (content.length <= 23) return content;
    return `${content.slice(0, 10)}...${content.slice(-10)}`;
  };

  return (
    <div>
      {props.sequence.items.map((item, index) => {
        if (item.type === "fork") {
          return (
            <div key={index}>
              {item.items.map((fork, idx) => {
                const [first] = fork.items;
                return (
                  <details key={idx}>
                    <summary>
                      Fork{" "}
                      <span className="font-mono">
                        ...{first.path.at(-1)?.slice(-4)}
                      </span>
                    </summary>
                    <div className="ml-4">
                      <DebugSegmentsView sequence={fork} />
                    </div>
                  </details>
                );
              })}
            </div>
          );
        }

        if (item.type === "node") {
          return (
            <div key={index} className="flex items-center gap-2">
              <pre>
                ({item.value.metadata?.step}) ...
                {item.value.checkpoint.checkpoint_id?.slice(-4)} (
                {item.value.metadata?.source}): {concatContent(item.value)}
              </pre>
              <button
                type="button"
                className="border rounded-sm text-sm py-0.5 px-1 text-muted-foreground"
                onClick={() => console.log(item.path, item.value)}
              >
                console.log
              </button>
            </div>
          );
        }

        return null;
      })}
    </div>
  );
}
