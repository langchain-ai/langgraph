type ImageDetail = "auto" | "low" | "high";
type MessageContentImageUrl = {
  type: "image_url";
  image_url: string | { url: string; detail?: ImageDetail | undefined };
};

type MessageContentText = { type: "text"; text: string };
type MessageContentComplex = MessageContentText | MessageContentImageUrl;
type MessageContent = string | MessageContentComplex[];

/**
 * Model-specific additional kwargs, which is passed back to the underlying LLM.
 */
type MessageAdditionalKwargs = Record<string, unknown>;

export type HumanMessage = {
  type: "human";
  id?: string | undefined;
  content: MessageContent;
};

export type AIMessage = {
  type: "ai";
  id?: string | undefined;
  content: MessageContent;
  tool_calls?:
    | {
        name: string;
        args: { [x: string]: any };
        id?: string | undefined;
        type?: "tool_call" | undefined;
      }[]
    | undefined;
  invalid_tool_calls?:
    | {
        name?: string | undefined;
        args?: string | undefined;
        id?: string | undefined;
        error?: string | undefined;
        type?: "invalid_tool_call" | undefined;
      }[]
    | undefined;
  usage_metadata?:
    | {
        input_tokens: number;
        output_tokens: number;
        total_tokens: number;
        input_token_details?:
          | {
              audio?: number | undefined;
              cache_read?: number | undefined;
              cache_creation?: number | undefined;
            }
          | undefined;
        output_token_details?:
          | { audio?: number | undefined; reasoning?: number | undefined }
          | undefined;
      }
    | undefined;
  additional_kwargs?: MessageAdditionalKwargs | undefined;
  response_metadata?: Record<string, unknown> | undefined;
};

export type ToolMessage = {
  type: "tool";
  name?: string | undefined;
  id?: string | undefined;
  content: MessageContent;
  status?: "error" | "success" | undefined;
  tool_call_id: string;
  additional_kwargs?: MessageAdditionalKwargs | undefined;
  response_metadata?: Record<string, unknown> | undefined;
  /**
   * Artifact of the Tool execution which is not meant to be sent to the model.
   *
   * Should only be specified if it is different from the message content, e.g. if only
   * a subset of the full tool output is being passed as message content but the full
   * output is needed in other parts of the code.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  artifact?: any;
};

export type SystemMessage = {
  type: "system";
  id?: string | undefined;
  content: MessageContent;
};

export type FunctionMessage = {
  type: "function";
  id?: string | undefined;
  content: MessageContent;
};

export type RemoveMessage = {
  type: "remove";
  id: string;
  content: MessageContent;
};

export type Message =
  | HumanMessage
  | AIMessage
  | ToolMessage
  | SystemMessage
  | FunctionMessage
  | RemoveMessage;
