type ImageDetail = "auto" | "low" | "high";
type MessageContentImageUrl = {
  type: "image_url";
  image_url: string | { url: string; detail?: ImageDetail | undefined };
};

type MessageContentText = { type: "text"; text: string };
type MessageContentComplex = MessageContentText | MessageContentImageUrl;

type MessageAdditionalKwargs = {
  [x: string]: unknown;

  function_call?: { arguments: string; name: string } | undefined;
  tool_calls?:
    | {
        id: string;
        function: { arguments: string; name: string };
        type: "function";
        index?: number | undefined;
      }[]
    | undefined;
};

export type HumanMessage = {
  type: "human";
  id?: string | undefined;
  content: string | MessageContentComplex[];
};

export type AIMessage = {
  type: "ai";
  id?: string | undefined;
  content: string | MessageContentComplex[];
  tool_calls?:
    | {
        name: string;
        args: { [x: string]: { [x: string]: any } };
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
  content: string | MessageContentComplex[];
  status?: "error" | "success" | undefined;
  lc_direct_tool_output: boolean;
  tool_call_id: string;
  additional_kwargs?: MessageAdditionalKwargs | undefined;
  response_metadata?: Record<string, unknown> | undefined;
};

export type SystemMessage = {
  type: "system";
  id?: string | undefined;
  content: string | MessageContentComplex[];
};

export type FunctionMessage = {
  type: "function";
  id?: string | undefined;
  content: string | MessageContentComplex[];
};

export type RemoveMessage = {
  type: "remove";
  id: string;
  content: string | MessageContentComplex[];
};

export type Message =
  | HumanMessage
  | AIMessage
  | ToolMessage
  | SystemMessage
  | FunctionMessage
  | RemoveMessage;
