#ifndef LANGGRAPH_RUST_CORE_H
#define LANGGRAPH_RUST_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Engine Engine;
typedef char* (*rc_node_callback_t)(
    unsigned long user_data,
    char* node,
    char* arg_json,
    char* state_json
);

Engine* rc_engine_new(void);
void rc_engine_free(Engine* ptr);

char* rc_add_async_channel(Engine* ptr, const char* channel);
char* rc_add_custom_output_stream(Engine* ptr, const char* stream_name);
char* rc_publish_json(Engine* ptr, const char* channel, const char* value_json);
char* rc_wait_any_of_json(Engine* ptr, const char* any_of_json);
char* rc_start_stream(Engine* ptr, const char* stream_mode);
char* rc_receive_stream_json(Engine* ptr, const char* stream_name);
char* rc_send_custom_stream_event(Engine* ptr, const char* stream_name, const char* value_json);
char* rc_close_all_streams(Engine* ptr);
char* rc_run_graph_json(
    Engine* ptr,
    const char* entry_point,
    const char* finish_point,
    const char* initial_state_json,
    const char* initial_input_json,
    const char* stream_mode,
    unsigned long user_data,
    rc_node_callback_t callback
);

void rc_string_free(char* ptr);

#ifdef __cplusplus
}
#endif

#endif

