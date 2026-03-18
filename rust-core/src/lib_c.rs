use crate::engine::{
    parse_callback_envelope_json, run_graph_json_with_callback, run_loop_block_on, run_loop_spawn,
    AnyOfCondition, Engine, NodeOutcome,
};
use serde_json::Value;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::mpsc;

type CNodeCallback = unsafe extern "C" fn(
    user_data: libc::c_ulong,
    node: *mut c_char,
    arg_json: *mut c_char,
    state_json: *mut c_char,
) -> *mut c_char;

fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("Received null pointer".to_string());
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str()
        .map_err(|e| format!("Invalid UTF-8 input string: {e}"))
}

fn into_c_ptr(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => CString::new("{\"error\":\"NUL byte in output\"}")
            .expect("static string is valid")
            .into_raw(),
    }
}

#[no_mangle]
pub extern "C" fn rc_engine_new() -> *mut Engine {
    Box::into_raw(Box::new(Engine::new()))
}

#[no_mangle]
/// # Safety
/// `ptr` must be either null or a valid pointer returned by `rc_engine_new`.
pub unsafe extern "C" fn rc_engine_free(ptr: *mut Engine) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(ptr));
}

#[no_mangle]
/// # Safety
/// `ptr` must be either null or a valid pointer returned by this library.
pub unsafe extern "C" fn rc_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    drop(CString::from_raw(ptr));
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
/// `channel` must be a valid null-terminated UTF-8 string pointer.
pub unsafe extern "C" fn rc_add_async_channel(
    ptr: *mut Engine,
    channel: *const c_char,
) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let channel = match cstr_to_str(channel) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    (*ptr).add_async_channel(channel);
    into_c_ptr("{\"ok\":true}".to_string())
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
/// `channel` and `value_json` must be valid null-terminated UTF-8 string pointers.
pub unsafe extern "C" fn rc_publish_json(
    ptr: *mut Engine,
    channel: *const c_char,
    value_json: *const c_char,
) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let channel = match cstr_to_str(channel) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let value_json = match cstr_to_str(value_json) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let value: Value = match serde_json::from_str(value_json) {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"invalid JSON value: {e}\"}}"
            ))
        }
    };
    let result = (*ptr).publish_json(channel, value);
    match result {
        Ok(()) => into_c_ptr("{\"ok\":true}".to_string()),
        Err(e) => into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    }
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
/// `any_of_json` must be a valid null-terminated UTF-8 string pointer.
pub unsafe extern "C" fn rc_wait_any_of_json(
    ptr: *mut Engine,
    any_of_json: *const c_char,
) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let any_of_json = match cstr_to_str(any_of_json) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let any_of: AnyOfCondition = match serde_json::from_str(any_of_json) {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"invalid any_of JSON: {e}\"}}"
            ))
        }
    };
    let result = run_loop_block_on((*ptr).wait_for_any_of_async(&any_of));
    match result {
        Ok(event) => match serde_json::to_string(&event) {
            Ok(s) => into_c_ptr(format!("{{\"ok\":true,\"event\":{s}}}")),
            Err(e) => into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
        },
        Err(e) => into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    }
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
/// `stream_mode` must be null or a valid null-terminated UTF-8 string pointer.
pub unsafe extern "C" fn rc_start_stream(
    ptr: *mut Engine,
    stream_mode: *const c_char,
) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let mode = if stream_mode.is_null() {
        None
    } else {
        match cstr_to_str(stream_mode) {
            Ok(v) => Some(v),
            Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
        }
    };
    match (*ptr).start_stream(mode) {
        Ok(()) => into_c_ptr("{\"ok\":true}".to_string()),
        Err(e) => into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    }
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
pub unsafe extern "C" fn rc_receive_stream_json(ptr: *mut Engine) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let event = run_loop_block_on((*ptr).receive_stream_async());
    match event {
        Some(value) => match serde_json::to_string(&value) {
            Ok(s) => into_c_ptr(format!("{{\"ok\":true,\"has_event\":true,\"event\":{s}}}")),
            Err(e) => into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
        },
        None => into_c_ptr("{\"ok\":true,\"has_event\":false}".to_string()),
    }
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
/// `value_json` must be a valid null-terminated UTF-8 string pointer.
pub unsafe extern "C" fn rc_send_custom_stream_event(
    ptr: *mut Engine,
    value_json: *const c_char,
) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let value_json = match cstr_to_str(value_json) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let value: Value = match serde_json::from_str(value_json) {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"invalid JSON value: {e}\"}}"
            ))
        }
    };
    (*ptr).send_custom_stream_event(value);
    into_c_ptr("{\"ok\":true}".to_string())
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
pub unsafe extern "C" fn rc_close_stream(ptr: *mut Engine) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    (*ptr).close_stream();
    into_c_ptr("{\"ok\":true}".to_string())
}

#[no_mangle]
/// # Safety
/// `ptr` must be a valid engine pointer from `rc_engine_new`.
/// `entry_point`, `finish_point`, and `initial_state_json` must be valid null-terminated UTF-8 pointers.
/// `callback` must be a valid function pointer that returns a malloc-allocated C string.
pub unsafe extern "C" fn rc_run_graph_json(
    ptr: *mut Engine,
    entry_point: *const c_char,
    finish_point: *const c_char,
    initial_state_json: *const c_char,
    initial_input_json: *const c_char,
    stream_mode: *const c_char,
    user_data: libc::c_ulong,
    callback: Option<CNodeCallback>,
) -> *mut c_char {
    if ptr.is_null() {
        return into_c_ptr("{\"ok\":false,\"error\":\"null engine pointer\"}".to_string());
    }
    let Some(callback) = callback else {
        return into_c_ptr("{\"ok\":false,\"error\":\"null callback pointer\"}".to_string());
    };
    let entry_point = match cstr_to_str(entry_point) {
        Ok(v) => v.to_string(),
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let finish_point = match cstr_to_str(finish_point) {
        Ok(v) => v.to_string(),
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let initial_state_json = match cstr_to_str(initial_state_json) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let initial_state: Value = match serde_json::from_str(initial_state_json) {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"invalid initial_state JSON: {e}\"}}"
            ))
        }
    };
    let initial_input_json = match cstr_to_str(initial_input_json) {
        Ok(v) => v,
        Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    };
    let initial_input: Value = match serde_json::from_str(initial_input_json) {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"invalid initial_input JSON: {e}\"}}"
            ))
        }
    };
    let stream_mode = if stream_mode.is_null() {
        None
    } else {
        match cstr_to_str(stream_mode) {
            Ok(v) => Some(v.to_string()),
            Err(e) => return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
        }
    };

    if let Some(mode) = stream_mode.as_deref() {
        if let Err(e) = (*ptr).start_stream(Some(mode)) {
            return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}"));
        }
    }

    let (tx, rx) = mpsc::channel::<Result<Value, String>>();
    let user_data_bits = user_data;
    let run_engine = (*ptr).clone();
    let submit = run_loop_spawn(async move {
        let callback_wrapper = move |node: String,
                                     arg: Value,
                                     state_snapshot: Value|
              -> Result<NodeOutcome<Value, Value>, String> {
            let node_c =
                CString::new(node.clone()).map_err(|e| format!("invalid node name: {e}"))?;
            let arg_json = serde_json::to_string(&arg)
                .map_err(|e| format!("serialize arg for `{node}` failed: {e}"))?;
            let state_json = serde_json::to_string(&state_snapshot)
                .map_err(|e| format!("serialize state for `{node}` failed: {e}"))?;
            let arg_c =
                CString::new(arg_json).map_err(|e| format!("invalid arg JSON bytes: {e}"))?;
            let state_c =
                CString::new(state_json).map_err(|e| format!("invalid state JSON bytes: {e}"))?;
            let out_ptr = unsafe {
                callback(
                    user_data_bits,
                    node_c.as_ptr() as *mut c_char,
                    arg_c.as_ptr() as *mut c_char,
                    state_c.as_ptr() as *mut c_char,
                )
            };
            if out_ptr.is_null() {
                return Err(format!("callback returned null for `{node}`"));
            }
            let out_raw = unsafe { CStr::from_ptr(out_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe {
                libc::free(out_ptr.cast());
            }
            parse_callback_envelope_json(&out_raw, &node)
        };
        let out = run_graph_json_with_callback(
            entry_point,
            finish_point,
            initial_state,
            initial_input,
            run_engine.clone(),
            callback_wrapper,
        )
        .await;
        run_engine.close_stream();
        let _ = tx.send(out);
    });
    if let Err(e) = submit {
        return into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}"));
    }

    let run_result = match rx.recv() {
        Ok(v) => v,
        Err(e) => {
            return into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"run-loop recv failed: {e}\"}}"
            ))
        }
    };
    match run_result {
        Ok(state) => match serde_json::to_string(&state) {
            Ok(s) => into_c_ptr(format!("{{\"ok\":true,\"state\":{s}}}")),
            Err(e) => into_c_ptr(format!(
                "{{\"ok\":false,\"error\":\"serialize state failed: {e}\"}}"
            )),
        },
        Err(e) => into_c_ptr(format!("{{\"ok\":false,\"error\":\"{e}\"}}")),
    }
}
