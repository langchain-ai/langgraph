use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum WaitCondition {
    #[serde(rename = "channel")]
    Channel { channel: String, n: usize },
    #[serde(rename = "timer")]
    Timer { seconds: f64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnyOfCondition {
    pub conditions: Vec<WaitCondition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "condition")]
pub enum WaitEvent {
    #[serde(rename = "channel")]
    Channel {
        channel: String,
        value: serde_json::Value,
    },
    #[serde(rename = "timer")]
    Timer { seconds: f64 },
}

pub struct SendPayload<A> {
    pub node: String,
    pub arg: A,
}

pub struct NodeExecResult<U, A> {
    pub update: Option<U>,
    pub sends: Vec<SendPayload<A>>,
}

type Task = Box<dyn FnOnce() + Send + 'static>;

struct ThreadPool {
    tx: mpsc::Sender<Task>,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl ThreadPool {
    fn new(size: usize, label: &str) -> Self {
        let (tx, rx) = mpsc::channel::<Task>();
        let rx = Arc::new(StdMutex::new(rx));
        let mut workers = Vec::with_capacity(size);
        for idx in 0..size {
            let thread_name = format!("{label}-{idx}");
            let rx = Arc::clone(&rx);
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || loop {
                    let task = {
                        let guard = rx.lock().expect("thread-pool receiver mutex poisoned");
                        guard.recv()
                    };
                    match task {
                        Ok(task) => task(),
                        Err(_) => break,
                    }
                })
                .expect("failed to spawn thread-pool worker");
            workers.push(handle);
        }
        Self {
            tx,
            _workers: workers,
        }
    }

    fn execute<F>(&self, task: F) -> Result<(), String>
    where
        F: FnOnce() + Send + 'static,
    {
        self.tx
            .send(Box::new(task))
            .map_err(|e| format!("thread-pool send failed: {e}"))
    }
}

pub fn node_pool_execute<F>(task: F) -> Result<(), String>
where
    F: FnOnce() + Send + 'static,
{
    static NODE_POOL: OnceLock<ThreadPool> = OnceLock::new();
    let pool = NODE_POOL.get_or_init(|| {
        let size = thread::available_parallelism()
            .map(|n| n.get().max(2))
            .unwrap_or(4);
        ThreadPool::new(size, "langgraph-node")
    });
    pool.execute(task)
}

pub fn run_loop_pool_execute<F>(task: F) -> Result<(), String>
where
    F: FnOnce() + Send + 'static,
{
    static RUN_LOOP_POOL: OnceLock<ThreadPool> = OnceLock::new();
    let pool = RUN_LOOP_POOL.get_or_init(|| ThreadPool::new(2, "langgraph-runloop"));
    pool.execute(task)
}

pub fn run_scheduler_loop<U: Send + 'static, A: Send + 'static, FSpawn, FMerge>(
    entry_point: String,
    finish_point: &str,
    initial_arg: A,
    mut spawn: FSpawn,
    mut merge: FMerge,
    rx: mpsc::Receiver<Result<(String, NodeExecResult<U, A>), String>>,
) -> Result<(), String>
where
    FSpawn: FnMut(String, A) -> Result<(), String>,
    FMerge: FnMut(&str, Option<U>) -> Result<(), String>,
{
    let mut active: usize = 1;
    spawn(entry_point, initial_arg)?;

    while active > 0 {
        let item = rx
            .recv()
            .map_err(|e| format!("scheduler recv failed: {e}"))?;
        active = active.saturating_sub(1);
        let (node_name, node_result) = item.map_err(|e| format!("node execution failed: {e}"))?;

        merge(&node_name, node_result.update)?;

        if node_name == finish_point {
            break;
        }

        for send in node_result.sends {
            active += 1;
            spawn(send.node, send.arg)?;
        }
    }

    Ok(())
}

pub fn merge_json_update(state: &mut Value, update: Option<Value>) {
    let Some(update_value) = update else {
        return;
    };
    match (&mut *state, update_value) {
        (Value::Object(state_obj), Value::Object(update_obj)) => {
            for (k, v) in update_obj {
                state_obj.insert(k, v);
            }
        }
        (Value::Object(state_obj), Value::Array(entries)) => {
            for entry in entries {
                if let Value::Array(pair) = entry {
                    if pair.len() == 2 {
                        if let Value::String(key) = &pair[0] {
                            state_obj.insert(key.clone(), pair[1].clone());
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

#[derive(Clone, Default)]
pub struct Engine {
    channels: Arc<Mutex<HashMap<String, VecDeque<serde_json::Value>>>>,
    channel_notify: Arc<Condvar>,
}

impl Engine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_async_channel(&self, name: &str) {
        let mut channels = self.channels.lock();
        channels.entry(name.to_owned()).or_default();
    }

    pub fn publish_json(&self, channel: &str, value: serde_json::Value) -> Result<(), String> {
        let mut channels = self.channels.lock();
        let queue = channels
            .get_mut(channel)
            .ok_or_else(|| format!("Unknown channel `{channel}`"))?;
        queue.push_back(value);
        // Wake up waiters blocked on channel conditions/any_of.
        self.channel_notify.notify_all();
        Ok(())
    }

    pub fn wait_for(&self, cond: &WaitCondition) -> Result<WaitEvent, String> {
        match cond {
            WaitCondition::Channel { channel, n } => {
                if *n < 1 {
                    return Err("channel condition n must be >= 1".to_string());
                }
                let mut channels = self.channels.lock();
                loop {
                    let queue = channels
                        .get_mut(channel)
                        .ok_or_else(|| format!("Unknown channel `{channel}`"))?;
                    if queue.len() >= *n {
                        if *n == 1 {
                            if let Some(value) = queue.pop_front() {
                                return Ok(WaitEvent::Channel {
                                    channel: channel.clone(),
                                    value,
                                });
                            }
                        } else {
                            let mut values = Vec::with_capacity(*n);
                            for _ in 0..*n {
                                if let Some(v) = queue.pop_front() {
                                    values.push(v);
                                }
                            }
                            return Ok(WaitEvent::Channel {
                                channel: channel.clone(),
                                value: serde_json::Value::Array(values),
                            });
                        }
                    }
                    self.channel_notify.wait(&mut channels);
                }
            }
            WaitCondition::Timer { seconds } => {
                if *seconds <= 0.0 {
                    return Err("timer condition must be > 0".to_string());
                }
                std::thread::sleep(Duration::from_secs_f64(*seconds));
                Ok(WaitEvent::Timer { seconds: *seconds })
            }
        }
    }

    pub fn wait_for_any_of(&self, any_of: &AnyOfCondition) -> Result<WaitEvent, String> {
        if any_of.conditions.is_empty() {
            return Err("any_of requires at least one condition".to_string());
        }

        let started = Instant::now();
        let mut min_timer: Option<f64> = None;
        for cond in &any_of.conditions {
            if let WaitCondition::Timer { seconds } = cond {
                if *seconds <= 0.0 {
                    return Err("timer condition must be > 0".to_string());
                }
                min_timer = Some(min_timer.map_or(*seconds, |x| x.min(*seconds)));
            }
        }

        let mut channels = self.channels.lock();
        loop {
            for cond in &any_of.conditions {
                if let WaitCondition::Channel { channel, n } = cond {
                    if *n < 1 {
                        return Err("channel condition n must be >= 1".to_string());
                    }
                    let queue = channels
                        .get_mut(channel)
                        .ok_or_else(|| format!("Unknown channel `{channel}`"))?;
                    if queue.len() >= *n {
                        if *n == 1 {
                            if let Some(value) = queue.pop_front() {
                                return Ok(WaitEvent::Channel {
                                    channel: channel.clone(),
                                    value,
                                });
                            }
                        } else {
                            let mut values = Vec::with_capacity(*n);
                            for _ in 0..*n {
                                if let Some(v) = queue.pop_front() {
                                    values.push(v);
                                }
                            }
                            return Ok(WaitEvent::Channel {
                                channel: channel.clone(),
                                value: serde_json::Value::Array(values),
                            });
                        }
                    }
                }
            }

            if let Some(seconds) = min_timer {
                let timeout = Duration::from_secs_f64(seconds);
                let elapsed = started.elapsed();
                if elapsed >= timeout {
                    return Ok(WaitEvent::Timer { seconds });
                }
                let remaining = timeout.saturating_sub(elapsed);
                self.channel_notify.wait_for(&mut channels, remaining);
            } else {
                self.channel_notify.wait(&mut channels);
            }
        }
    }
}
