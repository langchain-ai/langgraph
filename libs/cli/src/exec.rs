use std::io::Write;
use std::process::{Command, Stdio};

/// Run a command synchronously, optionally piping stdin, and capturing stdout/stderr.
///
/// If `verbose` is true, the command is echoed to stdout before execution.
///
/// Returns `(Option<stdout>, Option<stderr>)` on success, or an error message on failure.
pub fn run_command(
    cmd: &str,
    args: &[&str],
    input: Option<&str>,
    verbose: bool,
) -> Result<(Option<String>, Option<String>), String> {
    if verbose {
        let cmd_str = format!("+ {} {}", cmd, args.join(" "));
        if let Some(inp) = input {
            let filtered: Vec<&str> = inp.lines().filter(|l| !l.is_empty()).collect();
            println!("{} <\n{}", cmd_str, filtered.join("\n"));
        } else {
            println!("{cmd_str}");
        }
    }

    let stdin_cfg = if input.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    };

    let mut child = Command::new(cmd)
        .args(args)
        .stdin(stdin_cfg)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to execute `{cmd}`: {e}"))?;

    if let Some(input_data) = input {
        if let Some(ref mut stdin_handle) = child.stdin {
            stdin_handle
                .write_all(input_data.as_bytes())
                .map_err(|e| format!("Failed to write to stdin of `{cmd}`: {e}"))?;
        }
        // Drop stdin to signal EOF
        drop(child.stdin.take());
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait for `{cmd}`: {e}"))?;

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        // 130 = SIGINT (Ctrl-C), not an error
        if code == 130 {
            return Ok((None, None));
        }
        let stdout_str = String::from_utf8_lossy(&output.stdout);
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Command `{cmd}` exited with code {code}\nstdout: {stdout_str}\nstderr: {stderr_str}"
        ));
    }

    let stdout = if output.stdout.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    };
    let stderr = if output.stderr.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(&output.stderr).to_string())
    };

    Ok((stdout, stderr))
}

/// Run a command and stream its stdout/stderr to the parent process in real-time.
///
/// This is useful for long-running commands like `docker compose up` where we
/// want to see output as it happens.
pub fn run_command_streaming(
    cmd: &str,
    args: &[&str],
    input: Option<&str>,
    verbose: bool,
) -> Result<(), String> {
    if verbose {
        let cmd_str = format!("+ {} {}", cmd, args.join(" "));
        if let Some(inp) = input {
            let filtered: Vec<&str> = inp.lines().filter(|l| !l.is_empty()).collect();
            println!("{} <\n{}", cmd_str, filtered.join("\n"));
        } else {
            println!("{cmd_str}");
        }
    }

    let stdin_cfg = if input.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    };

    let mut child = Command::new(cmd)
        .args(args)
        .stdin(stdin_cfg)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("Failed to execute `{cmd}`: {e}"))?;

    if let Some(input_data) = input {
        if let Some(ref mut stdin_handle) = child.stdin {
            stdin_handle
                .write_all(input_data.as_bytes())
                .map_err(|e| format!("Failed to write to stdin of `{cmd}`: {e}"))?;
        }
        drop(child.stdin.take());
    }

    let status = child
        .wait()
        .map_err(|e| format!("Failed to wait for `{cmd}`: {e}"))?;

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        if code == 130 {
            return Ok(());
        }
        return Err(format!("Command `{cmd}` exited with code {code}"));
    }

    Ok(())
}
