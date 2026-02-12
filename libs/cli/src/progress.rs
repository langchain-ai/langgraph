use indicatif::{ProgressBar, ProgressStyle};

/// Terminal spinner using indicatif.
///
/// Wraps an indicatif spinner that displays an animated progress indicator
/// with a configurable message.
pub struct Progress {
    spinner: ProgressBar,
}

impl Progress {
    /// Create a new spinner with the given initial message.
    pub fn new(message: &str) -> Self {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .tick_strings(&["|", "/", "-", "\\", ""])
                .template("{spinner} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        spinner.set_message(message.to_string());
        spinner.enable_steady_tick(std::time::Duration::from_millis(100));
        Self { spinner }
    }

    /// Update the spinner message.
    ///
    /// If the message is empty, the spinner is effectively hidden but still running.
    pub fn set_message(&self, msg: &str) {
        self.spinner.set_message(msg.to_string());
    }

    /// Stop the spinner and clear it from the terminal.
    pub fn finish(&self) {
        self.spinner.finish_and_clear();
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        self.spinner.finish_and_clear();
    }
}
