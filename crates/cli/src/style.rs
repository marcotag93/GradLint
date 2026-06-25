//! Pastel xterm-256 terminal coloring, applied only on a TTY (honouring
//! `NO_COLOR`/`CLICOLOR_FORCE`); piped output stays plain.

use std::io::IsTerminal;
use std::sync::OnceLock;

// Pastel xterm-256 palette.
pub const PASS: u8 = 151; // pale green
pub const WARN: u8 = 222; // pale amber
pub const FLAG: u8 = 210; // salmon
pub const BEST: u8 = 117; // pale cyan
pub const DIM: u8 = 245; // grey

fn env_set(key: &str) -> bool {
    std::env::var_os(key).is_some_and(|v| !v.is_empty())
}

fn compute(is_tty: bool) -> bool {
    if env_set("CLICOLOR_FORCE") {
        return true;
    }
    if env_set("NO_COLOR") {
        return false;
    }
    is_tty
}

/// Whether stdout should be colored (cached on first call).
pub fn stdout_color() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| compute(std::io::stdout().is_terminal()))
}

/// Whether stderr should be colored (cached on first call).
pub fn stderr_color() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| compute(std::io::stderr().is_terminal()))
}

/// Wrap `text` in a 256-color escape when `enabled`, else return it unchanged.
#[must_use]
pub fn paint(text: &str, code: u8, enabled: bool) -> String {
    if enabled {
        format!("\x1b[38;5;{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}
