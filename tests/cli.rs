//! CLI integration tests: the real binary, real arguments, real exit
//! codes. The serve default is pinned by every test in http_api.rs;
//! this file covers everything that must NOT start a server.

use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_taguru"))
        .args(args)
        .env_remove("TAGURU_CONFIG")
        .output()
        .expect("binary must run")
}

#[test]
fn version_subcommand_prints_the_version_and_nothing_else() {
    let output = run(&["version"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        format!("taguru {}", env!("CARGO_PKG_VERSION"))
    );
    // The old failure mode: any argument silently started the server.
    assert!(!stdout.contains("listening on"));
}

#[test]
fn help_prints_usage_without_starting_a_server() {
    for args in [&["--help"][..], &["-h"][..], &["help"][..]] {
        let output = run(args);
        assert!(output.status.success(), "{args:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("USAGE"), "{args:?}: {stdout}");
        assert!(stdout.contains("TAGURU_DATA_DIR"), "{args:?}: {stdout}");
        assert!(!stdout.contains("listening on"), "{args:?}");
    }
}

#[test]
fn an_unknown_argument_is_refused_with_a_usage_exit() {
    for args in [&["--version"][..], &["nonsense"][..], &["serve", "-x"][..]] {
        let output = run(args);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("--help"), "{args:?}: {stderr}");
        assert!(
            String::from_utf8_lossy(&output.stdout).is_empty(),
            "{args:?}"
        );
    }
}

#[test]
fn version_refuses_trailing_arguments() {
    let output = run(&["version", "extra"]);
    assert_eq!(output.status.code(), Some(2));
}
