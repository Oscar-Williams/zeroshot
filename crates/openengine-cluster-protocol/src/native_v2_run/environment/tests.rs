use super::*;
use crate::RuntimePlan;
use serde_json::json;

#[test]
fn environment_rejects_reserved_names_oversized_scripts_and_ambiguous_credentials() {
    for name in ["PATH", "HOME", "ZEROSHOT_TOOLS", "CODEX_HOME", "TMPDIR"] {
        let environment: RuntimeEnvironment =
            serde_json::from_value(json!({"variables": {name: "override"}})).unwrap();
        assert!(environment.validate().is_err());
    }
    let oversized = RuntimeEnvironment {
        setup: Some("x".repeat(MAX_RUNTIME_SCRIPT_BYTES + 1)),
        ..Default::default()
    };
    assert!(oversized.validate().is_err());
    let environment: RuntimeEnvironment = serde_json::from_value(json!({
        "variables": {"REGISTRY_TOKEN":"public"}, "connections": {"registry":["REGISTRY_TOKEN"]}
    }))
    .unwrap();
    assert!(environment.validate().is_err());
    let environment: RuntimeEnvironment = serde_json::from_value(json!({
        "startup":"echo ready", "variables":{"NODE_ENV":"test"}, "connections":{"registry":["REGISTRY_TOKEN"]}
    })).unwrap();
    assert!(environment.validate().is_ok());
}

#[test]
fn runtime_hook_connections_join_requirements_without_mutating_node_bindings() {
    let runtime: RuntimePlan = serde_json::from_value(json!({
        "harness":"codex", "provider":"openai", "size":"small", "nodes":{},
        "environment":{"connections":{"registry":["REGISTRY_TOKEN"]}}
    }))
    .unwrap();
    assert_eq!(runtime.connection_requirements().len(), 1);
    assert!(runtime.nodes().is_empty());
    let serialized = serde_json::to_value(runtime).unwrap();
    assert_eq!(
        serialized["environment"]["connections"]["registry"][0],
        "REGISTRY_TOKEN"
    );
}

#[test]
fn preparation_failures_cannot_be_authored_as_graph_outcomes() {
    for reason in [
        "environment_setup_failed",
        "environment_startup_failed",
        "environment_preparation_timeout",
    ] {
        assert!(crate::FailReason::new(crate::EnumLabel::new(reason).unwrap()).is_err());
    }
}
