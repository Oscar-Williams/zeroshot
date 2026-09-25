use super::*;
use openengine_cluster_testkit::assertions::AssertValue;

fn usage(
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
) -> TokenUsageDelta {
    TokenUsageDelta {
        input_tokens: TokenCount::new(input_tokens).assert_value(),
        output_tokens: TokenCount::new(output_tokens).assert_value(),
        cache_read_input_tokens: cache_read_input_tokens
            .map(|value| TokenCount::new(value).assert_value()),
        cache_creation_input_tokens: cache_creation_input_tokens
            .map(|value| TokenCount::new(value).assert_value()),
    }
}

#[tokio::test]
async fn thread_ids_are_opaque_beyond_process_argv_requirements() {
    let session = CodexSession::new(true);
    let thread_id = format!("thread-{}\nwith-tab\t", "x".repeat(512));

    assert_eq!(session.record_thread(Some(&thread_id), None).await, Ok(()));
    assert_eq!(
        session
            .record_thread(Some(&thread_id), Some(&thread_id))
            .await,
        Ok(())
    );
    assert_eq!(
        session.thread_id.lock().await.as_deref(),
        Some(thread_id.as_str())
    );
}

#[tokio::test]
async fn thread_ids_reject_only_missing_empty_nul_or_conflicting_values() {
    for (observed, expected) in [
        (None, "Codex output did not provide a thread ID"),
        (Some(""), "Codex output provided an empty thread ID"),
        (
            Some("thread\0id"),
            "Codex output thread ID contained a NUL byte",
        ),
    ] {
        assert_eq!(
            CodexSession::new(true).record_thread(observed, None).await,
            Err(expected)
        );
    }

    let session = CodexSession::new(true);
    assert_eq!(session.record_thread(Some("one"), None).await, Ok(()));
    assert_eq!(
        session.record_thread(Some("two"), Some("one")).await,
        Err("Codex output thread ID did not match the resumed session")
    );
    assert_eq!(
        session.record_thread(None, Some("two")).await,
        Err("Codex output thread ID changed across turns")
    );
}

#[tokio::test]
async fn cumulative_usage_is_normalized_before_commit() {
    let session = CodexSession::new(true);
    let first = usage(10, 4, Some(3), Some(2));
    assert_eq!(session.usage_delta(Some(first)).await, Some(first));
    session.commit_usage(Some(first)).await;

    let second = usage(16, 9, Some(5), Some(7));
    assert_eq!(
        session.usage_delta(Some(second)).await,
        Some(usage(6, 5, Some(2), Some(5)))
    );
    assert_eq!(session.usage_delta(None).await, None);
}

#[tokio::test]
async fn a_decreased_counter_starts_a_new_usage_generation() {
    let session = CodexSession::new(true);
    let previous = usage(10, 4, Some(3), Some(2));
    session.commit_usage(Some(previous)).await;

    let observed = usage(2, 1, Some(1), Some(1));
    assert_eq!(session.usage_delta(Some(observed)).await, Some(observed));
}
