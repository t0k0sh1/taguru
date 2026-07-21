use super::*;

#[cfg(test)]
mod tests {
    use super::test_support::scratch_dir;
    use super::*;
    use crate::context_proptest::{config as proptest_config, wal_op_strategy};
    use proptest::prelude::*;

    #[test]
    fn file_stem_roundtrips_any_name() {
        for name in [
            "sake",
            "用語集",
            "a/b\\c..d",
            "MiXed-123_ok",
            "%weird%",
            "空白 と 記号!?",
        ] {
            let stem = file_stem(name);
            assert!(
                stem.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'%'),
                "stem '{stem}' carries raw special bytes"
            );
            assert_eq!(name_from_stem(&stem).as_deref(), Some(name));
        }
    }

    #[test]
    fn name_from_stem_rejects_torn_encodings() {
        assert_eq!(name_from_stem("%"), None);
        assert_eq!(name_from_stem("%4"), None);
        assert_eq!(name_from_stem("%zz"), None);
        // Undecodable UTF-8 is refused rather than lossily replaced.
        assert_eq!(name_from_stem("%FF%FE"), None);
    }

    #[test]
    fn meta_sidecar_without_usage_field_loads_with_zeroed_counters() {
        // A sidecar written before usage counters existed.
        let json = br#"{"description":"d","pinned":false,"stats":{"associations":3}}"#;
        let file: MetaFile = serde_json::from_slice(json).unwrap();
        assert_eq!(file.stats.associations, 3);
        assert_eq!(file.usage.reads, 0);
        assert_eq!(file.usage.last_read_epoch, 0);
    }

    #[test]
    fn meta_sidecar_with_tuple_top_concepts_loads_the_pre_36_shape() {
        // A sidecar written before #36 switched top_concepts from
        // [name, count] tuples to {label, count} objects.
        let json = br#"{"description":"d","pinned":false,"stats":{"associations":3,
            "top_concepts":[["sake brewery",6],["brewing",5]]}}"#;
        let file: MetaFile = serde_json::from_slice(json).unwrap();
        assert_eq!(file.meta.description, "d");
        assert_eq!(
            file.stats.top_concepts,
            vec![
                LabelUsage {
                    label: "sake brewery".to_string(),
                    count: 6
                },
                LabelUsage {
                    label: "brewing".to_string(),
                    count: 5
                },
            ]
        );
    }

    #[test]
    fn meta_sidecar_with_object_top_concepts_loads_the_current_shape() {
        let json = br#"{"description":"d","pinned":false,"stats":{"associations":3,
            "top_concepts":[{"label":"sake brewery","count":6}]}}"#;
        let file: MetaFile = serde_json::from_slice(json).unwrap();
        assert_eq!(
            file.stats.top_concepts,
            vec![LabelUsage {
                label: "sake brewery".to_string(),
                count: 6
            }]
        );
    }

    proptest! {
        #![proptest_config(proptest_config())]

        /// Any image watermark splits the acknowledged operation history into
        /// an already-baked prefix and a replayed suffix. A further partial
        /// record beyond that is crash debris from an in-flight write: most
        /// cuts leave invalid or checksum-mismatched bytes, which must
        /// vanish without changing the acknowledged state — but the one cut
        /// that removes only the trailing newline is indistinguishable from
        /// an already-acknowledged record that lost just its delimiter, and
        /// replay keeps it (see `TornTail::Recovered`).
        #[test]
        fn wal_replay_from_any_acknowledged_prefix_rebuilds_the_acknowledged_state(
            candidates in prop::collection::vec(wal_op_strategy(), 1..16),
            watermark_pick in any::<prop::sample::Index>(),
            torn_op in wal_op_strategy(),
            torn_at in any::<prop::sample::Index>(),
        ) {
            let mut expected = Context::default();
            let candidates: Vec<_> = candidates.into_iter().map(WalOp::from).collect();
            let acknowledged: Vec<_> = candidates
                .into_iter()
                .filter(|op| apply_op(&mut expected, op).is_ok())
                .collect();
            let watermark = watermark_pick.index(acknowledged.len() + 1);

            let dir = scratch_dir("wal-prefix-property");
            fs::create_dir_all(&dir).unwrap();
            let path = dir.join("generated.wal.jsonl");
            wal::append_batch(&path, 1, &acknowledged).unwrap();
            let healthy_len = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);

            let mut restored = Context::default();
            for op in &acknowledged[..watermark] {
                replay_op(&mut restored, op);
            }
            let (pending, top) = wal::replay::<WalOp>(&path, watermark as u64).unwrap();
            for op in &pending {
                replay_op(&mut restored, op);
            }
            prop_assert_eq!(restored.to_bytes(), expected.to_bytes());
            prop_assert_eq!(top, acknowledged.len() as u64);

            // Build one real checksummed record, then retain an arbitrary
            // non-empty strict prefix: every possible cut is a torn tail.
            let fragment_path = dir.join("fragment.wal.jsonl");
            let torn_op = WalOp::from(torn_op);
            wal::append_batch(
                &fragment_path,
                acknowledged.len() as u64 + 1,
                std::slice::from_ref(&torn_op),
            )
            .unwrap();
            let record = fs::read(&fragment_path).unwrap();
            let cut = torn_at.index(record.len() - 1) + 1;
            let mut bytes = fs::read(&path).unwrap_or_default();
            bytes.extend_from_slice(&record[..cut]);
            fs::write(&path, bytes).unwrap();

            let (pending, top) = wal::replay::<WalOp>(&path, watermark as u64).unwrap();
            let mut healed = Context::default();
            for op in &acknowledged[..watermark] {
                replay_op(&mut healed, op);
            }
            for op in &pending {
                replay_op(&mut healed, op);
            }

            if cut == record.len() - 1 {
                // The retained prefix is the record's complete,
                // checksum-valid bytes minus only its own trailing
                // newline — byte-for-byte what an already-acknowledged
                // record that lost just its delimiter looks like.
                // `replay` cannot tell the two apart and, by design (see
                // `TornTail::Recovered`), keeps it rather than discarding
                // it as debris.
                let mut fully_acknowledged = Context::default();
                for op in &acknowledged {
                    replay_op(&mut fully_acknowledged, op);
                }
                replay_op(&mut fully_acknowledged, &torn_op);
                prop_assert_eq!(healed.to_bytes(), fully_acknowledged.to_bytes());
                prop_assert_eq!(top, acknowledged.len() as u64 + 1);
                prop_assert_eq!(
                    fs::metadata(&path).unwrap().len(),
                    healthy_len + record.len() as u64
                );
            } else {
                // Every shorter prefix is genuinely incomplete — either
                // invalid JSON or bytes whose checksum cannot match — so
                // it is unacknowledged crash debris that must vanish.
                prop_assert_eq!(healed.to_bytes(), expected.to_bytes());
                prop_assert_eq!(top, acknowledged.len() as u64);
                prop_assert_eq!(fs::metadata(&path).unwrap().len(), healthy_len);
            }

            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn write_atomic_replaces_content_and_leaves_no_staging_file() {
        let dir = scratch_dir("atomic");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file.bin");

        write_atomic(&path, b"first").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"first");
        write_atomic(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        // A successful write consumes its (uniquely named) staging file.
        let leftovers = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.starts_with("tmp"))
            })
            .count();
        assert_eq!(leftovers, 0, "staging files must not survive a commit");

        let _ = fs::remove_dir_all(dir);
    }

    /// The private variant owns its bytes from the first write: the
    /// mode is set on the staged file before content lands, and the
    /// rename carries it to the final name.
    #[cfg(unix)]
    #[test]
    fn write_atomic_private_lands_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("atomic-private");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("secrets.json");
        write_atomic_private(&path, b"{}").unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "mode {mode:o}");
        // Re-persisting keeps the tightened mode.
        write_atomic_private(&path, b"{\"v\":2}").unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "mode {mode:o}");

        let _ = fs::remove_dir_all(dir);
    }

    /// The STAGING file — not just the committed one — is owner-only
    /// the instant it exists: `stage_bytes` must leave no readable
    /// window between create and the secret write (the TOCTOU the
    /// create_new+mode fix closes). Inspect the temp file's mode before
    /// it is committed.
    #[cfg(unix)]
    #[test]
    fn private_staging_file_is_born_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("atomic-private-staging");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("secrets.json");
        let staged = stage_bytes(&path, b"secret", true).unwrap();
        let mode = fs::metadata(&staged).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "staging file mode {mode:o} — must be 0600 at birth, not chmod'd after"
        );
        // The non-private path stays world-default (no regression).
        let staged_plain = stage_bytes(&dir.join("plain.bin"), b"x", false).unwrap();
        assert!(fs::metadata(&staged_plain).is_ok());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn write_atomic_cleans_up_its_staging_file_when_the_commit_fails() {
        let dir = scratch_dir("atomic-commit-fail");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file.bin");
        // Renaming a plain file onto a non-empty directory fails on
        // every platform this project targets — staging (a distinctly
        // named file) still succeeds, so this isolates a commit-phase
        // failure without touching stage_bytes.
        fs::create_dir(&path).unwrap();
        fs::write(path.join("occupied"), b"x").unwrap();

        let result = write_atomic(&path, b"payload");
        assert!(
            result.is_err(),
            "renaming a file onto a non-empty directory must fail"
        );

        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.starts_with("tmp"))
            })
            .map(|entry| entry.path())
            .collect();
        assert!(
            leftovers.is_empty(),
            "a failed commit must not leave its staging file behind: {leftovers:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// parking_lot locks don't poison: a panic while holding one just
    /// unwinds and releases it, so neither the context that panicked, nor
    /// a sibling, nor the registry's own listing bricks for the rest of
    /// the process.
    #[test]
    fn a_panic_mid_write_bricks_only_that_context_not_a_sibling_or_the_registry() {
        let dir = scratch_dir("panic-mid-write");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("cherry", ContextMeta::default()).unwrap();

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.write_context("sake", |_context| {
                panic!("simulated failure mid-write");
            })
        }));
        assert!(
            panicked.is_err(),
            "the panic must propagate, not be swallowed"
        );

        state
            .write_context("sake", |_context| {})
            .expect("the context that panicked stays usable — the lock never poisoned");
        state
            .write_context("cherry", |_context| {})
            .expect("a sibling context is unaffected by the panic");
        assert_eq!(
            state.directory().len(),
            2,
            "the registry's own listing survives the panic too"
        );

        let _ = fs::remove_dir_all(dir);
    }
}
