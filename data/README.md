# Phase 0 dataset

This directory contains the controlled, synthetic corpus for **I Got This — What's Next?**

- `sample/` is safe to commit and contains exactly 20 fictional source documents.
- `private/` is reserved for personal documents and is ignored by Git.
- The dataset's reference date is **Thursday, August 20, 2026**. Relative phrases in the evaluation set such as "this week" and "this weekend" are interpreted from that date in the `America/Los_Angeles` timezone.
- Family members use anonymous identifiers: `adult_01`, `adult_02`, `child_01`, and `child_02`.

Every sample document begins with YAML-style front matter containing stable source metadata. Event-level dates, statuses, and action items remain in the document body so later ingestion experiments can test extraction and retrieval rather than relying only on filters.

The corpus is intentionally redundant: dates and commitments overlap, some facts are repeated across domains, and several complete answers require more than one source.
