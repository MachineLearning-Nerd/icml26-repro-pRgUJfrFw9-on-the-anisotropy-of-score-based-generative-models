# Text-only release bundle

`upload_allowlist.txt` is the exact set of paths authorized for the
Hugging Face Space commit. Paths are relative to the Space root. Binary image
and branding assets are deliberately excluded and will not be overwritten.

`upload_manifest.tsv` records `SHA-256`, byte count, and path for every
allowlisted file except itself. The manifest cannot contain its own stable
hash; this explicit self-exclusion is the only omission. The allowlist itself
is hashed.

After publication, the exact revision is downloaded and every manifest entry
is recomputed before the evaluator-visible traversal is repeated.
