# Security Policy

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Report it privately through GitHub's
[Report a vulnerability](https://github.com/knowlytix/forgeloop/security/advisories/new)
form, or by email to **security@knowlytix.ai**. Include what you did, what
happened, and the version or commit you were on. We aim to acknowledge within
five working days.

## Scope

This repository is companion code for a book trilogy: example agents, notebooks
and training scripts. It is **teaching material, not a hardened product.** In
scope for a report:

- Anything in this repo that executes untrusted input in a way the examples do
  not intend (the injection-defence chapters deliberately *demonstrate* attacks;
  that is the subject matter, not a vulnerability).
- Secrets, keys or personal data committed to this repo or its history.
- Supply-chain problems: a dependency we pin or a script we ship that fetches
  code from somewhere unexpected.

Out of scope, and better reported elsewhere:

- The `knowlytix` substrate itself — it ships separately and is not part of this
  repo. Report those to security@knowlytix.ai as well, but say so.
- Model behaviour: a prompt that makes an example agent produce a bad answer is
  a modelling limitation the books discuss, not a vulnerability.

## Running this code safely

- **Never commit your license key.** `knowlytix` reads
  `~/.knowlytix/license.key`; that path and `*.key`, `*.lic`, `.env*` are
  git-ignored here. Keep it that way.
- **The example agents call tools and models.** Run them in a virtual
  environment, with API keys scoped to a throwaway project, not your production
  credentials.
- **The banking corpora are synthetic.** Every customer, case and figure in
  `data/` is invented for the examples. Do not replace them with real customer
  data in a clone you intend to push.
