# Real data exports (never committed)

Put exports from IT here. Everything in this folder except this README is excluded from Git
by `.gitignore`, because it contains real customer data.

```
data/
├── emails/          ← .eml files exported by IT (one file per email)
└── outcomes.csv     ← observed outcomes for the Trust metrics page (pilot)
```

- **Emails:** IT exports the agreed mailbox or folder as `.eml` files into `data/emails/`.
  Only customer emails from the agreed period; no personal or HR mail.
- **Docker:** the `pipeline` container sees this folder read-only at `/data`
  (e.g. `/data/emails`). It can read the files but never change or delete them.
- **Access:** the files are personal data under GDPR. Keep the folder on the approved
  machine only, and delete exports once they are indexed and no longer needed.
