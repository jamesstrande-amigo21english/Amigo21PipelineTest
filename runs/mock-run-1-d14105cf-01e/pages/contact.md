[CONTACT | Contact page — inquiry form documenting existing POST /api/contact (no real call in mock)] PAGE DRAFT
Goal: Build Amigo 21 English marketing website page drafts for home, courses/tracks, about, pricing (informational/nonproduction only), contact (document existing POST /api/contact — do not call it), and FAQ. Emit structured page content for the sheer-test pipeline repo; do not deploy, push, or touch Webflow/Cloud SQL/payments/live services.

# Contact — Amigo 21 English
Inquiry form fields: firstName, lastName, email, phone (optional), lessonInterest, message, hidden website honeypot.
Required: firstName, lastName, email, valid course interest, message.
Frontend contract: POST to existing production backend https://amigo21website-7doj6ahycq-ue.a.run.app/api/contact (also reachable as relative /api/contact on the Node app).
MOCK MODE: document the contract only — do NOT perform a real POST.
Health: GET; CORS: OPTIONS. Honeypot returns success-like without insert.
