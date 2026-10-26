"""Generates 3 synthetic multi-page PDFs with known facts, so the retrieval
and evaluation pipeline can be tested against ground-truth answers.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")
os.makedirs(OUT_DIR, exist_ok=True)
styles = getSampleStyleSheet()


def make_pdf(filename, title, sections):
    doc = SimpleDocTemplate(os.path.join(OUT_DIR, filename), pagesize=A4)
    flow = [Paragraph(title, styles["Title"]), Spacer(1, 0.3 * inch)]
    for heading, body in sections:
        flow.append(Paragraph(heading, styles["Heading2"]))
        flow.append(Paragraph(body, styles["BodyText"]))
        flow.append(Spacer(1, 0.2 * inch))
    doc.build(flow)
    print(f"Wrote {filename}")


make_pdf(
    "company_handbook.pdf",
    "Northbridge Analytics — Employee Handbook",
    [
        ("Leave Policy", "Full-time employees at Northbridge Analytics accrue 18 days of paid annual leave per year, "
         "plus 10 public holidays. Leave requests must be submitted at least 5 business days in advance through the "
         "internal HR portal for approval by a direct manager."),
        ("Remote Work Policy", "Employees may work remotely up to 3 days per week. Fully remote arrangements require "
         "written approval from the department head and are reviewed every 6 months."),
        ("Expense Reimbursement", "Business expenses under $150 can be self-approved and reimbursed within 7 business "
         "days of submission. Expenses above $150 require prior manager approval before being incurred."),
        ("Probation Period", "New employees undergo a 90-day probation period, during which either party may terminate "
         "employment with 1 week's notice. Performance reviews occur at the 30-day and 60-day marks."),
    ],
)

make_pdf(
    "product_release_notes.pdf",
    "Northbridge Analytics — Platform Release Notes v3.2",
    [
        ("Overview", "Version 3.2 of the Northbridge Analytics platform introduces real-time anomaly detection, "
         "a redesigned dashboard, and expanded API rate limits."),
        ("Anomaly Detection", "The new anomaly detection module uses a rolling 30-day statistical baseline and flags "
         "any metric deviating more than 3 standard deviations from that baseline. Alerts are delivered via email "
         "or Slack within 2 minutes of detection."),
        ("API Rate Limits", "Free-tier API access is now capped at 1,000 requests per hour, up from 500 in v3.1. "
         "Enterprise-tier customers have no fixed cap and are instead subject to fair-use monitoring."),
        ("Known Issues", "Dashboard export to PDF occasionally times out for reports containing more than 50 charts. "
         "A fix is scheduled for v3.3, expected in Q3."),
    ],
)

make_pdf(
    "security_whitepaper.pdf",
    "Northbridge Analytics — Security & Compliance Whitepaper",
    [
        ("Data Encryption", "All customer data is encrypted at rest using AES-256 and in transit using TLS 1.3. "
         "Encryption keys are rotated every 90 days and managed via a dedicated hardware security module (HSM)."),
        ("Compliance Certifications", "Northbridge Analytics is SOC 2 Type II certified and GDPR compliant. "
         "ISO 27001 certification is currently in progress, with completion targeted for the next fiscal year."),
        ("Incident Response", "The security team maintains a documented incident response plan with a target "
         "detection-to-containment time of under 4 hours for critical severity incidents."),
        ("Data Retention", "Customer data is retained for 24 months after account closure, after which it is "
         "permanently and irreversibly deleted, unless a longer retention period is contractually required."),
    ],
)
