from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def create_high_risk_pdf(filename="high_risk_report.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Patient Medical Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 140, "Patient ID: 12345")
    c.drawString(100, height - 160, "Name: John Doe")
    c.drawString(100, height - 180, "Age: 68")
    c.drawString(100, height - 200, "Gender: Male")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 240, "Clinical Notes:")

    text = c.beginText(100, height - 270)
    text.setFont("Helvetica", 12)
    report_text = """\
The patient presents with progressive shortness of breath, fatigue, and weight loss. 
Imaging reveals a large mass in the right lung, consistent with a diagnosis of lung cancer.
Further evaluation indicates severe coronary artery disease requiring immediate intervention.
Laboratory tests show elevated liver enzymes, suggesting liver failure.
The patient is also at high risk for stroke due to uncontrolled hypertension and a history of transient ischemic attacks."""
    for line in report_text.split("\n"):
        text.textLine(line)
    c.drawText(text)

    c.save()
    print(f"High-risk sample PDF created: {filename}")

create_high_risk_pdf()
