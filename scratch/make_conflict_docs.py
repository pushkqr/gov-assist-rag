"""Generate the conflicting-provision demonstration pair.

The live corpus has no clean contradiction on a single question, so the conflict-detection
path in the system prompt (the `> [!WARNING]` instruction) had nothing to trigger it. These
two circulars disagree on one number and nothing else, which is exactly the scenario a judge
asked about.

They are deliberately self-labelling: both carry a document number beginning "DEMO", so the
citation chip in the portal identifies them as demonstration records without needing a
disclaimer in the body text that would leak into generated answers.

Nothing in the retrieval or generation path is special-cased to these files. Detection is the
generic instruction in retrieval/pipeline.py, driven by the [Supersedes: ...] tag that
ingestion extracts.

Usage:  python -m scratch.make_conflict_docs
"""

import pathlib

import pymupdf

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

DOC_2019 = """Government of Maharashtra
Department of Higher and Technical Education
Government Resolution No. DEMO-2019/14/TE-4
Hutatma Rajguru Chowk, Madam Cama Marg,
Mantralaya, Mumbai - 400 032.
Date: 14th March 2019

Subject: Maximum age limit for direct recruitment to the post of Assistant
Professor in Government Colleges and Government Institutes of Technical Education.

Reference: 1) Government Resolution, Higher and Technical Education Department,
No. NGC-2010/193/Mashi-4, dated 30.10.2010.

Preamble:
Representations have been received from various Government Colleges regarding the
age criteria applicable to candidates appearing for direct recruitment to the post
of Assistant Professor. In order to bring uniformity across all Government Colleges
and Government Institutes of Technical Education in the State, the Government has
considered it necessary to prescribe a single maximum age limit. Accordingly, the
Government is pleased to issue the following decision.

GOVERNMENT DECISION:

1. The maximum age limit for a candidate applying for direct recruitment to the
post of Assistant Professor in Government Colleges shall be 35 years, calculated
as on the last date prescribed for submission of the application.

2. Candidates belonging to the reserved categories notified by the Government
shall be granted a relaxation of 5 years in the maximum age limit prescribed
under clause 1 above.

3. Candidates who are already serving as full-time employees of the Government of
Maharashtra shall be granted a relaxation of 5 years in the maximum age limit.

4. No relaxation beyond that prescribed in clauses 2 and 3 shall be permitted by
any appointing authority without the prior approval of the Government.

5. This Government Resolution shall come into force from the date of its issue and
shall apply to all recruitment processes initiated on or after that date.

This Government Resolution is being issued with the approval of the competent
authority.

By order and in the name of the Governor of Maharashtra,

Under Secretary
Department of Higher and Technical Education
"""

DOC_2022 = """Government of Maharashtra
Department of Higher and Technical Education
Government Resolution No. DEMO-2022/88/TE-4
Hutatma Rajguru Chowk, Madam Cama Marg,
Mantralaya, Mumbai - 400 032.
Date: 9th August 2022

Subject: Revision of the maximum age limit for direct recruitment to the post of
Assistant Professor in Government Colleges.

Reference: 1) Government Resolution, Higher and Technical Education Department,
No. DEMO-2019/14/TE-4, dated 14.03.2019.

Preamble:
The maximum age limit for direct recruitment to the post of Assistant Professor was
prescribed by the Government Resolution cited at reference 1 above. Subsequently, the
University Grants Commission revised the minimum qualifications for appointment, which
extended the period ordinarily required for a candidate to become eligible. A number of
universities represented that the existing age limit was resulting in otherwise eligible
candidates being excluded. The matter was under consideration of the Government.

In supersession of Government Resolution No. DEMO-2019/14/TE-4 dated 14th March 2019,
the Government is pleased to issue the following decision.

GOVERNMENT DECISION:

1. The maximum age limit for a candidate applying for direct recruitment to the post
of Assistant Professor in Government Colleges shall be 36 years, calculated as on the
last date prescribed for submission of the application.

2. The relaxations of 5 years for reserved category candidates and for serving
full-time employees of the Government of Maharashtra shall continue to apply, and
shall be calculated against the revised limit prescribed in clause 1 above.

3. Recruitment processes for which advertisements were published prior to the date of
this Government Resolution shall continue to be governed by the age limit prescribed
in Government Resolution No. DEMO-2019/14/TE-4 dated 14th March 2019.

4. This Government Resolution shall come into force from the date of its issue.

This Government Resolution is being issued with the approval of the competent
authority.

By order and in the name of the Governor of Maharashtra,

Under Secretary
Department of Higher and Technical Education
"""

FILES = [
    ("DEMO-2019-AsstProf-AgeLimit.pdf", DOC_2019),
    ("DEMO-2022-AsstProf-AgeLimit-Revision.pdf", DOC_2022),
]


def write_pdf(path: pathlib.Path, text: str) -> None:
    doc = pymupdf.open()
    lines = text.strip().split("\n")
    per_page = 46
    for start in range(0, len(lines), per_page):
        page = doc.new_page()
        rect = pymupdf.Rect(60, 55, 545, 780)
        chunk = "\n".join(lines[start:start + per_page])
        leftover = page.insert_textbox(rect, chunk, fontsize=10, fontname="helv", lineheight=1.35)
        if leftover < 0:
            raise RuntimeError(f"{path.name}: text overflowed the page box by {abs(leftover):.0f}pt")
    doc.save(str(path))
    doc.close()


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, body in FILES:
        target = DOCS / name
        write_pdf(target, body)
        print(f"wrote {target}  ({target.stat().st_size} bytes)")
