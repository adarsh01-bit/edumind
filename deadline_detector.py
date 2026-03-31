# deadline_detector.py
# PURPOSE: Scan document text, find all dates/deadlines,
#          label them, calculate days remaining
#
# BEGINNER NOTE: This uses NLP (Natural Language Processing)
#                spaCy reads text like a human and finds dates


# ── IMPORTS ──────────────────────────────────────────────

import spacy  # NLP library
from dateutil import parser as dateparser  # converts text → date
from dateutil.parser import ParserError  # handles bad dates
from datetime import datetime, date  # date operations
import re  # pattern matching


# ── LOAD SPACY MODEL ─────────────────────────────────────
# en_core_web_sm = small English model
# loads once when file is imported

print("🔤 Loading spaCy language model...")
import spacy


def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        # fallback (NO crash on cloud)
        return spacy.blank("en")


nlp = load_nlp()


# ── KEYWORD MAPS FOR LABELING ─────────────────────────────
# These keywords help us decide what TYPE each deadline is
# We look at words NEAR the date to figure out the label

EXAM_KEYWORDS = [
    "exam",
    "examination",
    "test",
    "quiz",
    "assessment",
    "midterm",
    "final",
    "viva",
    "practical",
]

ASSIGNMENT_KEYWORDS = [
    "assignment",
    "submit",
    "submission",
    "homework",
    "project",
    "report",
    "due",
    "deadline",
    "hand in",
    "upload",
    "turn in",
]

EVENT_KEYWORDS = [
    "workshop",
    "seminar",
    "webinar",
    "event",
    "session",
    "class",
    "lecture",
    "meeting",
    "orientation",
    "registration",
    "enrollment",
    "fest",
    "hackathon",
]

HOLIDAY_KEYWORDS = ["holiday", "vacation", "break", "leave", "closed", "off", "recess"]


# ── FUNCTION 1: Extract Raw Dates From Text ───────────────


def extract_dates_from_text(text):
    """
    Uses spaCy NER to find all date mentions in text.
    Returns list of dicts with date string + surrounding context.

    spaCy labels dates as 'DATE' entities automatically.
    """

    # process text with spaCy
    # spaCy reads the text and identifies entities
    doc = nlp(text)

    raw_dates = []

    for entity in doc.ents:

        # only care about DATE entities
        if entity.label_ == "DATE":

            date_text = entity.text.strip()

            # skip vague dates like "today", "yesterday",
            # "next week" — we need specific dates
            vague_terms = [
                "today",
                "yesterday",
                "tomorrow",
                "last week",
                "next week",
                "recently",
                "soon",
                "earlier",
                "later",
                "now",
                "currently",
                "annually",
                "monthly",
                "weekly",
                "daily",
                "every",
            ]

            is_vague = any(term in date_text.lower() for term in vague_terms)

            if is_vague:
                continue

            # get surrounding context (50 chars before + after)
            # this helps us label the deadline type
            start = max(0, entity.start_char - 80)
            end = min(len(text), entity.end_char + 80)
            context = text[start:end].strip()
            context = context.replace("\n", " ")

            raw_dates.append({"date_text": date_text, "context": context})

    return raw_dates


# ── FUNCTION 2: Parse Date Text → Python Date ─────────────


def parse_date(date_text):
    """
    Converts a date string like "November 15" or "15/11/2025"
    into a Python date object.

    Returns None if the date can't be parsed.
    """

    try:
        # dateutil is smart — handles many formats:
        # "Nov 15", "15-11-2025", "November 15, 2025" etc.
        parsed = dateparser.parse(
            date_text,
            dayfirst=False,  # prefer MM/DD over DD/MM
            default=datetime(  # if year missing, use current year
                datetime.now().year, 1, 1
            ),
        )

        if parsed:
            return parsed.date()  # return just the date part
        return None

    except (ParserError, ValueError, OverflowError):
        return None


# ── FUNCTION 3: Label The Deadline Type ───────────────────


def label_deadline(context):
    """
    Looks at the words around a date and decides
    what TYPE of deadline it is.

    Returns one of: EXAM / ASSIGNMENT / EVENT / HOLIDAY / OTHER
    """

    context_lower = context.lower()

    # check each category — return first match
    for keyword in EXAM_KEYWORDS:
        if keyword in context_lower:
            return "📝 EXAM"

    for keyword in ASSIGNMENT_KEYWORDS:
        if keyword in context_lower:
            return "📋 ASSIGNMENT"

    for keyword in EVENT_KEYWORDS:
        if keyword in context_lower:
            return "🎯 EVENT"

    for keyword in HOLIDAY_KEYWORDS:
        if keyword in context_lower:
            return "🏖️ HOLIDAY"

    return "📌 OTHER"


# ── FUNCTION 4: Calculate Days Remaining ─────────────────


def days_remaining(target_date):
    """
    Calculates how many days from TODAY until target_date.

    Returns:
      positive number = days in future
      0               = today!
      negative number = already passed
    """

    today = date.today()
    delta = target_date - today
    return delta.days


# ── FUNCTION 5: Get Status Label ─────────────────────────


def get_status(days):
    """
    Returns a status string based on days remaining.
    """

    if days < 0:
        return f"✅ Passed ({abs(days)} days ago)"
    elif days == 0:
        return "🔴 TODAY!"
    elif days <= 3:
        return f"🔴 URGENT — {days} day(s) left"
    elif days <= 7:
        return f"🟠 This week — {days} days left"
    elif days <= 14:
        return f"🟡 Soon — {days} days left"
    else:
        return f"🟢 Upcoming — {days} days left"


# ── FUNCTION 6: MAIN — Detect All Deadlines ───────────────


def detect_deadlines(text):
    """
    MASTER FUNCTION — runs everything in order.
    Takes raw document text, returns sorted deadline list.

    Returns list of dicts, each containing:
    - date_text    : original text ("November 15")
    - parsed_date  : Python date object
    - label        : EXAM / ASSIGNMENT / EVENT etc.
    - context      : surrounding sentence
    - days_left    : integer days from today
    - status       : human readable status string
    """

    print("\n🗓️  Scanning document for deadlines...")

    # Step 1: find all raw dates
    raw_dates = extract_dates_from_text(text)
    print(f"   Found {len(raw_dates)} date mentions")

    deadlines = []
    seen_dates = set()  # avoid duplicates

    for item in raw_dates:

        date_text = item["date_text"]
        context = item["context"]

        # Step 2: parse the date text → Python date
        parsed = parse_date(date_text)

        if parsed is None:
            continue  # skip unparseable dates

        # skip duplicates (same date appearing multiple times)
        if parsed in seen_dates:
            continue
        seen_dates.add(parsed)

        # Step 3: label the deadline type
        label = label_deadline(context)

        # Step 4: calculate days remaining
        days = days_remaining(parsed)

        # Step 5: get status
        status = get_status(days)

        # Step 6: add to results
        deadlines.append(
            {
                "date_text": date_text,
                "parsed_date": parsed,
                "label": label,
                "context": context,
                "days_left": days,
                "status": status,
            }
        )

    # Step 7: sort by date — nearest first
    deadlines.sort(key=lambda x: x["parsed_date"])

    print(f"✅ Detected {len(deadlines)} unique deadlines")
    return deadlines


# ── FUNCTION 7: Pretty Print For Terminal ─────────────────


def print_deadlines(deadlines):
    """
    Prints deadlines in a readable format in terminal.
    Used for testing. UI version is in app.py
    """

    if not deadlines:
        print("No specific deadlines found in document.")
        return

    print("\n" + "=" * 55)
    print("🗓️  DEADLINE DASHBOARD")
    print("=" * 55)

    for i, d in enumerate(deadlines):
        print(f"\n  {i+1}. {d['label']}")
        print(f"     Date    : {d['parsed_date'].strftime('%B %d, %Y')}")
        print(f"     Status  : {d['status']}")
        print(f"     Context : ...{d['context'][:80]}...")

    print("\n" + "=" * 55)


# ── TEST BLOCK ────────────────────────────────────────────

if __name__ == "__main__":

    # Test with a fake academic document
    # This simulates what a real syllabus/notice would look like

    sample_text = """
    UNIVERSITY — ACADEMIC CALENDAR 2026

    Dear Students,

    Please note the following important dates for this semester:

    The Mid-Term Examination will be held on April 15, 2026.
    All students must report by 9:00 AM.

    Assignment 1 submission deadline is April 5, 2026.
    Please upload your reports on the portal before midnight.

    The Annual Tech Fest (TechFusion) is scheduled for
    May 2, 2026. Registration closes on April 25, 2026.

    Final Examination will be conducted from June 10, 2026
    to June 20, 2026. Admit cards must be collected before
    June 5, 2026.

    Project Report submission is due on May 28, 2026.

    Summer vacation starts July 1, 2026.

    Regards,
    Academic Office
    """

    print("\n" + "=" * 55)
    print("🧪 Testing Deadline Detector")
    print("=" * 55)
    print(f"📅 Today's date: {date.today().strftime('%B %d, %Y')}")

    deadlines = detect_deadlines(sample_text)
    print_deadlines(deadlines)

    print("\n✅ Deadline Detector Test Complete!\n")
