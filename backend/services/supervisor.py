"""Deterministic intent supervisor. LLM never selects endpoint or role."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    COLLEGE_INFORMATION = "COLLEGE_INFORMATION"
    ADMISSION_INFORMATION = "ADMISSION_INFORMATION"
    ADMISSION_APPLICATION = "ADMISSION_APPLICATION"
    ACADEMIC_INFORMATION = "ACADEMIC_INFORMATION"
    ATTENDANCE_POLICY = "ATTENDANCE_POLICY"
    PERSONAL_ATTENDANCE = "PERSONAL_ATTENDANCE"
    TIMETABLE = "TIMETABLE"
    CALENDAR = "CALENDAR"
    LEAVE_POLICY = "LEAVE_POLICY"
    LEAVE_REQUEST = "LEAVE_REQUEST"
    OD_POLICY = "OD_POLICY"
    OD_REQUEST = "OD_REQUEST"
    REQUEST_STATUS = "REQUEST_STATUS"
    REQUEST_CANCELLATION = "REQUEST_CANCELLATION"
    PLACEMENT_INFORMATION = "PLACEMENT_INFORMATION"
    PLACEMENT_REGISTRATION = "PLACEMENT_REGISTRATION"
    TEACHER_CLASSES = "TEACHER_CLASSES"
    TEACHER_REQUESTS = "TEACHER_REQUESTS"
    TEACHER_ATTENDANCE = "TEACHER_ATTENDANCE"
    TEACHER_REQUEST_APPROVAL = "TEACHER_REQUEST_APPROVAL"
    TEACHER_REQUEST_REJECTION = "TEACHER_REQUEST_REJECTION"
    LESSON_DOUBT = "LESSON_DOUBT"
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class IntentDecision:
    intent: Intent
    source: str
    action: Optional[str] = None
    method: Optional[str] = None
    path: Optional[str] = None
    required_role: Optional[str] = None
    write: bool = False


class SupervisorAgent:
    """Small rules-based router with explicit roles and route allowlist."""

    _policy_words = r"requirement|required|minimum|threshold|policy|rule|eligible|eligibility|how\s+(?:do|can)\s+i\s+(?:apply|request)|process|procedure"

    def classify(self, text: str, role: Optional[str] = None) -> IntentDecision:
        q = re.sub(r"\s+", " ", text.lower()).strip()
        if not q:
            return IntentDecision(Intent.UNKNOWN, "none")

        # Teacher writes must be recognized before generic request/status intents.
        if re.search(r"\b(reject|deny)\b.{0,50}\b(request|leave|od)\b|\b(reject|deny)\b", q):
            return IntentDecision(Intent.TEACHER_REQUEST_REJECTION, "kit_api", "reject_teacher_request", "POST", None, "teacher", True)
        if re.search(r"\b(approve|accept|approval)\b.{0,50}\b(request|leave|od)\b|\b(approve|accept)\s+(?:request\s+)?#?\d+", q):
            return IntentDecision(Intent.TEACHER_REQUEST_APPROVAL, "kit_api", "approve_teacher_request", "POST", None, "teacher", True)
        if re.search(r"\b(class|classes)\b.*\b(attendance|present|absent)\b", q):
            return IntentDecision(Intent.TEACHER_ATTENDANCE, "kit_api", "get_teacher_class_attendance", "GET", None, "teacher")
        if re.search(r"\b(my\s+)?(classes|class list|assigned classes)\b", q):
            return IntentDecision(Intent.TEACHER_CLASSES, "kit_api", "get_teacher_classes", "GET", "/v1/teacher/classes", "teacher")
        if role == "teacher" and re.search(r"\b(pending|student)\b.{0,30}\b(requests?)\b|\b(requests?)\b.{0,30}\b(pending|student)\b", q):
            return IntentDecision(Intent.TEACHER_REQUESTS, "kit_api", "get_teacher_requests", "GET", "/v1/teacher/requests", "teacher")

        # Separate policy questions from personal attendance before broad matches.
        if re.search(r"\b(attendance|minimum attendance|attendance percentage)\b", q) and re.search(self._policy_words, q):
            return IntentDecision(Intent.ATTENDANCE_POLICY, "chroma", "search_kit_documents")
        if re.search(r"\b(on[- ]?duty|\bod\b)\b", q) and re.search(self._policy_words, q):
            return IntentDecision(Intent.OD_POLICY, "chroma", "search_kit_documents")
        if re.search(r"\b(leave|medical leave|personal leave)\b", q) and re.search(self._policy_words, q):
            return IntentDecision(Intent.LEAVE_POLICY, "chroma", "search_kit_documents")

        if re.search(r"\b(cancel|withdraw)\b.{0,40}\b(request|leave|od)\b", q):
            return IntentDecision(Intent.REQUEST_CANCELLATION, "kit_api", "cancel_student_request", "POST", None, "student", True)
        if re.search(r"\b(register|sign up|apply)\b.{0,50}\b(placement|drive)\b|\b(placement|drive)\b.{0,50}\b(register|sign up)\b", q):
            return IntentDecision(Intent.PLACEMENT_REGISTRATION, "kit_api", "register_placement", "POST", None, "student", True)
        if re.search(r"\b(submit|make|create|apply for|request)\b.{0,40}\b(on[- ]?duty|\bod\b)\b|\b(on[- ]?duty|\bod\b)\s+(request|application)\b", q):
            return IntentDecision(Intent.OD_REQUEST, "kit_api", "create_student_od_request", "POST", None, "student", True)
        if re.search(r"\b(submit|make|create|apply for|request)\b.{0,40}\b(leave|medical leave|personal leave)\b|\b(leave|medical leave|personal leave)\s+(request|application)\b", q):
            return IntentDecision(Intent.LEAVE_REQUEST, "kit_api", "create_student_leave_request", "POST", None, "student", True)
        if re.search(r"\b(request|leave|od)\b.{0,35}\b(status|detail|details)\b|\b(status|detail|details)\b.{0,35}\b(request|leave|od)\b", q):
            return IntentDecision(Intent.REQUEST_STATUS, "kit_api", "get_student_requests", "GET", "/v1/me/requests", "student")

        if re.search(r"\b(my\s+)?(attendance|attended|absent|low attendance)\b", q) or re.search(r"\b(subject[- ]wise|which subjects?)\b.{0,50}\b(attendance|low)\b", q):
            return IntentDecision(Intent.PERSONAL_ATTENDANCE, "kit_api", "get_student_attendance", "GET", "/v1/me/attendance", "student")
        if re.search(r"\b(timetable|schedule|class time)\b", q):
            return IntentDecision(Intent.TIMETABLE, "kit_api", "get_student_timetable", "GET", "/v1/me/timetable", "student")
        if re.search(r"\b(calendar|holiday|exam date|events? today|what happens today)\b", q):
            if role == "student":
                return IntentDecision(Intent.CALENDAR, "kit_api", "get_calendar", "GET", "/v1/calendar", "student")
            return IntentDecision(Intent.CALENDAR, "kit_api", "get_today", "GET", "/v1/today")
        if re.search(r"\b(academic|academics|subjects?|semester|marks|grades|result)\b", q):
            action = "get_student_subjects" if re.search(r"\b(subject|subjects)\b", q) else "get_student_academics"
            path = "/v1/me/subjects" if action == "get_student_subjects" else "/v1/me/academics"
            return IntentDecision(Intent.ACADEMIC_INFORMATION, "kit_api", action, "GET", path, "student")
        if re.search(r"\b(my\s+)?(placement|placements|drives|internship)\b", q):
            if re.search(r"\b(my|registered|registration|applied)\b", q):
                return IntentDecision(Intent.PLACEMENT_INFORMATION, "kit_api", "get_placement_registrations", "GET", "/v1/me/placements", "student")
            return IntentDecision(Intent.PLACEMENT_INFORMATION, "kit_api", "get_placement_drives", "GET", "/v1/placements/drives", "student")

        if re.search(r"\b(apply|application|enroll|enrol)\b.{0,40}\b(admission|course|college)\b", q):
            return IntentDecision(Intent.ADMISSION_APPLICATION, "frontend_form", "prepare_admission", write=True)
        if re.search(r"\b(admission|admissions|course fee|tuition|eligibility|brochure)\b", q):
            return IntentDecision(Intent.ADMISSION_INFORMATION, "hybrid", "get_courses_and_admission_documents")
        if re.search(r"\b(college|campus|contact|address|about kit|today)\b", q):
            return IntentDecision(Intent.COLLEGE_INFORMATION, "kit_api", "get_college_information", "GET", "/v1/college")
        if re.search(r"\b(lesson|explain|doubt|unit|chapter|handbook|brochure|document|circular|pdf|rulebook|rule book)\b", q):
            return IntentDecision(Intent.LESSON_DOUBT, "chroma", "search_kit_documents")
        if re.search(r"\b(hello|hi|hey|thanks|thank you|good morning|good evening)\b", q):
            return IntentDecision(Intent.GENERAL_CONVERSATION, "gemini", "generate_text")
        if len(q.split()) < 3 or q.endswith("?"):
            return IntentDecision(Intent.UNKNOWN, "none")
        return IntentDecision(Intent.GENERAL_CONVERSATION, "gemini", "generate_text")


supervisor = SupervisorAgent()
