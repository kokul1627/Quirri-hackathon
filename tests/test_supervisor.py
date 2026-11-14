from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend import app
from backend.services.supervisor import Intent, supervisor


STUDENT = {"id": 12, "role": "student", "token": "server-session"}
TEACHER = {"id": 31, "role": "teacher", "token": "server-session"}
CHUNK = {"text": "Attendance rules are described here.", "metadata": {"source": "kit_student_handbook_2026.pdf", "page": 11}, "distance": 0.2}


class SupervisorRoutingTests(unittest.TestCase):
    def run_async(self, coroutine):
        return asyncio.run(coroutine)

    def test_attendance_requirement_routes_to_documents_not_personal_api(self):
        api = AsyncMock()
        with patch.object(app, "college", api), patch.object(app, "search_documents", return_value=[CHUNK]), patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            result = self.run_async(app.answer_text("What is the B.Tech attendance requirement?", STUDENT, [], "test-policy"))
        api.assert_not_awaited()
        self.assertEqual(result["intent"], Intent.ATTENDANCE_POLICY.value)
        self.assertEqual(result["source"], "chroma")
        self.assertEqual(result["citations"], [{"source": "kit_student_handbook_2026.pdf", "page": 11}])

    def test_my_attendance_uses_authenticated_attendance_endpoint(self):
        api = AsyncMock(return_value={"overall_percentage": 76, "total_classes_attended": 38, "total_classes_held": 50})
        with patch.object(app, "college", api), patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            result = self.run_async(app.answer_text("What is my attendance percentage?", STUDENT, [], "test-attendance"))
        api.assert_awaited_once_with("GET", "/v1/me/attendance", token=STUDENT["token"], params=None)
        self.assertIn("76%", result["answer"])

    def test_od_policy_uses_rag(self):
        decision = supervisor.classify("How do I apply for OD?", "student")
        self.assertEqual(decision.intent, Intent.OD_POLICY)
        self.assertEqual(decision.source, "chroma")

    def test_od_submission_waits_for_confirmation_before_post(self):
        state = {}
        def load(cid, user): return state.get(cid)
        def save(cid, user, value): state[cid] = value
        def delete(cid, user, value=None): state.pop(cid, None)
        prepare = AsyncMock(return_value={"pending_id": "draft-1", "calendar_conflicts": []})
        submit = AsyncMock(return_value={"id": 81, "status": "submitted"})
        api = AsyncMock()
        message = "Submit an OD request from 2026-11-20 to 2026-11-21 for conference"
        with patch.object(app, "_load_supervisor_pending", side_effect=load), patch.object(app, "_save_supervisor_pending", side_effect=save), patch.object(app, "_delete_supervisor_pending", side_effect=delete), patch.object(app, "prepare_student_request", prepare), patch.object(app, "submit_student_request", submit), patch.object(app, "college", api):
            draft = self.run_async(app.answer_text(message, STUDENT, [], "od-confirm"))
            self.assertTrue(draft["confirmation_required"])
            submit.assert_not_awaited()
            api.assert_not_awaited()
            completed = self.run_async(app.answer_text("CONFIRM", STUDENT, [], "od-confirm"))
        prepare.assert_awaited_once()
        submit.assert_awaited_once_with("draft-1", STUDENT)
        self.assertFalse(completed["confirmation_required"])
        self.assertEqual(completed["tool_result"]["status"], "submitted")

    def test_od_collects_missing_details_then_requires_confirmation(self):
        state = {}
        def load(cid, user): return state.get(cid)
        def save(cid, user, value): state[cid] = value
        def delete(cid, user, value=None): state.pop(cid, None)
        prepare = AsyncMock(return_value={"pending_id": "draft-2", "calendar_conflicts": []})
        submit = AsyncMock(return_value={"id": 82, "status": "submitted"})
        with patch.object(app, "_load_supervisor_pending", side_effect=load), patch.object(app, "_save_supervisor_pending", side_effect=save), patch.object(app, "_delete_supervisor_pending", side_effect=delete), patch.object(app, "prepare_student_request", prepare), patch.object(app, "submit_student_request", submit):
            prompt = self.run_async(app.answer_text("Submit an OD request", STUDENT, [], "od-details"))
            self.assertIn("start date and end date", prompt["answer"])
            self.assertFalse(prompt["confirmation_required"])
            draft = self.run_async(app.answer_text("2026-11-20 to 2026-11-21 reason: conference", STUDENT, [], "od-details"))
            self.assertTrue(draft["confirmation_required"])
            submit.assert_not_awaited()
            completed = self.run_async(app.answer_text("CONFIRM", STUDENT, [], "od-details"))
        prepare.assert_awaited_once()
        submit.assert_awaited_once_with("draft-2", STUDENT)
        self.assertEqual(completed["tool_result"]["status"], "submitted")

    def test_student_cannot_approve_teacher_request(self):
        result = self.run_async(app.answer_text("Approve request 17", STUDENT, [], "student-approve"))
        self.assertEqual(result["intent"], Intent.TEACHER_REQUEST_APPROVAL.value)
        self.assertIn("authorized teacher", result["answer"])

    def test_teacher_approval_uses_teacher_endpoint_after_confirmation(self):
        state = {}
        def load(cid, user): return state.get(cid)
        def save(cid, user, value): state[cid] = value
        def delete(cid, user, value=None): state.pop(cid, None)
        api = AsyncMock(return_value={"status": "approved"})
        with patch.object(app, "_load_supervisor_pending", side_effect=load), patch.object(app, "_save_supervisor_pending", side_effect=save), patch.object(app, "_delete_supervisor_pending", side_effect=delete), patch.object(app, "college", api):
            draft = self.run_async(app.answer_text("Approve request 17", TEACHER, [], "teacher-approve"))
            self.assertTrue(draft["confirmation_required"])
            api.assert_not_awaited()
            result = self.run_async(app.answer_text("CONFIRM", TEACHER, [], "teacher-approve"))
        api.assert_awaited_once_with("POST", "/v1/teacher/requests/17/approve", token=TEACHER["token"], body={})
        self.assertEqual(result["tool_result"]["status"], "approved")

    def test_timetable_routes_to_student_timetable(self):
        decision = supervisor.classify("What is my timetable tomorrow?", "student")
        self.assertEqual((decision.method, decision.path), ("GET", "/v1/me/timetable"))

    def test_admission_fee_uses_courses_and_brochure(self):
        api = AsyncMock(return_value=[{"code": "BTECH", "annual_fee": 100000}])
        with patch.object(app, "college", api), patch.object(app, "search_documents", return_value=[CHUNK]), patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            result = self.run_async(app.answer_text("What is the admission fee for B.Tech?", None, [], "admission-fee"))
        api.assert_awaited_once_with("GET", "/v1/courses")
        self.assertEqual(result["source"], "hybrid")
        self.assertEqual(result["tool_result"]["courses"][0]["code"], "BTECH")
        self.assertTrue(result["citations"])

    def test_placement_drives_use_student_placement_endpoint(self):
        api = AsyncMock(return_value=[])
        with patch.object(app, "college", api), patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            self.run_async(app.answer_text("Show placement drives", STUDENT, [], "placements"))
        api.assert_awaited_once_with("GET", "/v1/placements/drives", token=STUDENT["token"], params=None)

    def test_gemini_failure_logs_status_without_leaking_credentials(self):
        request = httpx.Request("POST", "https://generativelanguage.googleapis.com/test")
        response = httpx.Response(404, request=request)
        class FailedClient:
            def __init__(self, *args, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def post(self, *args, **kwargs):
                raise httpx.HTTPStatusError("not found", request=request, response=response)
        with patch.object(app.httpx, "AsyncClient", FailedClient), patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}), self.assertLogs("backend.app", level="WARNING") as logs:
            result, error = self.run_async(app.generate_grounded_answer("question", [], "", None))
        self.assertIsNone(result)
        self.assertEqual(error, "gemini_model_not_found")
        self.assertTrue(any("HTTP 404" in message for message in logs.output))
        self.assertTrue(all("test-secret" not in message for message in logs.output))

    def test_voice_transcript_and_typed_text_use_same_intent(self):
        typed = supervisor.classify("What is the B.Tech attendance requirement?", "student")
        transcribed = supervisor.classify("What is the B.Tech attendance requirement?", "student")
        self.assertEqual(typed, transcribed)
        self.assertEqual(typed.intent, Intent.ATTENDANCE_POLICY)

    def test_unknown_question_gets_clarification_without_api_call(self):
        result = self.run_async(app.answer_text("What is quantum entanglement?", None, [], "unknown"))
        self.assertEqual(result["intent"], Intent.UNKNOWN.value)
        self.assertIn("not sure", result["answer"])


if __name__ == "__main__":
    unittest.main()
