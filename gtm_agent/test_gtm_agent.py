import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch


with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})

    def send(self, prospect, **kwargs):
        return agent.send_prospect_email.func(
            prospect, "Introduction", "Hello", self.runtime, **kwargs
        )

    def test_disqualified_prospect_is_blocked(self):
        prospect = agent.get_prospect.invoke({"prospect_id": "LEAD-50001"})["prospect"]
        with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Message constructed")):
            result = self.send(prospect)
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
        })

    def test_caller_cannot_override_authoritative_disqualification(self):
        for flag in (False, None, True):
            with self.subTest(disqualified=flag):
                prospect = {"prospect_id": "LEAD-50001", "email": "prospect@example.com"}
                if flag is not None:
                    prospect["disqualified"] = flag
                self.assertEqual(self.send(prospect)["status"], "blocked")

    def test_missing_record_stops_before_message_construction(self):
        for prospect_id in ("missing", None):
            with self.subTest(prospect_id=prospect_id):
                with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Message constructed")):
                    result = self.send({"prospect_id": prospect_id, "email": "prospect@example.com"})
                self.assertEqual(result, {
                    "status": "failed",
                    "error": "Prospect record was not found; email not sent.",
                })

    def test_eligible_prospect_preserves_send_result_despite_caller_flag(self):
        prospect = agent.get_prospect.invoke({"prospect_id": "LEAD-71001"})["prospect"]
        for flag in (False, True):
            with self.subTest(disqualified=flag):
                result = self.send({**prospect, "disqualified": flag})
                self.assertTrue(result.pop("message_id").startswith("msg-"))
                self.assertEqual(result, {
                    "status": "sent",
                    "to": prospect["email"],
                    "to_name": prospect["name"],
                    "from": "ava.mills@northpoint.com",
                    "from_name": "Ava Mills",
                    "subject": "Introduction",
                    "body": "Hello",
                })

    def test_eligible_prospect_preserves_explicit_sender(self):
        prospect = agent.get_prospect.invoke({"prospect_id": "LEAD-71001"})["prospect"]
        result = self.send(prospect, from_rep={"name": "Rep", "email": "rep@example.com"})
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["from"], "rep@example.com")
        self.assertEqual(result["from_name"], "Rep")

    def test_eligible_prospect_without_email_still_fails(self):
        self.assertEqual(self.send({"prospect_id": "LEAD-71001"}), {
            "status": "failed",
            "error": "Prospect record has no email address.",
        })
