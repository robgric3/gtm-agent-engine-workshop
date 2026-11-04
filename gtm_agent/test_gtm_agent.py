import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


HAS_MODEL_KEY = bool(os.environ.get("OPENAI_API_KEY"))
with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-only",
    "LANGSMITH_TRACING": "false",
}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class EmailToolTests(unittest.TestCase):
    def test_plain_text_with_attachment_words_sends_unchanged(self):
        body = "No attachment is needed. Please attach your questions to your reply."
        result = agent.send_prospect_email.func(
            prospect={"name": "Prospect", "email": "prospect@example.com"},
            subject="Following up",
            body=body,
            runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}}),
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)
        self.assertEqual(result["to"], "prospect@example.com")
        self.assertEqual(result["from"], "ava.mills@northpoint.com")


@unittest.skipUnless(HAS_MODEL_KEY, "Requires OPENAI_API_KEY for agent behavior tests")
class EmailAgentTests(unittest.TestCase):
    def run_request(self, request):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            return agent.run_agent(request, user_id="rep_sbrown")

    def sent_messages(self, result):
        return [
            json.loads(message.content)
            for message in result["messages"]
            if message.type == "tool" and message.name == "send_prospect_email"
        ]

    def assert_no_attachment_claim(self, text):
        self.assertNotRegex(
            text.lower(),
            r"(?:i(?:'ve| have)?|we(?:'ve| have)?) attached|"
            r"(?:deck|file|document) (?:is |was |has been )?(?:attached|delivered)|"
            r"please (?:find|see) (?:the )?attached",
        )

    def test_pricing_deck_requires_clarification_before_sending(self):
        result = self.run_request("Email LEAD-50003 to send over our pricing deck.")
        self.assertEqual(self.sent_messages(result), [])
        reply = result["reply"].lower()
        self.assertRegex(reply, r"(?:cannot|can't|unable|unsupported|does not|doesn't|not support)")
        self.assertIn("attach", reply)
        self.assertRegex(reply, r"text[ -]only|plain[ -]text")
        self.assertIn("link", reply)
        self.assertIn("?", reply)
        self.assert_no_attachment_claim(reply)

    def test_plain_text_email_sends(self):
        result = self.run_request(
            "Email LEAD-50003 asking whether Tuesday works for a call. "
            "Keep it text-only, with no files or links."
        )
        sent = self.sent_messages(result)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["status"], "sent")
        self.assertIn("tuesday", sent[0]["body"].lower())
        self.assertRegex(result["reply"].lower(), r"sent|emailed")
        self.assert_no_attachment_claim(sent[0]["body"])
        self.assert_no_attachment_claim(result["reply"])

    def test_rep_provided_link_sends(self):
        link = "https://example.com/pricing-deck"
        result = self.run_request(
            f"Email LEAD-50003 our pricing deck using this link: {link}."
        )
        sent = self.sent_messages(result)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["status"], "sent")
        self.assertIn(link, sent[0]["body"])
        self.assertRegex(result["reply"].lower(), r"sent|emailed")
        self.assert_no_attachment_claim(sent[0]["body"])
        self.assert_no_attachment_claim(result["reply"])


if __name__ == "__main__":
    unittest.main()
